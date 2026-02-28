"""
Vision Transformer Model Compression Trainer

Complete hierarchical reinforcement learning based compression for ViTs.
Implements multi-cycle HRL with ViT-specific kernel identification,
Fisher information sensitivity analysis, and neural surrogate model.
"""

import copy
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import timm

from hirelc_package.config import ExperimentConfig, CompressionGoal, LayerBudget, LayerConfig, KernelConfig
from hirelc_package.trainers.base_trainer import BaseCompressionTrainer
from hirelc_package.utils import ExperimentLogger, ReproducibilityManager, DataManager
from hirelc_package.core.sensitivity import FisherSensitivityEstimator
from hirelc_package.core.surrogate import SurrogateModelTrainer
from hirelc_package.core.quantization import AdvancedQuantizer
from hirelc_package.core.pruning import AdvancedPruner
from hirelc_package.agents.high_level_agent import BudgetAllocationEnvironment, EnsembleHighLevelAgent
from hirelc_package.agents.low_level_agent import BudgetConstrainedCompressionEnv, OptimizedEnsembleLowLevelAgent


class ViTCompressionTrainer(BaseCompressionTrainer):
    """
    Hierarchical Reinforcement Learning based Vision Transformer compression trainer.
    
    ViT-specific features:
    - Identifies 4 kernel types per Transformer block: qkv, attn_proj, mlp_fc1, mlp_fc2
    - Uses Fisher information for sensitivity analysis
    - Multi-cycle HRL with budget allocation and parameter selection
    - Neural surrogate for fast accuracy prediction
    """
    
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        """Initialize ViT compression trainer."""
        super().__init__(config, logger)
        self.logger.log("=" * 80)
        self.logger.log("Initializing ViTCompressionTrainer")
        self.logger.log("=" * 80)
        
        # Load model and data
        self._load_model()
        self._load_data()
        
        # Get baseline accuracy
        self.baseline_accuracy = self.evaluate(self.model)
        self.logger.log(f"Baseline Model Accuracy: {self.baseline_accuracy:.2f}%")
        
        # Store original model for reference
        self.original_model = copy.deepcopy(self.model)
        
        # Initialize surrogate if enabled
        self.surrogate_trainer = None
        self.surrogate_model = None
        if self.config.use_surrogate:
            self._init_surrogate()
        
        # Storage for results
        self.cycle_results = []
        self.sensitivity_scores = {}
        self.all_configs = {}
        self.model_blocks = self._get_vit_blocks()
        self.num_blocks = len(self.model_blocks)
        self.best_global_config = None
        self.best_global_acc = 0.0
        
        self.logger.log(f"ViT Model Loaded: {self.config.model_name}")
        self.logger.log(f"Number of Blocks: {self.num_blocks}")
        self.logger.log(f"Device: {self.config.device}")
        self.logger.log(f"Compression Goal: {1/self.config.compression_goal.target_compression_ratio:.1f}x compression")
    
    def _load_model(self) -> None:
        """Load ViT model using timm."""
        self.logger.log(f"Loading model: {self.config.model_name}")
        try:
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=True,
                num_classes=self.config.num_classes
            )
        except Exception as e:
            self.logger.log(f"Failed to load {self.config.model_name}: {e}", level='ERROR')
            raise
        
        self.model = self.model.to(self.config.device)
        self.model.eval()
    
    def _load_data(self) -> None:
        """Load dataset for training and evaluation."""
        self.logger.log(f"Loading dataset: {self.config.dataset}")
        try:
            if self.config.dataset.lower() == 'tinyimagenet':
                self.train_loader, self.test_loader = DataManager.get_tinyimagenet(
                    batch_size=self.config.batch_size
                )
            elif self.config.dataset.lower() == 'cifar10':
                self.train_loader, self.test_loader = DataManager.get_cifar10(
                    batch_size=self.config.batch_size
                )
            elif self.config.dataset.lower() == 'cifar100':
                self.train_loader, self.test_loader = DataManager.get_cifar100(
                    batch_size=self.config.batch_size
                )
            else:
                raise ValueError(f"Unsupported dataset: {self.config.dataset}")
        except Exception as e:
            self.logger.log(f"Error loading dataset: {e}", level='ERROR')
            raise
        
        self.logger.log(f"Train batches: {len(self.train_loader)}, Test batches: {len(self.test_loader)}")
    
    def _init_surrogate(self) -> None:
        """Initialize surrogate model for fast accuracy prediction."""
        self.logger.log("Initializing surrogate model...")
        self.surrogate_trainer = SurrogateModelTrainer(
            num_blocks=self.num_blocks,
            num_kernels_per_block=4,
            hidden_dims=self.config.surrogate_hidden_dims,
            device=self.config.device,
            baseline_accuracy=self.baseline_accuracy,
            logger=self.logger
        )
    
    def _get_vit_blocks(self) -> List[List[Tuple[str, nn.Module]]]:
        """Extract ViT blocks with 4 kernel types: qkv, attn_proj, mlp_fc1, mlp_fc2."""
        blocks = []
        
        # For Vision Transformer models, extract transformer blocks
        if hasattr(self.model, 'blocks'):
            transformer_blocks = self.model.blocks
        elif hasattr(self.model, 'encoder'):
            if hasattr(self.model.encoder, 'layer'):
                transformer_blocks = self.model.encoder.layer
            else:
                transformer_blocks = self.model.encoder.blocks
        else:
            transformer_blocks = []
        
        for block_idx, block in enumerate(transformer_blocks):
            if block_idx >= 12:  # Limit to 12 blocks
                break
            
            block_layers = []
            
            # Extract the 4 key projections from Transformer block
            # 1. QKV projection (query-key-value)
            if hasattr(block.attn, 'qkv'):
                block_layers.append((f"block_{block_idx}.attn.qkv", block.attn.qkv))
            
            # 2. Output projection of attention (attn_proj)
            if hasattr(block.attn, 'proj'):
                block_layers.append((f"block_{block_idx}.attn.proj", block.attn.proj))
            
            # 3. First FC in MLP (mlp_fc1)
            if hasattr(block.mlp, 'fc1'):
                block_layers.append((f"block_{block_idx}.mlp.fc1", block.mlp.fc1))
            
            # 4. Second FC in MLP (mlp_fc2)
            if hasattr(block.mlp, 'fc2'):
                block_layers.append((f"block_{block_idx}.mlp.fc2", block.mlp.fc2))
            
            # If less than 4 kernels found, add padding
            while len(block_layers) < 4:
                # Use dummy modules as padding
                block_layers.append((f"block_{block_idx}.padding_{len(block_layers)}", nn.Identity()))
            
            blocks.append(block_layers[:4])
        
        # Pad to 12 blocks if needed
        while len(blocks) < 12:
            blocks.append([(f"padding_{len(blocks)}_{i}", nn.Identity()) for i in range(4)])
        
        return blocks[:12]
    
    def evaluate(self, model: nn.Module = None, max_batches: int = None) -> float:
        """Evaluate model accuracy on test set."""
        if model is None:
            model = self.model
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
    
    def compute_sensitivity(self) -> Dict[int, float]:
        """Compute Fisher information sensitivity scores for each ViT block."""
        self.logger.log("\nComputing layer sensitivity scores...")
        
        estimator = FisherSensitivityEstimator(
            model=self.model,
            dataloader=self.test_loader,
            device=self.config.device,
            num_blocks=self.num_blocks
        )
        
        self.sensitivity_scores = estimator.estimate_importance(num_samples=50)
        
        # Normalize scores to [0, 1]
        if self.sensitivity_scores:
            min_score = min(self.sensitivity_scores.values())
            max_score = max(self.sensitivity_scores.values())
            
            if max_score > min_score:
                self.sensitivity_scores = {
                    k: (v - min_score) / (max_score - min_score)
                    for k, v in self.sensitivity_scores.items()
                }
        
        self.logger.log("Sensitivity scores computed:")
        for block_idx in sorted(self.sensitivity_scores.keys()):
            score = self.sensitivity_scores.get(block_idx, 0.0)
            self.logger.log(f"  Block {block_idx}: {score:.4f}")
        
        return self.sensitivity_scores
    
    def pretrain_surrogate(self, num_random_samples: int = 20) -> None:
        """Generate random training samples and pretrain surrogate model."""
        if self.surrogate_trainer is None:
            return
        
        self.logger.log(f"\nPretraining surrogate with {num_random_samples} random samples...")
        
        for sample_idx in range(num_random_samples):
            self.logger.log(f"Surrogate pretraining sample {sample_idx + 1}/{num_random_samples}")
            
            random_config = self._generate_random_config()
            
            model_copy = copy.deepcopy(self.model)
            self.apply_configs_to_model(model_copy, random_config)
            
            accuracy = self.evaluate(model_copy, max_batches=10)
            self.surrogate_trainer.add_sample(random_config, accuracy)
        
        self.surrogate_trainer.train(epochs=30, batch_size=8)
        self.logger.log("Surrogate pretraining completed")
    
    def _generate_random_config(self) -> Dict[int, LayerConfig]:
        """Generate a random compression configuration for ViT blocks."""
        config = {}
        
        # ViT-specific kernel names
        kernel_names = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
        
        for block_idx in range(self.num_blocks):
            layer_config = LayerConfig(block_idx=block_idx)
            
            for k_idx, k_name in enumerate(kernel_names):
                kernel_config = KernelConfig(
                    weight_bits=np.random.randint(2, 9),
                    pruning_ratio=np.random.uniform(0.5, 1.0),
                    quant_type=np.random.choice(['INT', 'FLOAT']),
                    quant_mode=np.random.choice(['uniform', 'log', 'per-channel', 'learned'])
                )
                layer_config.kernels[k_name] = kernel_config
            
            config[block_idx] = layer_config
        
        return config
    
    def create_hla(self, sensitivity_scores: Dict[int, float]):
        """Initialize high-level agent ensemble for ViT."""
        self.logger.log("\nCreating High-Level Agents (HLA)...")
        
        agents = []
        algorithms = self.config.rl_algorithms[:self.config.num_hla_agents]
        
        if len(algorithms) < self.config.num_hla_agents:
            algorithms += ['PPO'] * (self.config.num_hla_agents - len(algorithms))
        
        for agent_idx in range(self.config.num_hla_agents):
            algo = algorithms[agent_idx].upper()
            
            env = BudgetAllocationEnvironment(
                model=self.model,
                eval_dataloader=self.test_loader,
                sensitivity_scores=sensitivity_scores,
                global_goal=self.config.compression_goal,
                device=self.config.device,
                num_blocks=self.num_blocks
            )
            
            from stable_baselines3 import PPO, A2C
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            vec_env = DummyVecEnv([lambda: env])
            
            if algo == 'A2C':
                agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4)
            else:
                agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4, n_steps=512)
            
            agents.append(agent)
            self.logger.log(f"  Created HLA {agent_idx}: {algo}")
        
        return agents, algorithms
    
    def train_hla(self, hla_agents: List, hla_algorithms: List,
                  num_steps: int = None, cycle: int = 0) -> Dict[int, LayerBudget]:
        """Train HLA agents and extract budget allocation."""
        if num_steps is None:
            num_steps = self.config.hla_timesteps
        
        self.logger.log(f"\nTraining HLA agents (Cycle {cycle}, {num_steps} steps)...")
        
        for agent_idx, (agent, algo) in enumerate(zip(hla_agents, hla_algorithms)):
            self.logger.log(f"Training HLA {agent_idx} ({algo})...")
            agent.learn(total_timesteps=num_steps)
        
        agent = hla_agents[0]
        env = agent.get_env().envs[0]
        
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        
        budgets = self._decode_hla_action(action, env)
        
        self.logger.log("HLA training completed. Budget allocation:")
        for block_idx, budget in budgets.items():
            self.logger.log(f"  Block {block_idx}: Ratio={budget.target_compression_ratio:.3f}")
        
        return budgets
    
    def _decode_hla_action(self, action: np.ndarray, env) -> Dict[int, LayerBudget]:
        """Decode HLA action to layer budgets."""
        if hasattr(env, '_decode_action'):
            return env._decode_action(action)
        
        budgets = {}
        compression_levels = {0: 0.35, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15}
        
        for block_idx in range(self.num_blocks):
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            comp_level = int(np.clip(action[block_idx * 2] if block_idx * 2 < len(action) else 2, 0, 4))
            
            budget = LayerBudget(
                block_idx=block_idx,
                target_compression_ratio=compression_levels[comp_level],
                max_accuracy_drop=self.config.compression_goal.target_accuracy_drop / self.num_blocks,
                priority=1.0 - sensitivity,
                sensitivity=sensitivity,
                global_min_bits=self.config.compression_goal.min_layer_bits,
                global_max_bits=self.config.compression_goal.max_layer_bits,
                global_min_pruning=self.config.compression_goal.min_layer_pruning,
                global_max_pruning=self.config.compression_goal.max_layer_pruning
            )
            budgets[block_idx] = budget
        
        return budgets
    
    def create_lla_with_budget(self, block_idx: int, budget: LayerBudget,
                              sensitivity: float):
        """Create ensemble low-level agents for a ViT block with budget constraint."""
        agents = []
        algorithms = self.config.rl_algorithms[:self.config.num_lla_agents]
        
        if len(algorithms) < self.config.num_lla_agents:
            algorithms += ['PPO'] * (self.config.num_lla_agents - len(algorithms))
        
        for agent_idx in range(self.config.num_lla_agents):
            algo = algorithms[agent_idx].upper()
            
            env = BudgetConstrainedCompressionEnv(
                model=self.model,
                dataloader=self.train_loader,
                eval_dataloader=self.test_loader,
                block_idx=block_idx,
                sensitivity_score=sensitivity,
                global_goal=self.config.compression_goal,
                device=self.config.device,
                layer_budget=budget,
                surrogate_model=self.surrogate_model,
                model_blocks=self.model_blocks,
                quantization_type=self.config.quantization_type,
                default_strategy=self.config.default_strategy
            )
            
            from stable_baselines3 import PPO, A2C
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            vec_env = DummyVecEnv([lambda e=env: e])
            
            if algo == 'A2C':
                agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=1e-3)
            else:
                agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=1e-3, n_steps=512)
            
            agents.append(agent)
        
        return agents, algorithms
    
    def apply_configs_to_model(self, model: nn.Module,
                               configs: Dict[int, LayerConfig]) -> None:
        """Apply quantization and pruning from configurations to ViT blocks."""
        model.eval()
        
        kernel_names = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
        
        for block_idx, layer_config in configs.items():
            if block_idx >= len(self.model_blocks):
                continue
            
            block_layers = self.model_blocks[block_idx]
            
            for k_idx, (name, module) in enumerate(block_layers):
                if k_idx >= len(kernel_names):
                    break
                
                k_name = kernel_names[k_idx]
                
                if k_name not in layer_config.kernels:
                    continue
                
                kernel_config = layer_config.kernels[k_name]
                
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                
                # Apply pruning
                if kernel_config.pruning_ratio < 1.0:
                    pruned_weight = AdvancedPruner.prune_magnitude(
                        module.weight.data,
                        pruning_ratio=1.0 - kernel_config.pruning_ratio
                    )
                    module.weight.data = pruned_weight
                
                # Apply quantization
                if kernel_config.weight_bits < 32:
                    if kernel_config.quant_mode == 'uniform':
                        q_weight = AdvancedQuantizer.quantize_uniform(
                            module.weight.data,
                            bits=kernel_config.weight_bits
                        )
                    elif kernel_config.quant_mode == 'log':
                        q_weight = AdvancedQuantizer.quantize_logarithmic(
                            module.weight.data,
                            bits=kernel_config.weight_bits
                        )
                    elif kernel_config.quant_mode == 'per-channel':
                        q_weight = AdvancedQuantizer.quantize_per_channel(
                            module.weight.data,
                            bits=kernel_config.weight_bits
                        )
                    else:
                        q_weight, _, _ = AdvancedQuantizer.quantize_learned(
                            module.weight.data,
                            bits=kernel_config.weight_bits
                        )
                    
                    module.weight.data = q_weight
    
    def run_hierarchical_compression(self) -> Dict[int, LayerConfig]:
        """
        Execute multi-cycle hierarchical reinforcement learning compression.
        
        Returns:
            Final LayerConfig dictionary
        """
        self.logger.log("\n" + "="*80)
        self.logger.log("STARTING HIERARCHICAL COMPRESSION (ViT)")
        self.logger.log("="*80)
        
        # Phase 1: Sensitivity Analysis
        self.logger.log("\nPhase 1: Sensitivity Analysis")
        self.compute_sensitivity()
        
        # Phase 2: Surrogate Pretraining
        if self.config.use_surrogate:
            self.logger.log("\nPhase 2: Surrogate Pretraining")
            self.pretrain_surrogate(num_random_samples=10)
        
        # Phase 3: Multi-cycle HRL
        self.logger.log("\nPhase 3: Multi-cycle Hierarchical Compression")
        
        num_cycles = 3
        
        for cycle in range(num_cycles):
            self.logger.log(f"\n{'='*80}")
            self.logger.log(f"CYCLE {cycle + 1}/{num_cycles}")
            self.logger.log(f"{'='*80}\n")
            
            cycle_start_time = time.time()
            
            # Create and train HLA
            hla_agents, hla_algorithms = self.create_hla(self.sensitivity_scores)
            
            # Update HLA with feedback from previous cycle
            if cycle > 0 and len(self.cycle_results) > 0:
                prev_result = self.cycle_results[-1]
                current_acc_drop = prev_result['accuracy_drop']
                current_compression = prev_result['compression_ratio']
                progress = cycle / float(num_cycles)
                
                # Pass feedback to HLA environments
                for hla_dict in hla_agents:
                    for env in hla_dict['agent'].get_env().envs:
                        env.update_feedback(current_acc_drop, current_compression, progress)
                
                self.logger.log(f"HLA feedback updated: AccDrop={current_acc_drop:.2f}%, CompressionRatio={current_compression:.3f}")
            
            budgets = self.train_hla(hla_agents, hla_algorithms, cycle=cycle + 1)
            
            # Extract configurations from LLA
            cycle_config = {}
            
            for block_idx, budget in budgets.items():
                self.logger.log(f"\nProcessing Block {block_idx}...")
                sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
                
                # Create and train LLA
                lla_agents, lla_algorithms = self.create_lla_with_budget(
                    block_idx, budget, sensitivity
                )
                
                # Train LLA
                for agent_idx, (agent, algo) in enumerate(zip(lla_agents, lla_algorithms)):
                    self.logger.log(f"  Training LLA {agent_idx} ({algo})...")
                    agent.learn(total_timesteps=self.config.lla_timesteps)
                
                # Extract best configuration
                best_agent = lla_agents[0]
                env = best_agent.get_env().envs[0]
                obs, _ = env.reset()
                action, _ = best_agent.predict(obs, deterministic=True)
                
                if hasattr(env, '_decode_action'):
                    layer_config = env._decode_action(action)
                else:
                    layer_config = LayerConfig(block_idx=block_idx)
                    kernel_names = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
                    for k_name in kernel_names:
                        layer_config.kernels[k_name] = KernelConfig(
                            weight_bits=max(4, int(budget.global_max_bits * (1 - budget.target_compression_ratio))),
                            pruning_ratio=0.8,
                            quant_type='INT',
                            quant_mode='uniform'
                        )
                
                cycle_config[block_idx] = layer_config
                
                # Cleanup
                del lla_agents, env
                gc.collect()
            
            # Apply compression
            self.logger.log("\nApplying compression configuration...")
            self.apply_configs_to_model(self.model, cycle_config)
            
            # Evaluate compressed model
            compressed_acc = self.evaluate(self.model, max_batches=50)
            self.logger.log(f"Compressed Model Accuracy: {compressed_acc:.2f}%")
            
            # Fine-tune
            if self.config.do_finetune:
                self.logger.log("Fine-tuning compressed model...")
                finetuned_acc = self.finetune_with_early_stopping(
                    self.model,
                    max_epochs=self.config.finetune_epochs,
                    lr=self.config.lr
                )
                final_acc = finetuned_acc
            else:
                final_acc = compressed_acc
            
            # Calculate compression ratio
            original_params = sum(p.numel() for p in self.original_model.parameters())
            compressed_params = sum(p.numel() for p in self.model.parameters() if p.data.abs().sum() > 0)
            compression_ratio = compressed_params / original_params if original_params > 0 else 1.0
            
            # Update surrogate
            if self.surrogate_trainer is not None:
                self.logger.log("Updating surrogate model...")
                self.surrogate_trainer.add_sample(cycle_config, final_acc)
                self.surrogate_trainer.train(epochs=10, batch_size=8)
            
            # Log results
            cycle_time = time.time() - cycle_start_time
            cycle_result = {
                'cycle': cycle + 1,
                'baseline_accuracy': self.baseline_accuracy,
                'compressed_accuracy': compressed_acc,
                'final_accuracy': final_acc,
                'compression_ratio': compression_ratio,
                'accuracy_drop': self.baseline_accuracy - final_acc,
                'time': cycle_time
            }
            
            self.cycle_results.append(cycle_result)
            self.all_configs[cycle] = cycle_config
            self.best_global_config = cycle_config
            self.best_global_acc = final_acc
            
            self.logger.log_cycle_result(cycle + 1, cycle_result)
            
            # Cleanup
            del hla_agents
            gc.collect()
        
        # Final logging
        self.logger.log("\n" + "="*80)
        self.logger.log("COMPRESSION COMPLETE (ViT)")
        self.logger.log(f"Final Accuracy: {self.best_global_acc:.2f}%")
        self.logger.log(f"Accuracy Drop: {self.baseline_accuracy - self.best_global_acc:.2f}%")
        self.logger.log("="*80)
        
        return self.best_global_config


def create_vit_trainer(model_name: str = "vit_base_patch16_224",
                       dataset: str = "cifar10",
                       num_classes: int = 10,
                       compression_goal: Optional[CompressionGoal] = None,
                       experiment_name: str = "vit_compression",
                       device: str = "cuda") -> ViTCompressionTrainer:
    """
    Create a ViTCompressionTrainer with default or custom configuration.
    
    Args:
        model_name: ViT model name (timm supported)
        dataset: Dataset name (cifar10, cifar100, tinyimagenet)
        num_classes: Number of classes
        compression_goal: Custom compression goal (default if None)
        experiment_name: Experiment name for logging
        device: Device to use (cuda or cpu)
    
    Returns:
        Initialized ViTCompressionTrainer
    """
    ReproducibilityManager.set_seed(42)
    
    if compression_goal is None:
        compression_goal = CompressionGoal(
            target_accuracy_drop=1.0,
            target_compression_ratio=0.25,
            min_layer_bits=2,
            max_layer_bits=8,
        )
    
    config = ExperimentConfig(
        model_name=model_name,
        dataset=dataset,
        num_classes=num_classes,
        compression_goal=compression_goal,
        experiment_name=experiment_name,
        device=device,
    )
    
    logger = ExperimentLogger(experiment_name, output_dir='./outputs')
    
    return ViTCompressionTrainer(config, logger)
