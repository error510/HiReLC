"""
CNN Model Compression Trainer

Complete hierarchical reinforcement learning based compression for CNNs.
Implements multi-cycle HRL with Fisher information sensitivity analysis,
high-level budget allocation, low-level parameter selection, and
neural surrogate for fast accuracy prediction.
"""

import copy
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Timm for model loading
import timm

from hirelc_package.config import ExperimentConfig, CompressionGoal, LayerBudget, LayerConfig, KernelConfig
from hirelc_package.trainers.base_trainer import BaseCompressionTrainer
from hirelc_package.utils import ExperimentLogger, ReproducibilityManager, DataManager, ComprehensiveVisualizer
from hirelc_package.core.sensitivity import FisherSensitivityEstimator
from hirelc_package.core.surrogate import AccuracySurrogateModel, SurrogateModelTrainer
from hirelc_package.core.quantization import AdvancedQuantizer
from hirelc_package.core.pruning import AdvancedPruner
from hirelc_package.agents.high_level_agent import BudgetAllocationEnvironment, EnsembleHighLevelAgent
from hirelc_package.agents.low_level_agent import BudgetConstrainedCompressionEnv, OptimizedEnsembleLowLevelAgent



class CNNCompressionTrainer(BaseCompressionTrainer):
    """
    Hierarchical Reinforcement Learning based CNN compression trainer.
    
    Multi-cycle compression pipeline:
    1. Compute Fisher information sensitivity
    2. Pretrain neural surrogate model
    3. For each cycle:
       - HLA allocates per-layer compression budgets
       - LLA selects bitwidth/pruning/quantization type per kernel
       - Apply compression, fine-tune, evaluate
       - Update surrogate with results
    4. Generate comprehensive visualizations
    """
    
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        """Initialize CNN compression trainer."""
        super().__init__(config, logger)
        self.logger.log("=" * 80)
        self.logger.log("Initializing CNNCompressionTrainer")
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
        self.model_blocks = self._get_model_blocks()
        self.num_blocks = len(self.model_blocks)
        self.best_global_config = None
        self.best_global_acc = 0.0
        
        self.logger.log(f"CNN Model Loaded: {self.config.model_name}")
        self.logger.log(f"Number of Blocks: {self.num_blocks}")
        self.logger.log(f"Device: {self.config.device}")
    
    def _load_model(self) -> None:
        """Load CNN model using timm."""
        self.logger.log(f"Loading model: {self.config.model_name}")
        try:
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=True,
                num_classes=self.config.num_classes
            )
        except Exception as e:
            self.logger.log(f"Failed to load {self.config.model_name} from timm: {e}")
            try:
                import torchvision.models as models
                model_loader = getattr(models, self.config.model_name, None)
                if model_loader is None:
                    raise ValueError(f"Model {self.config.model_name} not found")
                self.model = model_loader(pretrained=True)
                if hasattr(self.model, 'fc'):
                    in_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features, self.config.num_classes)
                elif hasattr(self.model, 'classifier'):
                    in_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = nn.Linear(in_features, self.config.num_classes)
            except Exception as e2:
                self.logger.log(f"Failed to load model: {e2}", level='ERROR')
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
    
    def _get_model_blocks(self) -> List[List[Tuple[str, nn.Module]]]:
        """Extract model blocks in hierarchical structure."""
        blocks = []
        
        # Try standard ResNet structure
        for block_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(self.model, block_name):
                block = getattr(self.model, block_name)
                block_layers = []
                
                for name, module in block.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        block_layers.append((f"{block_name}.{name}", module))
                
                if block_layers:
                    blocks.append(block_layers)
        
        # If no standard blocks found, group by module
        if not blocks:
            current_block = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    current_block.append((name, module))
                    if len(current_block) >= 4:
                        blocks.append(current_block)
                        current_block = []
            
            if current_block:
                blocks.append(current_block)
        
        # Pad blocks if needed
        while len(blocks) < 12:
            blocks.append([])
        
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
        """Compute Fisher information sensitivity scores for each layer."""
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
        """Generate a random compression configuration."""
        config = {}
        
        for block_idx in range(self.num_blocks):
            layer_config = LayerConfig(block_idx=block_idx)
            
            for k_idx in range(4):
                kernel_config = KernelConfig(
                    weight_bits=np.random.randint(2, 9),
                    pruning_ratio=np.random.uniform(0.5, 1.0),
                    quant_type=np.random.choice(['INT', 'FLOAT']),
                    quant_mode=np.random.choice(['uniform', 'log', 'per-channel', 'learned'])
                )
                layer_config.kernels[f'kernel_{k_idx}'] = kernel_config
            
            config[block_idx] = layer_config
        
        return config
    
    def create_hla(self, sensitivity_scores: Dict[int, float]):
        """Initialize high-level agent ensemble."""
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
        """Create ensemble low-level agents for a block with budget constraint."""
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
        """Apply quantization and pruning from configurations to model."""
        model.eval()
        
        for block_idx, layer_config in configs.items():
            if block_idx >= len(self.model_blocks):
                continue
            
            block_layers = self.model_blocks[block_idx]
            
            for k_idx, (name, module) in enumerate(block_layers):
                kernel_name = f'kernel_{k_idx}' if f'kernel_{k_idx}' in layer_config.kernels else (list(layer_config.kernels.keys())[0] if layer_config.kernels else None)
                
                if kernel_name is None or kernel_name not in layer_config.kernels:
                    continue
                
                kernel_config = layer_config.kernels[kernel_name]
                
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
        self.logger.log("STARTING HIERARCHICAL COMPRESSION")
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
                    for k_idx in range(4):
                        layer_config.kernels[f'kernel_{k_idx}'] = KernelConfig(
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
        self.logger.log("COMPRESSION COMPLETE")
        self.logger.log(f"Final Accuracy: {self.best_global_acc:.2f}%")
        self.logger.log(f"Accuracy Drop: {self.baseline_accuracy - self.best_global_acc:.2f}%")
        self.logger.log("="*80)
        
        return self.best_global_config
    
    def generate_all_visualizations(self) -> None:
        """Generate comprehensive visualizations of compression results."""
        self.logger.log("Generating visualizations...")
        
        output_dir = Path(self.logger.get_run_dir()) / "visualizations"
        output_dir.mkdir(exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # 1. Per-kernel decisions heatmap
            if self.best_global_config:
                self._plot_kernel_decisions(output_dir)
            
            # 2. Accuracy vs size
            self._plot_pareto_front(output_dir)
            
            # 3. Learning curves
            self._plot_learning_curves(output_dir)
            
            # 4. Sensitivity analysis
            self._plot_sensitivity_analysis(output_dir)
            
            self.logger.log(f"Visualizations saved to: {output_dir}")
        except Exception as e:
            self.logger.log(f"Error generating visualizations: {e}")
    
    def _plot_kernel_decisions(self, output_dir: Path) -> None:
        """Plot per-kernel compression decisions."""
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(self.num_blocks // 3, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for block_idx, ax in enumerate(axes):
                if block_idx >= self.num_blocks:
                    break
                
                if block_idx not in self.best_global_config:
                    ax.text(0.5, 0.5, f"Block {block_idx}\nNo Config", ha='center', va='center')
                    continue
                
                layer_config = self.best_global_config[block_idx]
                kernel_names = list(layer_config.kernels.keys())[:4]
                bits = [layer_config.kernels[k].weight_bits for k in kernel_names]
                prune = [layer_config.kernels[k].pruning_ratio for k in kernel_names]
                
                x = np.arange(len(bits))
                width = 0.35
                ax.bar(x - width/2, bits, width, label='Weight Bits', alpha=0.8)
                ax.bar(x + width/2, [p * 10 for p in prune], width, label='Pruning (×10)', alpha=0.8)
                ax.set_ylabel('Value')
                ax.set_title(f'Block {block_idx}')
                ax.set_xticks(x)
                ax.set_xticklabels([f'K{i}' for i in range(len(bits))])
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / "kernel_decisions.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.log("Saved: kernel_decisions.png")
        except Exception as e:
            self.logger.log(f"Error creating kernel decisions plot: {e}")
    
    def _plot_pareto_front(self, output_dir: Path) -> None:
        """Plot Pareto front: accuracy vs compression."""
        try:
            import matplotlib.pyplot as plt
            baseline_size = sum(p.numel() * 4 for p in self.original_model.parameters()) / (1024 * 1024)
            final_size = sum(p.numel() * 4 for p in self.model.parameters() if p.data.abs().sum() > 0) / (1024 * 1024)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter([baseline_size], [self.baseline_accuracy], s=200, c='red', marker='*', label='Baseline', zorder=5)
            ax.scatter([final_size], [self.best_global_acc], s=200, c='green', marker='o', label='Compressed', zorder=5)
            ax.plot([baseline_size, final_size], [self.baseline_accuracy, self.best_global_acc], 'b--', alpha=0.5)
            
            ax.set_xlabel('Model Size (MB)')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Pareto Front: Accuracy vs Size')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / "pareto_front.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.log("Saved: pareto_front.png")
        except Exception as e:
            self.logger.log(f"Error creating Pareto plot: {e}")
    
    def _plot_learning_curves(self, output_dir: Path) -> None:
        """Plot training/compression learning curves."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.cycle_results:
                return
            
            cycles = [r['cycle'] for r in self.cycle_results]
            accuracies = [r['final_accuracy'] for r in self.cycle_results]
            ratios = [r['compression_ratio'] for r in self.cycle_results]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            axes[0].plot(cycles, accuracies, 'bo-', linewidth=2, markersize=8)
            axes[0].axhline(y=self.baseline_accuracy, color='r', linestyle='--', label='Baseline')
            axes[0].set_xlabel('Cycle')
            axes[0].set_ylabel('Accuracy (%)')
            axes[0].set_title('Model Accuracy Over Cycles')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            axes[1].plot(cycles, ratios, 'go-', linewidth=2, markersize=8)
            axes[1].axhline(y=self.config.compression_goal.target_compression_ratio,
                           color='r', linestyle='--', label='Target')
            axes[1].set_xlabel('Cycle')
            axes[1].set_ylabel('Compression Ratio')
            axes[1].set_title('Compression Ratio Over Cycles')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.log("Saved: learning_curves.png")
        except Exception as e:
            self.logger.log(f"Error creating learning curves: {e}")
    
    def _plot_sensitivity_analysis(self, output_dir: Path) -> None:
        """Plot sensitivity analysis results."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.sensitivity_scores:
                return
            
            blocks = sorted(self.sensitivity_scores.keys())
            scores = [self.sensitivity_scores[b] for b in blocks]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(blocks)))
            bars = ax.bar(blocks, scores, color=colors, edgecolor='black', alpha=0.7)
            
            ax.set_xlabel('Block Index')
            ax.set_ylabel('Sensitivity Score')
            ax.set_title('Layer Sensitivity Scores (Fisher Information)')
            ax.grid(True, axis='y', alpha=0.3)
            
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "sensitivity_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.log("Saved: sensitivity_analysis.png")
        except Exception as e:
            self.logger.log(f"Error creating sensitivity plot: {e}")

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
        self.model_blocks = self._get_model_blocks()
        self.num_blocks = len(self.model_blocks)
        self.best_global_config = None
        self.best_global_acc = 0.0
        
        self.logger.log(f"CNN Model Loaded: {self.config.model_name}")
        self.logger.log(f"Number of Blocks: {self.num_blocks}")
        self.logger.log(f"Device: {self.config.device}")
        self.logger.log(f"Compression Goal: {1/self.config.compression_goal.target_compression_ratio:.1f}x compression")
    
    def _load_model(self) -> None:
        """Load CNN model using timm."""
        self.logger.log(f"Loading model: {self.config.model_name}")
        
        try:
            # Load from timm (torchvision models)
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=True,
                num_classes=self.config.num_classes
            )
        except Exception as e:
            self.logger.log(f"Failed to load {self.config.model_name} from timm: {e}")
            try:
                # Fallback to torchvision
                import torchvision.models as models
                model_loader = getattr(models, self.config.model_name, None)
                if model_loader is None:
                    raise ValueError(f"Model {self.config.model_name} not found")
                self.model = model_loader(pretrained=True)
                # Adjust final layer for num_classes if needed
                if hasattr(self.model, 'fc'):
                    in_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features, self.config.num_classes)
                elif hasattr(self.model, 'classifier'):
                    in_features = self.model.classifier[-1].in_features
                    self.model.classifier[-1] = nn.Linear(in_features, self.config.num_classes)
            except Exception as e2:
                self.logger.log(f"Failed to load model: {e2}", level='ERROR')
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
                self.logger.log(f"Unknown dataset: {self.config.dataset}", level='ERROR')
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
            num_kernels_per_block=4,  # Typical for ResNets
            hidden_dims=self.config.surrogate_hidden_dims,
            device=self.config.device,
            baseline_accuracy=self.baseline_accuracy,
            logger=self.logger
        )
    
    def _get_model_blocks(self) -> List[List[Tuple[str, nn.Module]]]:
        """
        Extract model blocks in a hierarchical structure.
        
        For ResNets: Returns blocks from layer1, layer2, layer3, layer4
        
        Returns:
            List of blocks, each containing layer tuples
        """
        blocks = []
        
        # Try to get standard ResNet structure
        for block_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(self.model, block_name):
                block = getattr(self.model, block_name)
                block_layers = []
                
                for name, module in block.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        block_layers.append((f"{block_name}.{name}", module))
                
                if block_layers:
                    blocks.append(block_layers)
        
        # If no standard blocks found, group by module
        if not blocks:
            current_block = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    current_block.append((name, module))
                    if len(current_block) >= 4:
                        blocks.append(current_block)
                        current_block = []
            
            if current_block:
                blocks.append(current_block)
        
        # Pad blocks if needed
        while len(blocks) < 12:
            blocks.append([])
        
        return blocks[:12]  # Limit to 12 blocks
    
    def evaluate(self, model: nn.Module = None, max_batches: int = None) -> float:
        """
        Evaluate model accuracy on test set.
        
        Args:
            model: Model to evaluate (uses self.model if None)
            max_batches: Maximum number of batches to evaluate (None for all)
        
        Returns:
            Accuracy percentage (0-100)
        """
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
        """
        Compute Fisher information sensitivity scores for each layer.
        
        Returns:
            Dict mapping block index to sensitivity score
        """
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
        """
        Generate random training samples and pretrain surrogate model.
        
        Args:
            num_random_samples: Number of random configurations to generate and evaluate
        """
        if self.surrogate_trainer is None:
            return
        
        self.logger.log(f"\nPretraining surrogate with {num_random_samples} random samples...")
        
        for sample_idx in range(num_random_samples):
            self.logger.log(f"Surrogate pretraining sample {sample_idx + 1}/{num_random_samples}")
            
            # Generate random configuration
            random_config = self._generate_random_config()
            
            # Apply configuration
            model_copy = copy.deepcopy(self.model)
            self.apply_configs_to_model(model_copy, random_config)
            
            # Evaluate
            accuracy = self.evaluate(model_copy, max_batches=10)
            
            # Add to surrogate
            self.surrogate_trainer.add_sample(random_config, accuracy)
        
        # Train surrogate
        self.surrogate_trainer.train(epochs=30, batch_size=8)
        self.logger.log("Surrogate pretraining completed")
    
    def _generate_random_config(self) -> Dict[int, LayerConfig]:
        """Generate a random compression configuration."""
        config = {}
        
        for block_idx in range(self.num_blocks):
            layer_config = LayerConfig(block_idx=block_idx)
            
            # Random kernels (typically 4 per block for ResNets)
            for k_idx in range(4):
                kernel_config = KernelConfig(
                    weight_bits=np.random.randint(2, 9),
                    pruning_ratio=np.random.uniform(0.5, 1.0),
                    quant_type=np.random.choice(['INT', 'FLOAT']),
                    quant_mode=np.random.choice(['uniform', 'log', 'per-channel', 'learned'])
                )
                layer_config.kernels[f'kernel_{k_idx}'] = kernel_config
            
            config[block_idx] = layer_config
        
        return config
    
    def create_hla(self, sensitivity_scores: Dict[int, float]) -> Tuple[List[Any], List[str]]:
        """
        Initialize high-level agent (ensemble of RL agents).
        
        Args:
            sensitivity_scores: Layer sensitivity scores
        
        Returns:
            Tuple of (agents_list, algorithms_list)
        """
        self.logger.log("\nCreating High-Level Agents (HLA)...")
        
        agents = []
        algorithms = self.config.rl_algorithms[:self.config.num_hla_agents]
        
        if len(algorithms) < self.config.num_hla_agents:
            algorithms += ['PPO'] * (self.config.num_hla_agents - len(algorithms))
        
        for agent_idx in range(self.config.num_hla_agents):
            algo = algorithms[agent_idx].upper()
            
            # Create environment
            env = BudgetAllocationEnvironment(
                model=self.model,
                eval_dataloader=self.test_loader,
                sensitivity_scores=sensitivity_scores,
                global_goal=self.config.compression_goal,
                device=self.config.device,
                num_blocks=self.num_blocks
            )
            
            # Create RL agent
            if algo == 'PPO':
                from stable_baselines3 import PPO
                agent = PPO('MlpPolicy', env, verbose=0, 
                           learning_rate=3e-4, n_steps=512)
            elif algo == 'A2C':
                from stable_baselines3 import A2C
                agent = A2C('MlpPolicy', env, verbose=0,
                           learning_rate=3e-4)
            else:
                from stable_baselines3 import PPO
                agent = PPO('MlpPolicy', env, verbose=0)
            
            agents.append(agent)
            self.logger.log(f"  Created HLA {agent_idx}: {algo}")
        
        return agents, algorithms
    
    def train_hla(self, hla_agents: List[Any], hla_algorithms: List[str],
                  num_steps: int = None, cycle: int = 0) -> Dict[int, LayerBudget]:
        """
        Train HLA agents and extract budget allocation.
        
        Args:
            hla_agents: List of HLA agents
            hla_algorithms: List of algorithm names
            num_steps: Number of training steps (uses config.hla_timesteps if None)
            cycle: Current cycle number for logging
        
        Returns:
            Dict mapping block index to LayerBudget
        """
        if num_steps is None:
            num_steps = self.config.hla_timesteps
        
        self.logger.log(f"\nTraining HLA agents (Cycle {cycle}, {num_steps} steps)...")
        
        # Train agents
        for agent_idx, (agent, algo) in enumerate(zip(hla_agents, hla_algorithms)):
            self.logger.log(f"Training HLA {agent_idx} ({algo})...")
            agent.learn(total_timesteps=num_steps)
        
        # Extract budget allocations from first agent
        agent = hla_agents[0]
        env = agent.get_env()
        
        obs, _ = env.reset()
        action, _ = agent.predict(obs, deterministic=True)
        
        # Decode action to budgets
        budgets = self._decode_hla_action(action, env)
        
        self.logger.log("HLA training completed. Budget allocation:")
        for block_idx, budget in budgets.items():
            self.logger.log(f"  Block {block_idx}: Ratio={budget.target_compression_ratio:.3f}")
        
        return budgets
    
    def _decode_hla_action(self, action: np.ndarray, env: Any) -> Dict[int, LayerBudget]:
        """Decode HLA action to layer budgets."""
        if hasattr(env, '_decode_action'):
            return env._decode_action(action)
        
        # Default implementation
        budgets = {}
        compression_levels = {
            0: 0.35, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15
        }
        
        for block_idx in range(self.num_blocks):
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            comp_level = int(np.clip(action[block_idx * 2], 0, 4))
            
            budget = LayerBudget(
                block_idx=block_idx,
                target_compression_ratio=compression_levels[comp_level],
                max_accuracy_drop=self.config.compression_goal.target_accuracy_drop / self.num_blocks,
                priority=1.0 - sensitivity,  # High sensitivity = low priority
                sensitivity=sensitivity,
                global_min_bits=self.config.compression_goal.min_layer_bits,
                global_max_bits=self.config.compression_goal.max_layer_bits,
                global_min_pruning=self.config.compression_goal.min_layer_pruning,
                global_max_pruning=self.config.compression_goal.max_layer_pruning
            )
            budgets[block_idx] = budget
        
        return budgets
    
    def create_lla_with_budget(self, block_idx: int, budget: LayerBudget,
                              sensitivity: float) -> Tuple[List[Any], List[str]]:
        """
        Create ensemble low-level agents for a block with budget constraint.
        
        Args:
            block_idx: Block index
            budget: LayerBudget for this block
            sensitivity: Sensitivity score for this block
        
        Returns:
            Tuple of (agents_list, algorithms_list)
        """
        agents = []
        algorithms = self.config.rl_algorithms[:self.config.num_lla_agents]
        
        if len(algorithms) < self.config.num_lla_agents:
            algorithms += ['PPO'] * (self.config.num_lla_agents - len(algorithms))
        
        for agent_idx in range(self.config.num_lla_agents):
            algo = algorithms[agent_idx].upper()
            
            # Create environment
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
            
            # Create RL agent
            if algo == 'PPO':
                from stable_baselines3 import PPO
                agent = PPO('MlpPolicy', env, verbose=0,
                           learning_rate=1e-3, n_steps=512)
            elif algo == 'A2C':
                from stable_baselines3 import A2C
                agent = A2C('MlpPolicy', env, verbose=0,
                           learning_rate=1e-3)
            else:
                from stable_baselines3 import PPO
                agent = PPO('MlpPolicy', env, verbose=0)
            
            agents.append(agent)
        
        return agents, algorithms
    
    def apply_configs_to_model(self, model: nn.Module,
                               configs: Dict[int, LayerConfig]) -> None:
        """
        Apply quantization and pruning from configurations to model.
        
        Args:
            model: Model to modify
            configs: Dict[block_idx -> LayerConfig]
        """
        model.eval()
        
        for block_idx, layer_config in configs.items():
            if block_idx >= len(self.model_blocks):
                continue
            
            block_layers = self.model_blocks[block_idx]
            
            for k_idx, (name, module) in enumerate(block_layers):
                kernel_name = f'kernel_{k_idx}' if f'kernel_{k_idx}' in layer_config.kernels else list(layer_config.kernels.keys())[0] if layer_config.kernels else None
                
                if kernel_name is None or kernel_name not in layer_config.kernels:
                    continue
                
                kernel_config = layer_config.kernels[kernel_name]
                
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
                    else:  # learned
                        q_weight, _, _ = AdvancedQuantizer.quantize_learned(
                            module.weight.data,
                            bits=kernel_config.weight_bits
                        )
                    
                    module.weight.data = q_weight

