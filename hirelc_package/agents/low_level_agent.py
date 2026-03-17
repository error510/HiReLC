"""
Low-Level RL Agents for per-layer compression parameter selection.

This module implements the ensemble of RL agents that decide compression parameters
for each layer (quantization bitwidth, pruning ratio, quantization type, granularity).
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from hirelc_package.config import CompressionGoal, LayerBudget, KernelConfig, LayerConfig
from hirelc_package.core import AdvancedQuantizer, AdvancedPruner
from hirelc_package.utils import DataManager


# Constants
QUANT_TYPES = ['INT', 'FLOAT']
QUANT_MODES = ['uniform', 'log', 'per-channel', 'learned']


class BudgetConstrainedCompressionEnv(gym.Env):
    """
    Per-block MDP for the Low-Level Agent ensemble.

    Action space (per kernel): [bits_idx, pruning_idx, type_idx, gran_idx]
      bits_idx   : 0-14  → bitwidth b_{i,k}
      pruning_idx: 0-14  → keep-ratio ρ_{i,k}
      type_idx   : 0-1   → τ_{i,k} ∈ {INT, FLOAT}
      gran_idx   : 0-3   → μ_{i,k} ∈ {uniform, log, per-channel, learned}
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader, eval_dataloader: DataLoader,
                 block_idx: int, sensitivity_score: float, global_goal: CompressionGoal,
                 device: str = 'cuda', curriculum_stage: int = 0,
                 layer_budget: Optional[LayerBudget] = None,
                 surrogate_model = None,
                 model_blocks: List[List[Tuple[str, nn.Module]]] = None,
                 quantization_type: str = 'mixed',
                 default_strategy: Optional[str] = None):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.block_idx = block_idx
        self.sensitivity_score = sensitivity_score
        self.global_goal = global_goal
        self.device = device
        self.curriculum_stage = curriculum_stage
        self.surrogate_model = surrogate_model
        self.model_blocks = model_blocks
        self.quantization_type = quantization_type.lower()
        self.default_strategy = default_strategy

        self.layer_budget = layer_budget
        if self.layer_budget is None:
            self.layer_budget = LayerBudget(
                block_idx=block_idx, target_compression_ratio=0.25,
                max_accuracy_drop=1.0, priority=0.5, sensitivity=sensitivity_score,
                global_min_bits=global_goal.min_layer_bits, global_max_bits=global_goal.max_layer_bits,
                global_min_pruning=global_goal.min_layer_pruning, global_max_pruning=global_goal.max_layer_pruning)
        
        self.kernel_modules = self._identify_kernels()
        num_kernels = len(self.kernel_modules)
        
        self.action_space = spaces.MultiDiscrete([15, 15, 2, 4] * num_kernels)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(12 + num_kernels,), dtype=np.float32)
        
        self.cached_inputs = None
        self.baseline_outputs = None
        self.baseline_accuracy = None
        self.original_weights = {}
        self.step_count = 0
        self.reward_history = []
        
        if self.cached_inputs is None:
            self.cached_inputs, _ = DataManager.get_cached_batches(self.dataloader, self.device, num_batches=3)
    
    def _identify_kernels(self) -> Dict[str, nn.Module]:
        """Identify kernels in the block"""
        kernels = {}
        if self.model_blocks is None or self.block_idx >= len(self.model_blocks):
            return kernels
        
        block_layers = self.model_blocks[self.block_idx]
        for idx, (name, module) in enumerate(block_layers):
            kernels[f'kernel{idx+1}'] = module
        return kernels
    
    def update_budget(self, new_budget: LayerBudget):
        """Update layer budget"""
        self.layer_budget = new_budget
    
    def _decode_action(self, action: np.ndarray) -> LayerConfig:
        """Decode action into LayerConfig"""
        layer_config = LayerConfig(block_idx=self.block_idx)
        layer_config.assigned_budget = self.layer_budget
        
        kernel_names = list(self.kernel_modules.keys())
        budget = self.layer_budget
        min_bits = max(budget.min_bits, budget.global_min_bits)
        max_bits = budget.global_max_bits
        max_pruning = min(budget.max_pruning, budget.global_max_pruning)
        min_pruning = budget.global_min_pruning
        
        for kernel_idx, kernel_name in enumerate(kernel_names):
            base_idx = kernel_idx * 4
            bits_idx = action[base_idx]
            pruning_idx = action[base_idx + 1]
            type_idx = int(action[base_idx + 2])
            gran_idx = int(action[base_idx + 3])
            module = self.kernel_modules[kernel_name]

            # Bitwidth
            bits = int(np.clip(
                min_bits + int(bits_idx * (max_bits - min_bits) / 14.0),
                min_bits, max_bits))

            # Keep ratio
            prune_amount = min_pruning + (pruning_idx / 14.0) * (max_pruning - min_pruning)
            keep_ratio = float(np.clip(1.0 - prune_amount, 1.0 - max_pruning, 1.0 - min_pruning))

            # Quantization type
            if self.quantization_type == 'int':
                quant_type = 'INT'
            elif self.quantization_type == 'float':
                quant_type = 'FLOAT'
            else:
                quant_type = QUANT_TYPES[type_idx % len(QUANT_TYPES)]

            # Granularity
            if self.default_strategy and self.default_strategy in QUANT_MODES:
                quant_mode = self.default_strategy
            else:
                quant_mode = QUANT_MODES[gran_idx % len(QUANT_MODES)]

            kernel_config_obj = KernelConfig(
                name=kernel_name, weight_bits=bits, act_bits=bits,
                quant_type=quant_type, quant_mode=quant_mode,
                pruning_ratio=keep_ratio, importance_method='l2',
                shape=tuple(module.weight.shape))
            
            if kernel_name == 'kernel1':
                layer_config.kernel1_config = kernel_config_obj
            elif kernel_name == 'kernel2':
                layer_config.kernel2_config = kernel_config_obj
            elif kernel_name == 'kernel3':
                layer_config.kernel3_config = kernel_config_obj
            elif kernel_name == 'kernel4':
                layer_config.kernel4_config = kernel_config_obj
        
        layer_config.update_aggregates()
        return layer_config
    
    def _compute_baseline_outputs(self):
        """Compute baseline outputs for comparison"""
        if self.baseline_outputs is not None:
            return
        self.model.eval()
        with torch.no_grad():
            self.baseline_outputs = self.model(self.cached_inputs).detach()
    
    def _apply_compression(self, config: LayerConfig):
        """Apply compression to the model"""
        if self.model_blocks is None or self.block_idx >= len(self.model_blocks):
            return
        
        for idx, (name, module) in enumerate(self.model_blocks[self.block_idx]):
            kernel_name = f'kernel{idx+1}'
            if kernel_name == 'kernel1':
                kernel_config = config.kernel1_config
            elif kernel_name == 'kernel2':
                kernel_config = config.kernel2_config
            elif kernel_name == 'kernel3':
                kernel_config = config.kernel3_config
            elif kernel_name == 'kernel4':
                kernel_config = config.kernel4_config
            else:
                continue
            
            if kernel_config is None:
                continue
            
            if name not in self.original_weights:
                self.original_weights[name] = module.weight.data.clone()
            
            weights = self.original_weights[name].clone()
            
            if kernel_config.pruning_ratio < 1.0:
                mask = AdvancedPruner.create_neuron_mask(
                    weights, kernel_config.pruning_ratio, 
                    importance_method=kernel_config.importance_method)
                weights = AdvancedPruner.apply_mask(weights, mask)
            
            weights = AdvancedQuantizer.quantize(
                weights, kernel_config.weight_bits,
                mode=kernel_config.quant_mode, quant_type=kernel_config.quant_type)
            
            module.weight.data = weights
    
    def _compute_reward(self, config: LayerConfig, 
                       full_model_configs: Optional[Dict[int, LayerConfig]] = None) -> Tuple[float, Dict]:
        """Compute reward for the configuration"""
        compression_ratio = config.compression_ratio()
        
        if self.surrogate_model is not None and self.surrogate_model.get_buffer_size() >= 3:
            if full_model_configs is None:
                full_model_configs = {self.block_idx: config}
            else:
                full_model_configs = {**full_model_configs, self.block_idx: config}
            
            predicted_accuracy = self.surrogate_model.predict(full_model_configs)
            baseline = self.baseline_accuracy if self.baseline_accuracy is not None else 85.0
            accuracy_drop = baseline - predicted_accuracy
            use_surrogate = True
        else:
            self.model.eval()
            with torch.no_grad():
                compressed_outputs = self.model(self.cached_inputs)
                mse = F.mse_loss(compressed_outputs, self.baseline_outputs)
            accuracy_drop = min(mse.item() * 20, 50.0)
            predicted_accuracy = 85.0 - accuracy_drop
            use_surrogate = False
        
        if accuracy_drop < 1.0:
            accuracy_reward = 100.0
        elif accuracy_drop < 2.0:
            accuracy_reward = 95.0 - (accuracy_drop - 1.0) * 5
        elif accuracy_drop < 3.0:
            accuracy_reward = 85.0 - (accuracy_drop - 2.0) * 10
        else:
            accuracy_reward = max(0, 70.0 - (accuracy_drop - 3.0) * 10)
        
        compression_reward = (1.0 - compression_ratio) * 60
        
        actual_compression = compression_ratio
        target_compression = self.layer_budget.target_compression_ratio
        compression_gap = abs(actual_compression - target_compression)
        budget_compliance_reward = np.exp(-10 * compression_gap) * 20
        
        sensitivity_penalty = (1.0 - compression_ratio) * self.sensitivity_score * 12
        
        stability_component = -np.std(self.reward_history[-5:]) * 2 if len(self.reward_history) > 5 else 0.0
        
        reward = (self.global_goal.alpha * accuracy_reward + 
                 self.global_goal.beta * compression_reward + 
                 budget_compliance_reward + 
                 self.global_goal.gamma * stability_component - 
                 sensitivity_penalty)
        
        if np.isnan(reward) or np.isinf(reward):
            reward = -10.0
        
        components = {
            'predicted_accuracy': predicted_accuracy,
            'baseline_accuracy': self.baseline_accuracy if self.baseline_accuracy else 85.0,
            'accuracy_drop': accuracy_drop,
            'compression_ratio': compression_ratio,
            'accuracy_reward': accuracy_reward,
            'compression_reward': compression_reward,
            'budget_compliance': budget_compliance_reward,
            'compression_gap': compression_gap,
            'target_compression': target_compression,
            'actual_compression': actual_compression,
            'reward': reward,
            'stability': stability_component,
            'used_surrogate': use_surrogate
        }
        return float(reward), components
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        for name, orig_weight in self.original_weights.items():
            module = dict(self.model.named_modules())[name]
            module.weight.data = orig_weight.clone()
        
        if self.cached_inputs is None:
            self.cached_inputs, _ = DataManager.get_cached_batches(self.dataloader, self.device, num_batches=3)
        
        self._compute_baseline_outputs()
        
        num_kernels = len(self.kernel_modules)
        state = np.zeros(12 + num_kernels, dtype=np.float32)
        state[0] = self.block_idx / 12.0
        state[1] = self.sensitivity_score
        state[2] = 1.0
        state[3] = 0.0
        state[4] = 0.0
        state[5] = self.curriculum_stage / 2.0
        state[6] = self.layer_budget.target_compression_ratio
        state[7] = self.layer_budget.priority
        state[8] = self.layer_budget.max_accuracy_drop / 10.0
        state[9] = 1.0
        state[10] = 1.0 if self.surrogate_model and self.surrogate_model.get_buffer_size() >= 3 else 0.0
        state[11] = 0.0
        
        for i in range(num_kernels):
            state[12 + i] = 0.5
        
        self.step_count = 0
        self.reward_history = []
        
        return state, {}
    
    def step(self, action):
        """Execute one step"""
        config = self._decode_action(action)
        self._apply_compression(config)
        reward, components = self._compute_reward(config)
        self.reward_history.append(reward)
        
        num_kernels = len(self.kernel_modules)
        state = np.zeros(12 + num_kernels, dtype=np.float32)
        state[0] = self.block_idx / 12.0
        state[1] = self.sensitivity_score
        state[2] = components['actual_compression']
        state[3] = components['compression_gap']
        state[4] = components['accuracy_drop'] / 10.0
        state[5] = self.curriculum_stage / 2.0
        state[6] = self.layer_budget.target_compression_ratio
        state[7] = self.layer_budget.priority
        state[8] = self.layer_budget.max_accuracy_drop / 10.0
        state[9] = 1.0
        state[10] = 1.0 if components['used_surrogate'] else 0.0
        state[11] = reward / 100.0
        
        for i in range(num_kernels):
            state[12 + i] = 0.5
        
        self.step_count += 1
        done = self.step_count >= 20
        
        return state, reward, done, False, {'config': config, 'reward_components': components}

    def __deepcopy__(self, memo):
        """Deep copy for vectorized environments"""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k in ['dataloader', 'eval_dataloader', 'cached_inputs', 'baseline_outputs', 
                     'surrogate_model', 'model_blocks']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        
        return result


class OptimizedEnsembleLowLevelAgent:
    """Ensemble of low-level RL agents with voting"""
    
    def __init__(self, env: BudgetConstrainedCompressionEnv, num_agents: int = 3, 
                 algorithms: List[str] = None, weights_config: List[Dict[str, float]] = None):
        self.env = env
        self.num_agents = num_agents
        
        if algorithms is None:
            algorithms = ['PPO', 'A2C', 'PPO'][:num_agents]
        
        self.algorithms = algorithms[:num_agents]
        self.agents = []
        self.agent_weights = np.ones(num_agents) / num_agents
        self.voting_history = []
        
        for i, algo in enumerate(self.algorithms):
            def make_env_thunk(agent_index, w_config):
                def _thunk():
                    e = copy.copy(env)
                    e.model = env.model
                    e.cached_inputs = env.cached_inputs
                    e.baseline_outputs = env.baseline_outputs
                    e.layer_budget = env.layer_budget
                    e.surrogate_model = env.surrogate_model
                    e.model_blocks = env.model_blocks
                    e.quantization_type = env.quantization_type
                    e.default_strategy = env.default_strategy
                    
                    if w_config and agent_index < len(w_config):
                        w = w_config[agent_index]
                        e.global_goal = copy.deepcopy(env.global_goal)
                        e.global_goal.alpha = w['alpha']
                        e.global_goal.beta = w['beta']
                        e.global_goal.gamma = w['gamma']
                    
                    return e
                return _thunk
            
            env_cmds = [make_env_thunk(i, weights_config) for _ in range(2)]
            vec_env = DummyVecEnv(env_cmds)
            
            if algo == 'PPO':
                agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4, 
                           n_steps=128, batch_size=32, ent_coef=0.01, 
                           seed=42 + i, max_grad_norm=0.5, device='cpu')
            elif algo == 'A2C':
                agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=7e-4, 
                           n_steps=64, ent_coef=0.01, seed=42 + i, 
                           max_grad_norm=0.5, device='cpu')
            else:
                agent = PPO('MlpPolicy', vec_env, verbose=0, seed=42 + i, 
                           max_grad_norm=0.5, device='cpu')
            
            self.agents.append({
                'agent': agent,
                'algorithm': algo,
                'env': vec_env,
                'performance_history': []
            })
        
        self.voting_method = 'weighted'
    
    def update_budget(self, new_budget: LayerBudget):
        """Update budget for all agents"""
        self.env.update_budget(new_budget)
        for agent_dict in self.agents:
            for env in agent_dict['env'].envs:
                env.update_budget(new_budget)
    
    def train(self, total_timesteps: int = 1000):
        """Train all agents"""
        for agent_dict in self.agents:
            agent_dict['agent'].learn(total_timesteps=total_timesteps)
    
    def _vote_on_action(self, actions: List[np.ndarray], method: str = 'weighted') -> np.ndarray:
        """Vote on action using weighted ensemble"""
        if method == 'weighted':
            voted_action = np.zeros(actions[0].shape, dtype=float)
            for i, action in enumerate(actions):
                voted_action += self.agent_weights[i] * action
            return np.round(voted_action).astype(int)
        else:
            return actions[0]
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, LayerConfig]:
        """Get configuration from ensemble"""
        actions = []
        for agent_dict in self.agents:
            action, _ = agent_dict['agent'].predict(state, deterministic=True)
            actions.append(action)
        
        voted_action = self._vote_on_action(actions, method=self.voting_method)
        config = self.env._decode_action(voted_action)
        
        self.voting_history.append({
            'actions': [a.tolist() for a in actions],
            'voted_action': voted_action.tolist(),
            'weights': self.agent_weights.tolist()
        })
        
        return voted_action, config
    
    def get_config(self) -> LayerConfig:
        """Get best configuration"""
        obs, _ = self.env.reset()
        voted_action, config = self.predict(obs)
        
        best_agent_idx = np.argmax(self.agent_weights)
        config.selected_by_agent = f"{self.algorithms[best_agent_idx]}_agent_{best_agent_idx}"
        config.agent_confidence = float(self.agent_weights[best_agent_idx])
        
        return config
