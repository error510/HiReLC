"""
High-Level RL Agent for budget allocation across layers.

This module implements the HLA which decides how much compression budget
(compression ratio per layer) to allocate to each layer.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from hirelc_package.config import CompressionGoal, LayerBudget


class BudgetAllocationEnvironment(gym.Env):
    """
    RL environment for the High-Level Agent that allocates compression budgets.
    
    Action space:
    - [5, 3] per block: compression_level (0-4), strategy (0-2)
    """
    
    def __init__(self, model: nn.Module, eval_dataloader: DataLoader,
                 sensitivity_scores: Dict[int, float],
                 global_goal: CompressionGoal, device: str = 'cuda',
                 num_blocks: int = 12):
        super().__init__()
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.sensitivity_scores = sensitivity_scores
        self.global_goal = global_goal
        self.device = device
        self.num_blocks = num_blocks
        
        self.action_space = spaces.MultiDiscrete([5, 3] * num_blocks)
        self.observation_space = spaces.Box(low=-10, high=10, 
                                           shape=(5 + num_blocks * 2,), 
                                           dtype=np.float32)
        
        self.baseline_accuracy = None
        self.step_count = 0
        self.current_budgets = {}
        
        self.last_acc_drop = 0.0
        self.last_compression = 1.0
        self.cycle_progress = 0.0
    
    def update_feedback(self, acc_drop: float, comp_ratio: float, cycle_prog: float):
        """Update feedback from previous cycle"""
        self.last_acc_drop = acc_drop
        self.last_compression = comp_ratio
        self.cycle_progress = cycle_prog

    def _evaluate_model(self) -> float:
        """Evaluate model on validation set (limited batches)"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.eval_dataloader):
                if i >= 10:  # Limit for speed
                    break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total if total > 0 else 0.0
    
    def _decode_action(self, action: np.ndarray) -> Dict[int, LayerBudget]:
        """Decode action into LayerBudget objects"""
        budgets = {}
        
        compression_levels = {
            0: 0.35, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15
        }
        
        pruning_limits = {
            0: min(0.20, self.global_goal.max_layer_pruning),
            1: min(0.30, self.global_goal.max_layer_pruning),
            2: min(0.40, self.global_goal.max_layer_pruning),
            3: min(0.60, self.global_goal.max_layer_pruning),
            4: min(0.80, self.global_goal.max_layer_pruning),
        }
        
        strategies = ['quantization', 'pruning', 'auto']
        
        for block_idx in range(self.num_blocks):
            base_idx = block_idx * 2
            comp_level = int(action[base_idx])
            strategy_idx = int(action[base_idx + 1])
            
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            
            # Sensitivity-aware adjustment
            if sensitivity > 0.7:
                comp_level = max(0, comp_level - 1)
            elif sensitivity < 0.3:
                comp_level = min(4, comp_level + 1)
            
            target_compression = compression_levels[comp_level]
            max_pruning = pruning_limits[comp_level]
            preferred_strategy = strategies[strategy_idx % len(strategies)]
            
            # Determine minimum bits based on compression level
            if comp_level <= 1:
                min_bits = max(6, self.global_goal.min_layer_bits)
            elif comp_level <= 2:
                min_bits = max(5, self.global_goal.min_layer_bits)
            else:
                min_bits = max(4, self.global_goal.min_layer_bits)
            
            priority = sensitivity
            max_acc_drop = (0.5 if sensitivity > 0.7 else 
                           1.0 if sensitivity > 0.5 else 2.0)
            
            budget = LayerBudget(
                block_idx=block_idx,
                target_compression_ratio=target_compression,
                max_accuracy_drop=max_acc_drop,
                priority=priority,
                sensitivity=sensitivity,
                preferred_strategy=preferred_strategy,
                min_bits=min_bits,
                max_pruning=max_pruning,
                global_min_bits=self.global_goal.min_layer_bits,
                global_max_bits=self.global_goal.max_layer_bits,
                global_min_pruning=self.global_goal.min_layer_pruning,
                global_max_pruning=self.global_goal.max_layer_pruning
            )
            budgets[block_idx] = budget
        
        return budgets
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if self.baseline_accuracy is None:
            self.baseline_accuracy = self._evaluate_model()
        
        state = np.zeros(5 + self.num_blocks * 2, dtype=np.float32)
        state[0] = self.last_acc_drop / 10.0   
        state[1] = self.last_compression      
        state[2] = self.cycle_progress        
        state[3] = self.global_goal.target_compression_ratio
        state[4] = self.global_goal.target_accuracy_drop / 10.0
        
        for i in range(self.num_blocks):
            state[5 + i * 2] = self.sensitivity_scores.get(i, 0.5)
            state[5 + i * 2 + 1] = 0.5
        
        self.step_count = 0
        self.current_budgets = {}
        
        return state, {}
    
    def step(self, action):
        """Execute one step"""
        budgets = self._decode_action(action)
        self.current_budgets = budgets
        
        # Compute balance reward (low variance in compression across layers)
        avg_target_compression = np.mean([b.target_compression_ratio 
                                          for b in budgets.values()])
        compression_variance = np.var([b.target_compression_ratio 
                                      for b in budgets.values()])
        balance_reward = -compression_variance * 10
        
        # Sensitivity-aware reward
        sensitivity_aware_reward = 0.0
        for block_idx, budget in budgets.items():
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            
            if sensitivity > 0.7 and budget.target_compression_ratio > 0.3:
                sensitivity_aware_reward += (budget.target_compression_ratio - 0.3) * 10
            elif sensitivity < 0.3 and budget.target_compression_ratio < 0.2:
                sensitivity_aware_reward += (0.2 - budget.target_compression_ratio) * 5
        
        # Global compression reward
        global_compression_gap = abs(avg_target_compression - 
                                    self.global_goal.target_compression_ratio)
        compression_reward = -global_compression_gap * 50
        
        # Dynamic reward based on previous cycle
        dynamic_reward = 0.0
        if self.last_acc_drop > self.global_goal.target_accuracy_drop:
            if avg_target_compression < self.last_compression:
                dynamic_reward -= 50.0 * (self.last_acc_drop - 
                                         self.global_goal.target_accuracy_drop)
        elif self.last_compression > self.global_goal.target_compression_ratio:
            if avg_target_compression < self.last_compression:
                dynamic_reward += 20.0
        
        reward = (balance_reward + sensitivity_aware_reward + 
                 compression_reward + dynamic_reward)
        
        state = np.zeros(5 + self.num_blocks * 2, dtype=np.float32)
        state[0] = self.last_acc_drop / 10.0
        state[1] = self.last_compression
        state[2] = self.cycle_progress
        state[3] = global_compression_gap
        state[4] = (balance_reward + sensitivity_aware_reward + dynamic_reward) / 20.0
        
        for i in range(self.num_blocks):
            state[5 + i * 2] = self.sensitivity_scores.get(i, 0.5)
            if i in budgets:
                state[5 + i * 2 + 1] = budgets[i].target_compression_ratio
        
        self.step_count += 1
        done = self.step_count >= 10
        info = {'budgets': budgets, 'avg_compression': avg_target_compression}
        
        return state, reward, done, False, info


class EnsembleHighLevelAgent:
    """
    Ensemble of HLA agents that allocate compression budgets across layers.
    """
    
    def __init__(self, model: nn.Module, eval_dataloader: DataLoader,
                 sensitivity_scores: Dict[int, float],
                 global_goal: CompressionGoal, device: str = 'cuda',
                 num_blocks: int = 12, num_hla_agents: int = 3,
                 weights_config: List[Dict[str, float]] = None):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.sensitivity_scores = sensitivity_scores
        self.global_goal = global_goal
        self.device = device
        self.num_blocks = num_blocks
        self.num_hla_agents = num_hla_agents
        
        self.hla_agents = []
        algorithms = ['PPO', 'A2C', 'PPO'][:num_hla_agents]
        self.budget_proposals = []
        
        for i in range(num_hla_agents):
            specific_goal = copy.deepcopy(global_goal)
            
            if weights_config and i < len(weights_config):
                w = weights_config[i]
                specific_goal.alpha = w.get('alpha', global_goal.alpha)
                specific_goal.beta = w.get('beta', global_goal.beta)
                specific_goal.gamma = w.get('gamma', global_goal.gamma)
            
            env = BudgetAllocationEnvironment(
                model=model, eval_dataloader=eval_dataloader,
                sensitivity_scores=sensitivity_scores,
                global_goal=specific_goal, device=device,
                num_blocks=num_blocks
            )
            
            vec_env = DummyVecEnv([lambda e=env: e])
            
            algo = algorithms[i]
            if algo == 'PPO':
                agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=1e-4,
                           n_steps=64, batch_size=32, ent_coef=0.01,
                           seed=42 + i, max_grad_norm=0.5, device='cpu')
            elif algo == 'A2C':
                agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4,
                           n_steps=64, ent_coef=0.01, seed=42 + i,
                           max_grad_norm=0.5, device='cpu')
            else:
                agent = PPO('MlpPolicy', vec_env, verbose=0, seed=42 + i,
                           max_grad_norm=0.5, device='cpu')
            
            self.hla_agents.append({
                'agent': agent,
                'env': vec_env,
                'algorithm': algo,
                'weight': 1.0 / num_hla_agents
            })
    
    def update_environments(self, acc_drop: float, comp_ratio: float, 
                           cycle: int, max_cycles: int):
        """Update HLA environments with feedback from previous cycle"""
        progress = cycle / float(max_cycles) if max_cycles > 0 else 0.0
        
        for hla_dict in self.hla_agents:
            for env in hla_dict['env'].envs:
                env.update_feedback(acc_drop, comp_ratio, progress)
    
    def train_hla_agents(self, total_timesteps: int = 2000):
        """Train HLA agents"""
        for hla_dict in self.hla_agents:
            hla_dict['agent'].learn(total_timesteps=total_timesteps)
    
    def allocate_budgets(self, deterministic: bool = True) -> Dict[int, LayerBudget]:
        """Allocate budgets using ensemble voting"""
        all_budgets = []
        
        for hla_dict in self.hla_agents:
            obs, _ = hla_dict['env'].reset()
            action, _ = hla_dict['agent'].predict(obs, deterministic=deterministic)
            current_env = hla_dict['env'].envs[0]
            budgets = current_env._decode_action(action[0])
            all_budgets.append(budgets)
        
        # Aggregate budgets from all agents
        final_budgets = {}
        for block_idx in range(self.num_blocks):
            block_proposals = [budgets[block_idx] for budgets in all_budgets 
                             if block_idx in budgets]
            
            if not block_proposals:
                continue
            
            # Average compression target
            avg_target_compression = np.mean(
                [b.target_compression_ratio for b in block_proposals])
            
            # Average accuracy tolerance
            avg_max_acc_drop = np.mean(
                [b.max_accuracy_drop for b in block_proposals])
            
            # Average priority
            avg_priority = np.mean([b.priority for b in block_proposals])
            
            # Average max pruning
            avg_max_pruning = np.mean([b.max_pruning for b in block_proposals])
            
            # Most common strategy
            strategies = [b.preferred_strategy for b in block_proposals]
            most_common_strategy = max(set(strategies), key=strategies.count)
            
            # Average minimum bits
            min_bits_list = [b.min_bits for b in block_proposals]
            avg_min_bits = int(np.round(np.mean(min_bits_list)))
            
            final_budget = LayerBudget(
                block_idx=block_idx,
                target_compression_ratio=float(avg_target_compression),
                max_accuracy_drop=float(avg_max_acc_drop),
                priority=float(avg_priority),
                sensitivity=self.sensitivity_scores.get(block_idx, 0.5),
                preferred_strategy=most_common_strategy,
                min_bits=avg_min_bits,
                max_pruning=float(avg_max_pruning)
            )
            final_budgets[block_idx] = final_budget
        
        return final_budgets
