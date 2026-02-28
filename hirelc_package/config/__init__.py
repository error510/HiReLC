"""Configuration classes for HiReLC framework"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class CompressionGoal:
    """
    Multi-objective compression targets with weighted optimization.
    
    Attributes:
        target_accuracy_drop: Maximum allowed accuracy drop (%).
        target_compression_ratio: Target compression ratio (< 1.0).
        target_flops_reduction: Target FLOPs reduction ratio.
        min_layer_bits: Minimum bitwidth for quantization.
        max_layer_bits: Maximum bitwidth for quantization.
        min_layer_pruning: Minimum pruning ratio (0.0 = no pruning).
        max_layer_pruning: Maximum pruning ratio (0.8 = 80% pruned).
        alpha, beta, gamma, delta: Loss function weights.
    """
    target_accuracy_drop: float = 1.0
    target_compression_ratio: float = 0.25
    target_flops_reduction: float = 0.30
    
    min_layer_bits: int = 2
    max_layer_bits: int = 8
    
    min_layer_pruning: float = 0.0
    max_layer_pruning: float = 0.8
    
    # Loss function weights
    alpha: float = 50.0   # Accuracy weight
    beta: float = 2.0     # Compression weight
    gamma: float = 1.0    # Budget compliance weight
    delta: float = 1.0    # Regularization weight

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'target_accuracy_drop': self.target_accuracy_drop,
            'target_compression_ratio': self.target_compression_ratio,
            'target_flops_reduction': self.target_flops_reduction,
            'min_layer_bits': self.min_layer_bits,
            'max_layer_bits': self.max_layer_bits,
            'min_layer_pruning': self.min_layer_pruning,
            'max_layer_pruning': self.max_layer_pruning,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionGoal':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class LayerBudget:
    """
    Budget allocated by HLA for a specific layer/block.
    
    Attributes:
        block_idx: Index of compression block.
        target_compression_ratio: Target compression for this block.
        max_accuracy_drop: Maximum allowed accuracy drop for this block.
        priority: Priority score for this block.
        sensitivity: Sensitivity score from fisher information or other methods.
        preferred_strategy: 'auto', 'quant_only', 'prune_only', or 'mixed'.
        min_bits: Minimum bitwidth for this block.
        max_pruning: Maximum pruning ratio for this block.
    """
    block_idx: int
    target_compression_ratio: float
    max_accuracy_drop: float
    priority: float
    sensitivity: float
    
    preferred_strategy: str = 'auto'
    min_bits: int = 4
    max_pruning: float = 0.4
    
    global_min_bits: int = 2
    global_max_bits: int = 8
    global_min_pruning: float = 0.0
    global_max_pruning: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'block_idx': self.block_idx,
            'target_compression_ratio': self.target_compression_ratio,
            'max_accuracy_drop': self.max_accuracy_drop,
            'priority': self.priority,
            'sensitivity': self.sensitivity,
            'preferred_strategy': self.preferred_strategy,
            'min_bits': self.min_bits,
            'max_pruning': self.max_pruning,
        }


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration with comprehensive parameter control.
    """
    # Model & Dataset
    model_name: str = "resnet18"
    dataset: str = "tinyimagenet"
    num_classes: int = 200
    
    # Training
    do_finetune: bool = True
    finetune_epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    
    # RL Agents
    num_lla_agents: int = 3
    num_hla_agents: int = 3
    lla_timesteps: int = 2048
    hla_timesteps: int = 2048
    rl_algorithms: List[str] = field(default_factory=lambda: ['PPO', 'A2C', 'PPO'])
    
    # Curriculum & Updates
    curriculum_stages: int = 3
    hla_budget_update_freq: int = 1
    
    # Surrogate Model
    use_surrogate: bool = True
    surrogate_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    # Sensitivity Estimation Method
    sensitivity_method: str = 'fisher'  # 'fisher', 'hessian', 'snip', 'gradient'
    
    # Compression Strategy
    quantization_type: str = 'mixed'  # 'mixed', 'int', 'float'
    default_strategy: Optional[str] = None  # None, 'uniform', 'log', 'per-channel', 'learned'
    
    # System
    device: str = 'cuda'
    output_dir: str = './outputs'
    seed: int = 42
    
    # Compression Goal
    compression_goal: Optional[CompressionGoal] = None
    experiment_name: str = 'hirelc_experiment'
    
    # HLA Weights Configuration
    hla_weights: Optional[List[Dict[str, float]]] = None

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        data = {
            'model_name': self.model_name,
            'dataset': self.dataset,
            'num_classes': self.num_classes,
            'do_finetune': self.do_finetune,
            'finetune_epochs': self.finetune_epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'num_lla_agents': self.num_lla_agents,
            'num_hla_agents': self.num_hla_agents,
            'lla_timesteps': self.lla_timesteps,
            'hla_timesteps': self.hla_timesteps,
            'rl_algorithms': self.rl_algorithms,
            'curriculum_stages': self.curriculum_stages,
            'hla_budget_update_freq': self.hla_budget_update_freq,
            'use_surrogate': self.use_surrogate,
            'surrogate_hidden_dims': self.surrogate_hidden_dims,
            'sensitivity_method': self.sensitivity_method,
            'quantization_type': self.quantization_type,
            'default_strategy': self.default_strategy,
            'device': self.device,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'compression_goal': self.compression_goal.to_dict() if self.compression_goal else None,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if data.get('compression_goal'):
            data['compression_goal'] = CompressionGoal.from_dict(data['compression_goal'])
        
        return cls(**data)


@dataclass
class KernelConfig:
    """Configuration for a single kernel/weight matrix."""
    weight_bits: int = 8
    pruning_ratio: float = 1.0
    quant_type: str = 'INT'  # 'INT' or 'FLOAT'
    quant_mode: str = 'uniform'  # 'uniform', 'log', 'per-channel', 'learned'
    
    def compression_ratio(self) -> float:
        """Compute compression ratio for this kernel."""
        original_bits = 32
        compressed_bits = self.weight_bits * self.pruning_ratio
        return compressed_bits / original_bits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'weight_bits': self.weight_bits,
            'pruning_ratio': self.pruning_ratio,
            'quant_type': self.quant_type,
            'quant_mode': self.quant_mode,
        }


@dataclass
class LayerConfig:
    """
    Configuration for a complete layer/block with multiple kernels.
    
    For ViT: Typically contains qkv, attn_proj, mlp_fc1, mlp_fc2 kernels.
    For CNN: Groups of Conv2d/Linear layers.
    """
    block_idx: int
    kernels: Dict[str, KernelConfig] = field(default_factory=dict)
    
    def compression_ratio(self) -> float:
        """Compute average compression ratio."""
        if not self.kernels:
            return 1.0
        ratios = [k.compression_ratio() for k in self.kernels.values()]
        return sum(ratios) / len(ratios) if ratios else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'block_idx': self.block_idx,
            'kernels': {name: kernel.to_dict() for name, kernel in self.kernels.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        """Create from dictionary."""
        kernels = {
            name: KernelConfig(**kdata)
            for name, kdata in data.get('kernels', {}).items()
        }
        return cls(block_idx=data['block_idx'], kernels=kernels)
