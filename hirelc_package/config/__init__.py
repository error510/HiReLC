"""Configuration classes for HiReLC framework"""

from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np


@dataclass
class CompressionGoal:
    """Multi-objective compression targets with weighted optimization."""
    target_accuracy_drop: float = 1.0
    target_compression_ratio: float = 0.25
    target_flops_reduction: float = 0.30

    min_layer_bits: int = 2
    max_layer_bits: int = 8

    min_layer_pruning: float = 0.0
    max_layer_pruning: float = 0.8

    alpha: float = 50.0
    beta: float = 2.0
    gamma: float = 1.0
    delta: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
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
        return cls(**data)


@dataclass
class LayerBudget:
    """Budget allocated by HLA for a specific layer/block."""
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
        return {
            'block_idx': self.block_idx,
            'target_compression_ratio': self.target_compression_ratio,
            'max_accuracy_drop': self.max_accuracy_drop,
            'priority': self.priority,
            'sensitivity': self.sensitivity,
            'preferred_strategy': self.preferred_strategy,
            'min_bits': self.min_bits,
            'max_pruning': self.max_pruning,
            'global_min_bits': self.global_min_bits,
            'global_max_bits': self.global_max_bits,
            'global_min_pruning': self.global_min_pruning,
            'global_max_pruning': self.global_max_pruning,
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with comprehensive parameter control."""
    model_name: str = "resnet18"
    dataset: str = "tinyimagenet"
    num_classes: int = 200

    do_finetune: bool = True
    finetune_epochs: int = 10
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01

    num_lla_agents: int = 3
    num_hla_agents: int = 3
    lla_timesteps: int = 2048
    hla_timesteps: int = 2048
    rl_algorithms: List[str] = field(default_factory=lambda: ['PPO', 'A2C', 'PPO'])
    curriculum_stages: int = 3
    hla_budget_update_freq: int = 1

    use_surrogate: bool = True
    surrogate_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    surrogate_warmup_samples: int = 100
    surrogate_update_freq: int = 1

    sensitivity_method: str = 'fisher'
    sensitivity_methods: Optional[List[str]] = None

    quantization_type: str = 'mixed'
    default_strategy: Optional[str] = None

    enable_pruning: bool = True
    enable_quantization: bool = True
    strategy: Optional[str] = None

    device: str = 'cuda'
    output_dir: str = './outputs'
    seed: int = 42
    num_seeds: int = 1
    verbose: int = 1
    save_checkpoints: bool = True

    compression_goal: Optional[CompressionGoal] = None
    experiment_name: str = 'hirelc_experiment'

    lla_weights: Optional[List[Dict[str, float]]] = None
    hla_weights: Optional[List[Dict[str, float]]] = None

    def __post_init__(self) -> None:
        if self.compression_goal is None:
            self.compression_goal = CompressionGoal()

    def to_json(self, filepath: Optional[str] = None) -> str:
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
            'surrogate_warmup_samples': self.surrogate_warmup_samples,
            'surrogate_update_freq': self.surrogate_update_freq,
            'sensitivity_method': self.sensitivity_method,
            'sensitivity_methods': self.sensitivity_methods,
            'quantization_type': self.quantization_type,
            'default_strategy': self.default_strategy,
            'enable_pruning': self.enable_pruning,
            'enable_quantization': self.enable_quantization,
            'strategy': self.strategy,
            'device': self.device,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'num_seeds': self.num_seeds,
            'verbose': self.verbose,
            'save_checkpoints': self.save_checkpoints,
            'experiment_name': self.experiment_name,
            'compression_goal': self.compression_goal.to_dict() if self.compression_goal else None,
            'lla_weights': self.lla_weights,
            'hla_weights': self.hla_weights,
        }

        json_str = json.dumps(data, indent=2)
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_text(json_str, encoding='utf-8')
        return json_str

    @classmethod
    def from_json(cls, json_input: str) -> 'ExperimentConfig':
        path = Path(json_input)
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
        else:
            data = json.loads(json_input)

        if data.get('compression_goal'):
            data['compression_goal'] = CompressionGoal.from_dict(data['compression_goal'])

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class KernelConfig:
    """Configuration for a single kernel/weight matrix."""
    name: str = ''
    weight_bits: int = 8
    act_bits: int = 8
    quant_type: str = 'INT'
    quant_mode: str = 'uniform'
    pruning_ratio: float = 1.0
    importance_method: str = 'l2'
    shape: Tuple[int, ...] = field(default_factory=tuple)

    def compression_ratio(self) -> float:
        return (self.weight_bits / 32.0) * self.pruning_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'weight_bits': self.weight_bits,
            'act_bits': self.act_bits,
            'quant_type': self.quant_type,
            'quant_mode': self.quant_mode,
            'pruning_ratio': self.pruning_ratio,
            'importance_method': self.importance_method,
            'shape': list(self.shape) if isinstance(self.shape, tuple) else self.shape,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KernelConfig':
        return cls(**data)


@dataclass
class LayerConfig:
    """Configuration for a complete layer/block with multiple kernels."""
    block_idx: int
    kernels: Dict[str, KernelConfig] = field(default_factory=dict)

    kernel1_config: Optional[KernelConfig] = None
    kernel2_config: Optional[KernelConfig] = None
    kernel3_config: Optional[KernelConfig] = None
    kernel4_config: Optional[KernelConfig] = None

    avg_weight_bits: int = 8
    avg_act_bits: int = 8
    avg_pruning_ratio: float = 1.0

    selected_by_agent: str = ''
    agent_confidence: float = 0.0
    assigned_budget: Optional[LayerBudget] = None

    def _collect_kernel_configs(self) -> List[KernelConfig]:
        configs: List[KernelConfig] = []
        seen = set()

        for kc in self.kernels.values():
            if kc is None:
                continue
            kid = id(kc)
            if kid not in seen:
                configs.append(kc)
                seen.add(kid)

        for kc in [self.kernel1_config, self.kernel2_config,
                   self.kernel3_config, self.kernel4_config]:
            if kc is None:
                continue
            kid = id(kc)
            if kid not in seen:
                configs.append(kc)
                seen.add(kid)

        return configs

    def _sync_kernels_from_attrs(self) -> None:
        for idx, kc in enumerate(
            [self.kernel1_config, self.kernel2_config,
             self.kernel3_config, self.kernel4_config],
            start=1
        ):
            if kc is None:
                continue
            key = kc.name or f"kernel{idx}"
            self.kernels.setdefault(key, kc)

    def _sync_attrs_from_kernels(self) -> None:
        if not self.kernels:
            return

        if all(k in self.kernels for k in ['kernel_0', 'kernel_1', 'kernel_2', 'kernel_3']):
            ordered = ['kernel_0', 'kernel_1', 'kernel_2', 'kernel_3']
        elif all(k in self.kernels for k in ['kernel1', 'kernel2', 'kernel3', 'kernel4']):
            ordered = ['kernel1', 'kernel2', 'kernel3', 'kernel4']
        elif all(k in self.kernels for k in ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']):
            ordered = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
        else:
            ordered = sorted(self.kernels.keys())[:4]

        if self.kernel1_config is None and len(ordered) > 0:
            self.kernel1_config = self.kernels[ordered[0]]
        if self.kernel2_config is None and len(ordered) > 1:
            self.kernel2_config = self.kernels[ordered[1]]
        if self.kernel3_config is None and len(ordered) > 2:
            self.kernel3_config = self.kernels[ordered[2]]
        if self.kernel4_config is None and len(ordered) > 3:
            self.kernel4_config = self.kernels[ordered[3]]

    def update_aggregates(self) -> None:
        self._sync_kernels_from_attrs()
        self._sync_attrs_from_kernels()

        configs = self._collect_kernel_configs()
        if configs:
            self.avg_weight_bits = int(np.mean([c.weight_bits for c in configs]))
            self.avg_act_bits = int(np.mean([c.act_bits for c in configs]))
            self.avg_pruning_ratio = float(np.mean([c.pruning_ratio for c in configs]))

    def compression_ratio(self) -> float:
        configs = self._collect_kernel_configs()
        if not configs:
            return 1.0
        return float(np.mean([c.compression_ratio() for c in configs]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'block_idx': self.block_idx,
            'kernels': {name: k.to_dict() for name, k in self.kernels.items()},
            'kernel1': self.kernel1_config.to_dict() if self.kernel1_config else None,
            'kernel2': self.kernel2_config.to_dict() if self.kernel2_config else None,
            'kernel3': self.kernel3_config.to_dict() if self.kernel3_config else None,
            'kernel4': self.kernel4_config.to_dict() if self.kernel4_config else None,
            'avg_weight_bits': self.avg_weight_bits,
            'avg_act_bits': self.avg_act_bits,
            'avg_pruning_ratio': self.avg_pruning_ratio,
            'selected_by_agent': self.selected_by_agent,
            'agent_confidence': self.agent_confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        kernels = {}
        if 'kernels' in data and isinstance(data['kernels'], dict):
            for name, kdata in data['kernels'].items():
                kernels[name] = KernelConfig.from_dict(kdata)

        cfg = cls(block_idx=data['block_idx'], kernels=kernels)

        for idx, key in enumerate(['kernel1', 'kernel2', 'kernel3', 'kernel4'], start=1):
            if data.get(key):
                setattr(cfg, f'kernel{idx}_config', KernelConfig.from_dict(data[key]))

        cfg.update_aggregates()
        return cfg
