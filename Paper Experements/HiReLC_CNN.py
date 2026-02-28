import os
import sys
import copy
import json
import time
import warnings
import platform
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr

import timm

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# ============================================================================
# PYTORCH OPTIMIZATIONS
# ============================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

QUANT_TYPES = ['INT', 'FLOAT']
QUANT_MODES = ['uniform', 'log', 'per-channel', 'learned']

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CompressionGoal:
    """Multi-objective compression targets"""
    target_accuracy_drop: float = 1.0
    target_compression_ratio: float = 0.25
    target_flops_reduction: float = 0.30
    
    # Global bit constraints
    min_layer_bits: int = 2
    max_layer_bits: int = 8
    
    # Global pruning constraints  
    min_layer_pruning: float = 0.0
    max_layer_pruning: float = 0.8
    
    alpha: float = 50.0
    beta: float = 2.0
    gamma: float = 1.0
    delta: float = 1.0


@dataclass
class LayerBudget:
    """Budget allocated by HLA for a specific layer"""
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


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model_name: str = "resnet18"  # Use ResNet18 or mobilenetv2_100
    dataset: str = "tinyimagenet" # Switched to TinyImageNet
    num_classes: int = 200        # TinyImageNet has 200 classes
    
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
    
    hla_budget_update_freq: int = 1 # Update budget every cycle to allow continuous learning
    
    # Surrogate config
    use_surrogate: bool = True
    surrogate_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    surrogate_warmup_samples: int = 100
    surrogate_update_freq: int = 1
    
    lla_weights: List[Dict[str, float]] = field(default_factory=lambda: [
        {'alpha': 30.0, 'beta': 1.5, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 5.0, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 8.0, 'gamma': 1.0},
    ])

    hla_weights: List[Dict[str, float]] = field(default_factory=lambda: [
        {'alpha': 30.0, 'beta': 6.0, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 10.0, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 4.0, 'gamma': 1.0},
    ])
    
    compression_goal: CompressionGoal = field(default_factory=CompressionGoal)
    enable_pruning: bool = True
    enable_quantization: bool = True

    quantization_type: str = 'mixed'

    strategy: str = None

    num_seeds: int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir: str = './'
    experiment_name: str = 'hrl_cnn_complete'
    
    verbose: int = 1
    save_checkpoints: bool = True


class ReproducibilityManager:
    """Ensure reproducibility"""
    @staticmethod
    def set_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        info = {
            'timestamp': datetime.now().isoformat(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        }
        return info


class ExperimentLogger:
    """Comprehensive logging"""
    def __init__(self, experiment_name: str, output_dir: str = './outputs'):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / experiment_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.run_dir / 'experiment.log'
        self.metrics_file = self.run_dir / 'metrics.json'
        
        self.metrics = {
            'environment': ReproducibilityManager.get_environment_info(),
            'config': {}, 'results': {}, 'timing': {}, 'checkpoints': [],
            'cycle_history': [], 'surrogate_history': [], 'surrogate_predictions': [] 
        }
        self.start_time = time.time()
        self._init_log_file()
    
    def _init_log_file(self):
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"EXPERIMENT: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, level: str = 'INFO', print_console: bool = True):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] [{level:5s}] {message}"
        if print_console:
            if level == 'ERROR': print(f"\033[91m{log_message}\033[0m")
            elif level == 'WARN': print(f"\033[93m{log_message}\033[0m")
            elif level == 'SUCCESS': print(f"\033[92m{log_message}\033[0m")
            else: print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_config(self, config: Dict[str, Any]):
        self.metrics['config'] = config
        self.log("="*80)
        self.log("EXPERIMENT CONFIGURATION")
        self.log("="*80)
        self.log(json.dumps(config, indent=2, default=str))
        self.log("="*80)
    
    def log_metric(self, key: str, value: Any, step: Optional[int] = None):
        if step is not None:
            if key not in self.metrics['results']:
                self.metrics['results'][key] = {}
            self.metrics['results'][key][step] = value
        else:
            self.metrics['results'][key] = value
    
    def log_cycle_result(self, cycle: int, data: Dict[str, Any]):
        self.metrics['cycle_history'].append({
            'cycle': cycle, 'timestamp': datetime.now().isoformat(), **data
        })
    
    def log_surrogate_training(self, epoch: int, loss: float, samples: int):
        self.metrics['surrogate_history'].append({
            'epoch': epoch, 'loss': loss, 'samples': samples, 'timestamp': datetime.now().isoformat()
        })
    
    def log_surrogate_prediction(self, cycle: int, predicted: float, actual: float):
        self.metrics['surrogate_predictions'].append({
            'cycle': cycle, 'predicted_accuracy': predicted, 'actual_accuracy': actual,
            'error': abs(predicted - actual), 'timestamp': datetime.now().isoformat()
        })
    
    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def get_run_dir(self) -> Path:
        return self.run_dir
    
    def save_checkpoint(self, model: nn.Module, name: str):
        try:
            checkpoint_path = self.run_dir / 'checkpoints' / f'{name}.pth'
            checkpoint_path.parent.mkdir(exist_ok=True)
            state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(state_dict, checkpoint_path)
            self.metrics['checkpoints'].append(str(checkpoint_path))
            self.log(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.log(f"Warning: Could not save checkpoint {name}: {str(e)}", level='WARN')


# ============================================================================
# DATA LOADING
# ============================================================================

class HFDatasetBridge(Dataset):
    """Bridge for HuggingFace datasets to standard PyTorch format"""
    def __init__(self, hf_ds, transform=None):
        self.hf_ds = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.hf_ds)
    def __getitem__(self, idx):
        item = self.hf_ds[idx]
        img = item['image'].convert('RGB')
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label

class DataManager:
    @staticmethod
    def get_tinyimagenet(batch_size: int = 64, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        try:
            from datasets import load_dataset
        except ImportError:
            os.system("pip install datasets")
            from datasets import load_dataset
            
        dataset = load_dataset('Maysee/tiny-imagenet')
        
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        trainset = HFDatasetBridge(dataset['train'], transform=transform_train)
        testset = HFDatasetBridge(dataset['valid'], transform=transform_test)
        
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, test_loader

    @staticmethod
    def get_cached_batches(dataloader, device, num_batches=5):
        cached_inputs, cached_targets = [], []
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches: break
            cached_inputs.append(x.to(device, non_blocking=True))
            cached_targets.append(y.to(device, non_blocking=True))
        return torch.cat(cached_inputs), torch.cat(cached_targets)

# ============================================================================
# QUANTIZATION & PRUNING
# ============================================================================

@torch.jit.script
def _quantize_uniform_jit(tensor: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    if bits >= 32: return tensor
    if symmetric:
        qmax = float(2 ** (bits - 1) - 1); qmin = float(-(2 ** (bits - 1)))
        max_val = tensor.abs().max(); scale = max_val / qmax if qmax > 0 else torch.tensor(1.0, device=tensor.device)
        zero_point = torch.tensor(0.0, device=tensor.device)
    else:
        qmin = 0.0; qmax = float(2 ** bits - 1)
        min_val, max_val = tensor.min(), tensor.max()
        scale = (max_val - min_val) / (qmax - qmin) if qmax > qmin else torch.tensor(1.0, device=tensor.device)
        zero_point = qmin - torch.round(min_val / scale)
    scale = torch.clamp(scale, min=1e-8)
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    return dequantized

class AdvancedQuantizer:
    @staticmethod
    def quantize_uniform(tensor: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
        return _quantize_uniform_jit(tensor, bits, symmetric)

    @staticmethod
    def quantize_log(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 32: return tensor
        sign    = torch.sign(tensor)
        abs_val = tensor.abs().clamp(min=1e-8)
        log_val = torch.log2(abs_val)
        qmax    = float(2 ** (bits - 1) - 1)
        log_min, log_max = log_val.min(), log_val.max()
        scale = ((log_max - log_min) / (2 * qmax)
                 if qmax > 0 else torch.tensor(1.0, device=tensor.device)).clamp(min=1e-8)
        q_log = torch.round((log_val - log_min) / scale - qmax).clamp(-qmax, qmax)
        dq_log = (q_log + qmax) * scale + log_min
        return sign * (2.0 ** dq_log)

    @staticmethod
    def quantize_per_channel(tensor, bits, symmetric=True):
        if bits >= 32: return tensor
        if tensor.dim() < 2:
            return AdvancedQuantizer.quantize_uniform(tensor, bits, symmetric)
        result = torch.zeros_like(tensor)
        qmax   = float(2 ** (bits - 1) - 1)
        for i in range(tensor.shape[0]):
            ch = tensor[i]
            if symmetric:
                scale = (ch.abs().max() / qmax if qmax > 0
                         else torch.tensor(1.0, device=tensor.device)).clamp(min=1e-8)
                result[i] = torch.round(ch / scale).clamp(-qmax, qmax) * scale
            else:
                mn, mx = ch.min(), ch.max()
                scale = ((mx - mn) / (2 * qmax) if qmax > 0
                         else torch.tensor(1.0, device=tensor.device)).clamp(min=1e-8)
                zp = -torch.round(mn / scale) - qmax
                result[i] = (torch.round(ch / scale + zp).clamp(-qmax, qmax) - zp) * scale
        return result

    @staticmethod
    def quantize_learned(tensor, bits):
        """Per-channel sign-magnitude quantization.

        One explicit bit per output channel stores the sign; the remaining
        (bits-1) bits uniformly quantize the absolute magnitude per channel.
        This is operationally distinct from per-channel: zero-crossings are
        represented exactly and magnitude resolution is one bit finer.
        """
        if bits >= 32: return tensor
        mag_bits = max(1, bits - 1)          # reserve 1 bit for sign
        qmax     = float(2 ** mag_bits - 1)
        if tensor.dim() < 2:
            # scalar/1-D: tensor-wise sign-magnitude
            sign      = torch.where(tensor >= 0,
                                    torch.ones_like(tensor),
                                    -torch.ones_like(tensor))
            magnitude = tensor.abs()
            scale     = (magnitude.max() / qmax).clamp(min=1e-8)
            q_mag     = torch.round(magnitude / scale).clamp(0, qmax)
            return sign * q_mag * scale
        result = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            ch        = tensor[i]
            sign      = torch.where(ch >= 0,
                                    torch.ones_like(ch),
                                    -torch.ones_like(ch))
            magnitude = ch.abs()
            max_mag   = magnitude.max().clamp(min=1e-8)
            scale     = (max_mag / qmax).clamp(min=1e-8)
            q_mag     = torch.round(magnitude / scale).clamp(0, qmax)
            result[i] = sign * q_mag * scale
        return result

    @staticmethod
    def quantize(tensor, bits, mode='uniform', quant_type='INT', **kwargs):
        symmetric = (quant_type == 'INT')
        if   mode == 'uniform':     return AdvancedQuantizer.quantize_uniform(tensor, bits, symmetric)
        elif mode == 'log':         return AdvancedQuantizer.quantize_log(tensor, bits)
        elif mode == 'per-channel': return AdvancedQuantizer.quantize_per_channel(tensor, bits, symmetric)
        elif mode == 'learned':     return AdvancedQuantizer.quantize_learned(tensor, bits)
        else:                       return AdvancedQuantizer.quantize_uniform(tensor, bits, symmetric)


class AdvancedPruner:
    @staticmethod
    def compute_importance_scores(weight: torch.Tensor, method: str = 'l2', gradient: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Adapt for Conv2d by flattening appropriately
        w_flat = weight.view(weight.shape[0], -1)
        if method == 'l2': return torch.norm(w_flat, p=2, dim=1)
        elif method == 'l1': return torch.norm(w_flat, p=1, dim=1)
        elif method == 'fisher':
            if gradient is not None:
                g_flat = gradient.view(gradient.shape[0], -1)
                fisher = (w_flat * g_flat) ** 2
                return fisher.sum(dim=1)
            else: return torch.norm(w_flat, p=2, dim=1)
        else: return torch.norm(w_flat, p=2, dim=1)
    
    @staticmethod
    def create_neuron_mask(weight: torch.Tensor, keep_ratio: float, importance_method: str = 'l2', gradient: Optional[torch.Tensor] = None) -> torch.Tensor:
        out_features = weight.shape[0]
        n_keep = int(out_features * keep_ratio)
        if n_keep == 0: return torch.zeros(out_features, device=weight.device)
        importance = AdvancedPruner.compute_importance_scores(weight, method=importance_method, gradient=gradient)
        _, indices = torch.topk(importance, n_keep, largest=True)
        mask = torch.zeros(out_features, device=weight.device)
        mask[indices] = 1.0
        return mask
    
    @staticmethod
    def apply_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 4: # Conv2d
            return tensor * mask.view(-1, 1, 1, 1)
        elif len(tensor.shape) == 2: # Linear
            return tensor * mask.unsqueeze(1)
        elif len(tensor.shape) == 1:
            return tensor * mask
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

# ============================================================================
# SENSITIVITY ESTIMATION
# ============================================================================

class CNNSensitivityEstimator:
    """CNN adapted Sensitivity Estimator"""
    def __init__(self, model: nn.Module, dataloader: DataLoader, model_blocks: List[List[Tuple[str, nn.Module]]], device: str = 'cuda'):
        self.model = model; self.dataloader = dataloader; self.device = device
        self.model_blocks = model_blocks; self.num_blocks = len(model_blocks)
        self.fisher_dict = {}; self.layer_sensitivity = {}
    
    def compute_fisher_information(self, num_samples: int = 50) -> Dict[str, torch.Tensor]:
        self.model.eval(); fisher_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad: fisher_dict[name] = torch.zeros_like(param.data)
        criterion = nn.CrossEntropyLoss(); count = 0; inputs_list, targets_list = [], []
        samples_collected = 0
        for x, y in self.dataloader:
            if samples_collected >= num_samples: break
            inputs_list.append(x); targets_list.append(y); samples_collected += x.size(0)
        if not inputs_list: return {}
        inputs = torch.cat(inputs_list).to(self.device); targets = torch.cat(targets_list).to(self.device)
        batch_size = self.dataloader.batch_size
        num_batches = (inputs.size(0) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Computing Fisher Info", leave=False):
            start = i * batch_size; end = min(start + batch_size, inputs.size(0))
            inp_batch = inputs[start:end]; tgt_batch = targets[start:end]
            self.model.zero_grad(); outputs = self.model(inp_batch); loss = criterion(outputs, tgt_batch)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            count += inp_batch.size(0)
        for name in fisher_dict: fisher_dict[name] /= count
        self.fisher_dict = fisher_dict
        return fisher_dict
    
    def get_layer_sensitivity_scores(self) -> Dict[int, float]:
        layer_scores = defaultdict(float)
        for name, fisher in self.fisher_dict.items():
            for block_idx, block_layers in enumerate(self.model_blocks):
                if any(name == l_name or name.startswith(l_name + '.') for l_name, _ in block_layers):
                    layer_scores[block_idx] += fisher.sum().item()
                    break
        if layer_scores:
            max_score = max(layer_scores.values())
            if max_score > 0: layer_scores = {k: v / max_score for k, v in layer_scores.items()}
        self.layer_sensitivity = dict(layer_scores)
        return self.layer_sensitivity

# ============================================================================
# SURROGATE MODEL (ENHANCED TRACKING)
# ============================================================================

class AccuracySurrogateModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []; prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim)); layers.append(nn.ReLU()); layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1)); layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x) * 100.0

class SurrogateModelTrainer:
    # Normalisation maps (matching ViT surrogate)
    _TYPE_NORM = {'INT': 0.0, 'FLOAT': 1.0}
    _GRAN_NORM = {'uniform': 0.00, 'log': 0.33, 'per-channel': 0.67, 'learned': 1.00}
    _FEATURES_PER_KERNEL = 5

    def __init__(self, num_blocks: int, num_kernels_per_block: int = 4, hidden_dims: List[int] = [64, 32], device: str = 'cuda', baseline_accuracy: float = 85.0, logger: Optional[Any] = None):
        self.device = device; self.num_blocks = num_blocks; self.num_kernels_per_block = num_kernels_per_block
        self.baseline_accuracy = baseline_accuracy; self.logger = logger
        self.input_dim = num_blocks * num_kernels_per_block * self._FEATURES_PER_KERNEL
        self.model = AccuracySurrogateModel(self.input_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.MSELoss(); self.config_buffer = []; self.accuracy_buffer = []
        self.best_loss = float('inf'); self.best_state = None
        self.training_history = {'epochs': [], 'losses': [], 'samples': [], 'train_times': []}
        self.total_training_epochs = 0; self.total_training_time = 0.0
        
    def encode_config(self, configs: Dict[int, Any]) -> torch.Tensor:
        features = []
        for block_idx in range(self.num_blocks):
            if block_idx not in configs:
                features.extend([8.0, 1.0, 0.0, 0.0, 0.0] * self.num_kernels_per_block)
                continue
            config = configs[block_idx]
            kernel_configs = [config.kernel1_config, config.kernel2_config, config.kernel3_config, config.kernel4_config]
            for kc in kernel_configs:
                if kc is None:
                    features.extend([8.0, 1.0, 0.0, 0.0, 0.0])
                else:
                    bits_norm     = kc.weight_bits / 10.0
                    prune_norm    = kc.pruning_ratio
                    type_norm     = self._TYPE_NORM.get(kc.quant_type, 0.0)
                    gran_norm     = self._GRAN_NORM.get(kc.quant_mode, 0.0)
                    combined_flag = 1.0 if (kc.pruning_ratio < 1.0 and kc.weight_bits < 8) else 0.5
                    features.extend([bits_norm, prune_norm, type_norm, gran_norm, combined_flag])
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def add_sample(self, configs: Dict[int, Any], post_finetune_accuracy: float):
        config_tensor = self.encode_config(configs)
        self.config_buffer.append(config_tensor); self.accuracy_buffer.append(post_finetune_accuracy)
    
    def train(self, epochs: int = 50, batch_size: int = 32):
        if len(self.config_buffer) < 3: return
        start_time = time.time()
        X = torch.stack(self.config_buffer); y = torch.tensor(self.accuracy_buffer, dtype=torch.float32, device=self.device).unsqueeze(1)
        dataset = TensorDataset(X, y); dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        self.model.train(); best_epoch_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad(); pred = self.model(batch_X); loss = self.criterion(pred, batch_y)
                loss.backward(); self.optimizer.step(); total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            self.training_history['epochs'].append(self.total_training_epochs + epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['samples'].append(len(self.config_buffer))
            if self.logger: self.logger.log_surrogate_training(self.total_training_epochs + epoch, avg_loss, len(self.config_buffer))
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss; self.best_state = copy.deepcopy(self.model.state_dict())
        
        if self.best_state is not None: self.model.load_state_dict(self.best_state)
        self.best_loss = best_epoch_loss; self.total_training_epochs += epochs
        training_time = time.time() - start_time
        self.total_training_time += training_time; self.training_history['train_times'].append(training_time)
    
    def predict(self, configs: Dict[int, Any]) -> float:
        if len(self.config_buffer) < 3: return self.baseline_accuracy - 2.0
        self.model.eval()
        with torch.no_grad():
            config_tensor = self.encode_config(configs).unsqueeze(0)
            pred = self.model(config_tensor)
            return pred.item()
    
    def get_buffer_size(self) -> int: return len(self.config_buffer)
    def update_baseline(self, baseline_accuracy: float): self.baseline_accuracy = baseline_accuracy
    def get_training_stats(self) -> Dict[str, Any]:
        return {
            'total_samples': len(self.config_buffer), 'total_epochs': self.total_training_epochs,
            'total_time': self.total_training_time, 'best_loss': self.best_loss,
            'avg_time_per_epoch': self.total_training_time / self.total_training_epochs if self.total_training_epochs > 0 else 0,
            'training_sessions': len(self.training_history['train_times']),
        }

# ============================================================================
# LAYER CONFIGURATION
# ============================================================================

@dataclass
class KernelConfig:
    name: str; weight_bits: int = 8; act_bits: int = 8; quant_type: str = 'INT'; quant_mode: str = 'uniform'
    pruning_ratio: float = 1.0; importance_method: str = 'l2'; shape: Tuple[int, ...] = field(default_factory=tuple)
    def to_dict(self) -> Dict[str, Any]: return asdict(self)
    def compression_ratio(self) -> float: return (self.weight_bits / 32.0) * self.pruning_ratio

@dataclass
class LayerConfig:
    block_idx: int; kernel1_config: Optional[KernelConfig] = None; kernel2_config: Optional[KernelConfig] = None
    kernel3_config: Optional[KernelConfig] = None; kernel4_config: Optional[KernelConfig] = None
    avg_weight_bits: int = 8; avg_act_bits: int = 8; avg_pruning_ratio: float = 1.0
    selected_by_agent: str = ''; agent_confidence: float = 0.0; assigned_budget: Optional[LayerBudget] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {'block_idx': self.block_idx, 'kernel1': self.kernel1_config.to_dict() if self.kernel1_config else None,
                'kernel2': self.kernel2_config.to_dict() if self.kernel2_config else None,
                'kernel3': self.kernel3_config.to_dict() if self.kernel3_config else None,
                'kernel4': self.kernel4_config.to_dict() if self.kernel4_config else None,
                'avg_weight_bits': self.avg_weight_bits, 'avg_act_bits': self.avg_act_bits,
                'avg_pruning_ratio': self.avg_pruning_ratio, 'selected_by_agent': self.selected_by_agent, 'agent_confidence': self.agent_confidence}
    
    def update_aggregates(self):
        configs = [c for c in [self.kernel1_config, self.kernel2_config, self.kernel3_config, self.kernel4_config] if c is not None]
        if configs:
            self.avg_weight_bits = int(np.mean([c.weight_bits for c in configs]))
            self.avg_act_bits = int(np.mean([c.act_bits for c in configs]))
            self.avg_pruning_ratio = float(np.mean([c.pruning_ratio for c in configs]))
    
    def compression_ratio(self) -> float:
        configs = [c for c in [self.kernel1_config, self.kernel2_config, self.kernel3_config, self.kernel4_config] if c is not None]
        if not configs: return 1.0
        return float(np.mean([c.compression_ratio() for c in configs]))

# ============================================================================
# LOW-LEVEL AGENT ENVIRONMENT (BOTH PRUNING + QUANTIZATION)
# ============================================================================

class BudgetConstrainedCompressionEnv(gym.Env):
    """
    Per-block MDP for the Low-Level Agent ensemble.

    Action space (per kernel): [bits_idx, pruning_idx, type_idx, gran_idx]
      bits_idx   : 0-14  → bitwidth b_{i,k}
      pruning_idx: 0-14  → keep-ratio ρ_{i,k}
      type_idx   : 0-1   → τ_{i,k} ∈ {INT, FLOAT}   (overridden by quantization_type)
      gran_idx   : 0-3   → μ_{i,k} ∈ {uniform, log, per-channel, learned}
                           (overridden / defaulted by strategy when not 'mixed')

    --------------------------
    quantization_type : str
        'mixed'  → agent chooses INT or FLOAT freely  (default paper behaviour)
        'int'    → all kernels locked to INT
        'float'  → all kernels locked to FLOAT
    default_strategy  : str
        Granularity fallback / fixed mode when you do not want the agent to choose.
        Recognised values: 'uniform', 'log', 'per-channel', 'learned'.
        When set, the agent's gran_idx is IGNORED and this mode is used for every kernel.
        Pass None (default) to let the agent pick freely.
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader, eval_dataloader: DataLoader,
                 block_idx: int, sensitivity_score: float, global_goal: CompressionGoal,
                 device: str = 'cuda', curriculum_stage: int = 0,
                 layer_budget: Optional[LayerBudget] = None,
                 surrogate_model: Optional[SurrogateModelTrainer] = None,
                 model_blocks: List[List[Tuple[str, nn.Module]]] = None,
                 quantization_type: str = 'mixed',
                 default_strategy:  Optional[str] = None):
        super().__init__()
        self.model = model; self.dataloader = dataloader; self.eval_dataloader = eval_dataloader
        self.block_idx = block_idx; self.sensitivity_score = sensitivity_score
        self.global_goal = global_goal; self.device = device
        self.curriculum_stage = curriculum_stage; self.surrogate_model = surrogate_model
        self.model_blocks = model_blocks
        self.quantization_type = quantization_type.lower()   # 'mixed' | 'int' | 'float'
        self.default_strategy  = default_strategy             # None or a QUANT_MODES string

        self.layer_budget = layer_budget
        if self.layer_budget is None:
            self.layer_budget = LayerBudget(block_idx=block_idx, target_compression_ratio=0.25,
                max_accuracy_drop=1.0, priority=0.5, sensitivity=sensitivity_score,
                global_min_bits=global_goal.min_layer_bits, global_max_bits=global_goal.max_layer_bits,
                global_min_pruning=global_goal.min_layer_pruning, global_max_pruning=global_goal.max_layer_pruning)
        
        self.kernel_modules = self._identify_kernels()
        num_kernels = len(self.kernel_modules)
        
        # Action space: [bits, pruning, type, gran] per kernel  (matches ViT)
        self.action_space = spaces.MultiDiscrete([15, 15, 2, 4] * num_kernels)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(12 + num_kernels,), dtype=np.float32)
        
        self.cached_inputs = None; self.baseline_outputs = None; self.baseline_accuracy = None
        self.original_weights = {}; self.step_count = 0; self.reward_history = []
        if self.cached_inputs is None:
            self.cached_inputs, _ = DataManager.get_cached_batches(self.dataloader, self.device, num_batches=3)
    
    def _identify_kernels(self) -> Dict[str, nn.Module]:
        kernels = {}
        block_layers = self.model_blocks[self.block_idx]
        for idx, (name, module) in enumerate(block_layers):
            kernels[f'kernel{idx+1}'] = module
        return kernels
    
    def update_budget(self, new_budget: LayerBudget): self.layer_budget = new_budget
    
    def _decode_action(self, action: np.ndarray) -> LayerConfig:
        layer_config = LayerConfig(block_idx=self.block_idx); layer_config.assigned_budget = self.layer_budget
        kernel_names = list(self.kernel_modules.keys())
        budget = self.layer_budget
        min_bits    = max(budget.min_bits,    budget.global_min_bits)
        max_bits    = budget.global_max_bits
        max_pruning = min(budget.max_pruning, budget.global_max_pruning)
        min_pruning = budget.global_min_pruning
        
        for kernel_idx, kernel_name in enumerate(kernel_names):
            base_idx    = kernel_idx * 4
            bits_idx    = action[base_idx]
            pruning_idx = action[base_idx + 1]
            type_idx    = int(action[base_idx + 2])
            gran_idx    = int(action[base_idx + 3])
            module      = self.kernel_modules[kernel_name]

            # ── b_{i,k}: bitwidth ───────────────────────────────────────────
            bits = int(np.clip(
                min_bits + int(bits_idx * (max_bits - min_bits) / 14.0),
                min_bits, max_bits))

            # ── ρ_{i,k}: keep-ratio ─────────────────────────────────────────
            prune_amount = min_pruning + (pruning_idx / 14.0) * (max_pruning - min_pruning)
            keep_ratio   = float(np.clip(
                1.0 - prune_amount, 1.0 - max_pruning, 1.0 - min_pruning))

            # ── τ_{i,k}: quantization numeric type ──────────────────────────
            #   Governed by ExperimentConfig.quantization_type
            if   self.quantization_type == 'int':   quant_type = 'INT'
            elif self.quantization_type == 'float':  quant_type = 'FLOAT'
            else:                                    # 'mixed' → agent decides
                quant_type = QUANT_TYPES[type_idx % len(QUANT_TYPES)]

            # ── μ_{i,k}: granularity scheme ─────────────────────────────────
            #   When default_strategy is set, it overrides the agent choice.
            if self.default_strategy and self.default_strategy in QUANT_MODES:
                quant_mode = self.default_strategy
            else:
                quant_mode = QUANT_MODES[gran_idx % len(QUANT_MODES)]

            kernel_config_obj = KernelConfig(
                name=kernel_name, weight_bits=bits, act_bits=bits,
                quant_type=quant_type, quant_mode=quant_mode,
                pruning_ratio=keep_ratio, importance_method='l2',
                shape=tuple(module.weight.shape))
            
            if kernel_name == 'kernel1': layer_config.kernel1_config = kernel_config_obj
            elif kernel_name == 'kernel2': layer_config.kernel2_config = kernel_config_obj
            elif kernel_name == 'kernel3': layer_config.kernel3_config = kernel_config_obj
            elif kernel_name == 'kernel4': layer_config.kernel4_config = kernel_config_obj
        layer_config.update_aggregates()
        return layer_config
    
    def _compute_baseline_outputs(self):
        if self.baseline_outputs is not None: return
        self.model.eval()
        with torch.no_grad(): self.baseline_outputs = self.model(self.cached_inputs).detach()
    
    def _apply_compression(self, config: LayerConfig):
        for idx, (name, module) in enumerate(self.model_blocks[self.block_idx]):
            kernel_name = f'kernel{idx+1}'
            if kernel_name == 'kernel1': kernel_config = config.kernel1_config
            elif kernel_name == 'kernel2': kernel_config = config.kernel2_config
            elif kernel_name == 'kernel3': kernel_config = config.kernel3_config
            elif kernel_name == 'kernel4': kernel_config = config.kernel4_config
            else: continue
            
            if kernel_config is None: continue
            if name not in self.original_weights: self.original_weights[name] = module.weight.data.clone()
            weights = self.original_weights[name].clone()
            
            if kernel_config.pruning_ratio < 1.0:
                mask = AdvancedPruner.create_neuron_mask(weights, kernel_config.pruning_ratio, importance_method=kernel_config.importance_method)
                weights = AdvancedPruner.apply_mask(weights, mask)
            weights = AdvancedQuantizer.quantize(
                weights, kernel_config.weight_bits,
                mode=kernel_config.quant_mode, quant_type=kernel_config.quant_type)
            module.weight.data = weights
    
    def _compute_reward(self, config: LayerConfig, full_model_configs: Optional[Dict[int, LayerConfig]] = None) -> Tuple[float, Dict[str, float]]:
        compression_ratio = config.compression_ratio()
        if self.surrogate_model is not None and self.surrogate_model.get_buffer_size() >= 3:
            if full_model_configs is None: full_model_configs = {self.block_idx: config}
            else: full_model_configs = {**full_model_configs, self.block_idx: config}
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
        
        if accuracy_drop < 1.0: accuracy_reward = 100.0
        elif accuracy_drop < 2.0: accuracy_reward = 95.0 - (accuracy_drop - 1.0) * 5
        elif accuracy_drop < 3.0: accuracy_reward = 85.0 - (accuracy_drop - 2.0) * 10
        else: accuracy_reward = max(0, 70.0 - (accuracy_drop - 3.0) * 10)
        
        compression_reward = (1.0 - compression_ratio) * 60
        actual_compression = compression_ratio; target_compression = self.layer_budget.target_compression_ratio
        compression_gap = abs(actual_compression - target_compression)
        budget_compliance_reward = np.exp(-10 * compression_gap) * 20
        sensitivity_penalty = (1.0 - compression_ratio) * self.sensitivity_score * 12
        stability_component = -np.std(self.reward_history[-5:]) * 2 if len(self.reward_history) > 5 else 0.0
        
        reward = (self.global_goal.alpha * accuracy_reward + self.global_goal.beta * compression_reward + budget_compliance_reward + self.global_goal.gamma * stability_component - sensitivity_penalty)
        if np.isnan(reward) or np.isinf(reward): reward = -10.0
        
        components = {
            'predicted_accuracy': predicted_accuracy, 'baseline_accuracy': self.baseline_accuracy if self.baseline_accuracy else 85.0,
            'accuracy_drop': accuracy_drop, 'compression_ratio': compression_ratio, 'accuracy_reward': accuracy_reward,
            'compression_reward': compression_reward, 'budget_compliance': budget_compliance_reward, 'compression_gap': compression_gap,
            'target_compression': target_compression, 'actual_compression': actual_compression, 'reward': reward,
            'stability': stability_component, 'used_surrogate': use_surrogate
        }
        return float(reward), components
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for name, orig_weight in self.original_weights.items():
            module = dict(self.model.named_modules())[name]
            module.weight.data = orig_weight.clone()
        if self.cached_inputs is None: self.cached_inputs, _ = DataManager.get_cached_batches(self.dataloader, self.device, num_batches=3)
        self.model.eval()
        with torch.no_grad(): self.baseline_outputs = self.model(self.cached_inputs).detach()
        num_kernels = len(self.kernel_modules); state = np.zeros(12 + num_kernels, dtype=np.float32)
        state[0] = self.block_idx / 12.0; state[1] = self.sensitivity_score; state[2] = 1.0; state[3] = 0.0; state[4] = 0.0
        state[5] = self.curriculum_stage / 2.0; state[6] = self.layer_budget.target_compression_ratio; state[7] = self.layer_budget.priority
        state[8] = self.layer_budget.max_accuracy_drop / 10.0; state[9] = 1.0
        state[10] = 1.0 if self.surrogate_model and self.surrogate_model.get_buffer_size() >= 3 else 0.0
        state[11] = 0.0
        for i in range(num_kernels): state[12 + i] = 0.5
        self.step_count = 0; self.reward_history = []
        return state, {}
    
    def step(self, action):
        config = self._decode_action(action)
        self._apply_compression(config)
        reward, components = self._compute_reward(config)
        self.reward_history.append(reward)
        num_kernels = len(self.kernel_modules); state = np.zeros(12 + num_kernels, dtype=np.float32)
        state[0] = self.block_idx / 12.0; state[1] = self.sensitivity_score; state[2] = components['actual_compression']
        state[3] = components['compression_gap']; state[4] = components['accuracy_drop'] / 10.0; state[5] = self.curriculum_stage / 2.0
        state[6] = self.layer_budget.target_compression_ratio; state[7] = self.layer_budget.priority; state[8] = self.layer_budget.max_accuracy_drop / 10.0
        state[9] = 1.0; state[10] = 1.0 if components['used_surrogate'] else 0.0; state[11] = reward / 100.0
        for i in range(num_kernels): state[12 + i] = 0.5
        self.step_count += 1; done = self.step_count >= 20
        return state, reward, done, False, {'config': config, 'reward_components': components}

    def __deepcopy__(self, memo):
        cls = self.__class__; result = cls.__new__(cls); memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['dataloader', 'eval_dataloader', 'cached_inputs', 'baseline_outputs', 'surrogate_model', 'model_blocks']: setattr(result, k, v)
            else: setattr(result, k, copy.deepcopy(v, memo))
        return result

class OptimizedEnsembleLowLevelAgent:
    def __init__(self, env: BudgetConstrainedCompressionEnv, num_agents: int = 3, algorithms: List[str] = None, weights_config: List[Dict[str, float]] = None):
        self.env = env; self.num_agents = num_agents
        if algorithms is None: algorithms = ['PPO', 'A2C', 'PPO'][:num_agents]
        self.algorithms = algorithms[:num_agents]; self.agents = []; self.agent_weights = np.ones(num_agents) / num_agents; self.voting_history = []
        for i, algo in enumerate(self.algorithms):
            def make_env_thunk(agent_index, w_config):
                def _thunk():
                    e = copy.copy(env); e.model = env.model; e.cached_inputs = env.cached_inputs
                    e.baseline_outputs = env.baseline_outputs; e.layer_budget = env.layer_budget
                    e.surrogate_model = env.surrogate_model; e.model_blocks = env.model_blocks
                    e.quantization_type = env.quantization_type; e.default_strategy = env.default_strategy
                    if w_config and agent_index < len(w_config):
                        w = w_config[agent_index]; e.global_goal = copy.deepcopy(env.global_goal)
                        e.global_goal.alpha = w['alpha']; e.global_goal.beta = w['beta']; e.global_goal.gamma = w['gamma']
                    return e
                return _thunk
            env_cmds = [make_env_thunk(i, weights_config) for _ in range(2)]
            vec_env = DummyVecEnv(env_cmds)
            if algo == 'PPO': agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4, n_steps=128, batch_size=32, ent_coef=0.01, seed=42 + i, max_grad_norm=0.5, device='cpu')
            elif algo == 'A2C': agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=7e-4, n_steps=64, ent_coef=0.01, seed=42 + i, max_grad_norm=0.5, device='cpu')
            else: agent = PPO('MlpPolicy', vec_env, verbose=0, seed=42 + i, max_grad_norm=0.5, device='cpu')
            self.agents.append({'agent': agent, 'algorithm': algo, 'env': vec_env, 'performance_history': []})
        self.voting_method = 'weighted'
    
    def update_budget(self, new_budget: LayerBudget):
        self.env.update_budget(new_budget)
        for agent_dict in self.agents:
            for env in agent_dict['env'].envs: env.update_budget(new_budget)
    
    def train(self, total_timesteps: int = 1000):
        for agent_dict in self.agents: agent_dict['agent'].learn(total_timesteps=total_timesteps)
    
    def _vote_on_action(self, actions: List[np.ndarray], method: str = 'weighted') -> np.ndarray:
        if method == 'weighted':
            voted_action = np.zeros(actions[0].shape, dtype=float)
            for i, action in enumerate(actions): voted_action += self.agent_weights[i] * action
            return np.round(voted_action).astype(int)
        else: return actions[0]
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, LayerConfig]:
        actions = []
        for agent_dict in self.agents:
            action, _ = agent_dict['agent'].predict(state, deterministic=True)
            actions.append(action)
        voted_action = self._vote_on_action(actions, method=self.voting_method)
        config = self.env._decode_action(voted_action)
        self.voting_history.append({'actions': [a.tolist() for a in actions], 'voted_action': voted_action.tolist(), 'weights': self.agent_weights.tolist()})
        return voted_action, config
    
    def get_config(self) -> LayerConfig:
        obs = self.env.reset()
        voted_action, config = self.predict(obs[0])
        best_agent_idx = np.argmax(self.agent_weights)
        config.selected_by_agent = f"{self.algorithms[best_agent_idx]}_agent_{best_agent_idx}"
        config.agent_confidence = float(self.agent_weights[best_agent_idx])
        return config


# ============================================================================
# TRUE HRL: HIGH-LEVEL AGENT ENVIRONMENT (DYNAMIC LEARNING ENABLED)
# ============================================================================

class BudgetAllocationEnvironment(gym.Env):
    def __init__(self, model: nn.Module, eval_dataloader: DataLoader,
                 sensitivity_scores: Dict[int, float],
                 global_goal: CompressionGoal, device: str = 'cuda',
                 num_blocks: int = 12):
        super().__init__()
        self.model = model; self.eval_dataloader = eval_dataloader
        self.sensitivity_scores = sensitivity_scores; self.global_goal = global_goal
        self.device = device; self.num_blocks = num_blocks
        
        self.action_space = spaces.MultiDiscrete([5, 3] * num_blocks)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5 + num_blocks * 2,), dtype=np.float32)
        
        self.baseline_accuracy = None; self.step_count = 0; self.current_budgets = {}
        
        self.last_acc_drop = 0.0; self.last_compression = 1.0; self.cycle_progress = 0.0
    
    def update_feedback(self, acc_drop: float, comp_ratio: float, cycle_prog: float):
        self.last_acc_drop = acc_drop; self.last_compression = comp_ratio; self.cycle_progress = cycle_prog

    def _evaluate_model(self) -> float:
        self.model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.eval_dataloader):
                if i >= 10: break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs); _, predicted = outputs.max(1)
                total += targets.size(0); correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
    
    def _decode_action(self, action: np.ndarray) -> Dict[int, LayerBudget]:
        budgets = {}
        compression_levels = {0: 0.35, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15}
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
            comp_level = action[base_idx]; strategy_idx = action[base_idx + 1]
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            
            if sensitivity > 0.7: comp_level = max(0, comp_level - 1)
            elif sensitivity < 0.3: comp_level = min(4, comp_level + 1)
            
            target_compression = compression_levels[comp_level]
            max_pruning = pruning_limits[comp_level]
            preferred_strategy = strategies[strategy_idx]
            
            if comp_level <= 1: min_bits = max(6, self.global_goal.min_layer_bits)
            elif comp_level <= 2: min_bits = max(5, self.global_goal.min_layer_bits)
            else: min_bits = max(4, self.global_goal.min_layer_bits)
            
            priority = sensitivity
            max_acc_drop = 0.5 if sensitivity > 0.7 else (1.0 if sensitivity > 0.5 else 2.0)
            
            budget = LayerBudget(
                block_idx=block_idx, target_compression_ratio=target_compression,
                max_accuracy_drop=max_acc_drop, priority=priority,
                sensitivity=sensitivity, preferred_strategy=preferred_strategy,
                min_bits=min_bits, max_pruning=max_pruning,
                global_min_bits=self.global_goal.min_layer_bits,
                global_max_bits=self.global_goal.max_layer_bits,
                global_min_pruning=self.global_goal.min_layer_pruning,
                global_max_pruning=self.global_goal.max_layer_pruning
            )
            budgets[block_idx] = budget
        return budgets
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.baseline_accuracy is None: self.baseline_accuracy = self._evaluate_model()
        
        state = np.zeros(5 + self.num_blocks * 2, dtype=np.float32)
        state[0] = self.last_acc_drop / 10.0   
        state[1] = self.last_compression      
        state[2] = self.cycle_progress        
        state[3] = self.global_goal.target_compression_ratio
        state[4] = self.global_goal.target_accuracy_drop / 10.0
        
        for i in range(self.num_blocks):
            state[5 + i * 2] = self.sensitivity_scores.get(i, 0.5)
            state[5 + i * 2 + 1] = 0.5
        
        self.step_count = 0; self.current_budgets = {}
        return state, {}
    
    def step(self, action):
        budgets = self._decode_action(action)
        self.current_budgets = budgets
        
        avg_target_compression = np.mean([b.target_compression_ratio for b in budgets.values()])
        compression_variance = np.var([b.target_compression_ratio for b in budgets.values()])
        balance_reward = -compression_variance * 10
        
        sensitivity_aware_reward = 0.0
        for block_idx, budget in budgets.items():
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            if sensitivity > 0.7 and budget.target_compression_ratio > 0.3:
                sensitivity_aware_reward += (budget.target_compression_ratio - 0.3) * 10
            elif sensitivity < 0.3 and budget.target_compression_ratio < 0.2:
                sensitivity_aware_reward += (0.2 - budget.target_compression_ratio) * 5
        
        global_compression_gap = abs(avg_target_compression - self.global_goal.target_compression_ratio)
        compression_reward = -global_compression_gap * 50
        
        dynamic_reward = 0.0
        if self.last_acc_drop > self.global_goal.target_accuracy_drop:
            if avg_target_compression < self.last_compression:
                dynamic_reward -= 50.0 * (self.last_acc_drop - self.global_goal.target_accuracy_drop)
        elif self.last_compression > self.global_goal.target_compression_ratio:
            if avg_target_compression < self.last_compression:
                dynamic_reward += 20.0
        
        reward = balance_reward + sensitivity_aware_reward + compression_reward + dynamic_reward
        
        state = np.zeros(5 + self.num_blocks * 2, dtype=np.float32)
        state[0] = self.last_acc_drop / 10.0; state[1] = self.last_compression
        state[2] = self.cycle_progress; state[3] = global_compression_gap
        state[4] = (balance_reward + sensitivity_aware_reward + dynamic_reward) / 20.0
        
        for i in range(self.num_blocks):
            state[5 + i * 2] = self.sensitivity_scores.get(i, 0.5)
            if i in budgets: state[5 + i * 2 + 1] = budgets[i].target_compression_ratio
        
        self.step_count += 1; done = self.step_count >= 10
        info = {'budgets': budgets, 'avg_compression': avg_target_compression}
        return state, reward, done, False, info


# ============================================================================
# ENSEMBLE HIGH-LEVEL AGENT (DYNAMIC UPDATES ENABLED)
# ============================================================================

class EnsembleHighLevelAgent:
    def __init__(self, model: nn.Module, eval_dataloader: DataLoader,
                 sensitivity_scores: Dict[int, float],
                 global_goal: CompressionGoal, device: str = 'cuda',
                 num_blocks: int = 12, num_hla_agents: int = 3,
                 weights_config: List[Dict[str, float]] = None):
        self.model = model; self.eval_dataloader = eval_dataloader
        self.sensitivity_scores = sensitivity_scores; self.global_goal = global_goal
        self.device = device; self.num_blocks = num_blocks; self.num_hla_agents = num_hla_agents
        
        self.hla_agents = []; algorithms = ['PPO', 'A2C', 'PPO'][:num_hla_agents]; self.budget_proposals = []
        
        for i in range(num_hla_agents):
            specific_goal = copy.deepcopy(global_goal)
            if weights_config and i < len(weights_config):
                w = weights_config[i]; specific_goal.alpha = w['alpha']; specific_goal.beta = w['beta']; specific_goal.gamma = w['gamma']
            
            env = BudgetAllocationEnvironment(
                model=model, eval_dataloader=eval_dataloader, sensitivity_scores=sensitivity_scores,
                global_goal=specific_goal, device=device, num_blocks=num_blocks
            )
            vec_env = DummyVecEnv([lambda e=env: e])
            
            algo = algorithms[i]
            if algo == 'PPO': agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=1e-4, n_steps=64, batch_size=32, ent_coef=0.01, seed=42 + i, max_grad_norm=0.5, device='cpu')
            elif algo == 'A2C': agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4, n_steps=64, ent_coef=0.01, seed=42 + i, max_grad_norm=0.5, device='cpu')
            else: agent = PPO('MlpPolicy', vec_env, verbose=0, seed=42 + i, max_grad_norm=0.5, device='cpu')
            
            self.hla_agents.append({'agent': agent, 'env': vec_env, 'algorithm': algo, 'weight': 1.0 / num_hla_agents})
    
    def update_environments(self, acc_drop: float, comp_ratio: float, cycle: int, max_cycles: int):
        progress = cycle / float(max_cycles)
        for hla_dict in self.hla_agents:
            for env in hla_dict['env'].envs:
                env.update_feedback(acc_drop, comp_ratio, progress)
                
    def train_hla_agents(self, total_timesteps: int = 2000):
        for hla_dict in self.hla_agents:
            hla_dict['agent'].learn(total_timesteps=total_timesteps)
    
    def allocate_budgets(self, deterministic: bool = True) -> Dict[int, LayerBudget]:
        all_budgets = []
        for hla_dict in self.hla_agents:
            obs = hla_dict['env'].reset()
            action, _ = hla_dict['agent'].predict(obs, deterministic=deterministic)
            current_env = hla_dict['env'].envs[0]
            budgets = current_env._decode_action(action[0])
            all_budgets.append(budgets)
        
        final_budgets = {}
        for block_idx in range(self.num_blocks):
            block_proposals = [budgets[block_idx] for budgets in all_budgets if block_idx in budgets]
            if not block_proposals: continue
            
            avg_target_compression = np.mean([b.target_compression_ratio for b in block_proposals])
            avg_max_acc_drop = np.mean([b.max_accuracy_drop for b in block_proposals])
            avg_priority = np.mean([b.priority for b in block_proposals])
            avg_max_pruning = np.mean([b.max_pruning for b in block_proposals])
            strategies = [b.preferred_strategy for b in block_proposals]
            most_common_strategy = max(set(strategies), key=strategies.count)
            min_bits_list = [b.min_bits for b in block_proposals]
            avg_min_bits = int(np.round(np.mean(min_bits_list)))
            
            final_budget = LayerBudget(
                block_idx=block_idx, target_compression_ratio=float(avg_target_compression),
                max_accuracy_drop=float(avg_max_acc_drop), priority=float(avg_priority),
                sensitivity=self.sensitivity_scores.get(block_idx, 0.5),
                preferred_strategy=most_common_strategy, min_bits=avg_min_bits,
                max_pruning=float(avg_max_pruning)
            )
            final_budgets[block_idx] = final_budget
        return final_budgets


# ============================================================================
# COMPREHENSIVE VISUALIZER (UNTOUCHED LOGIC, UPDATED LABELS FOR CNN)
# ============================================================================

class ComprehensiveVisualizer:
    @staticmethod
    def plot_pareto_front(cycle_history: List[Dict], save_path: Path):
        if not cycle_history: return
        accuracies = [c.get('accuracy', 0) for c in cycle_history]; compressions = [1.0/c.get('compression_ratio', 1) for c in cycle_history]; cycles = list(range(1, len(cycle_history) + 1))
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(compressions, accuracies, c=cycles, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        for i, (x, y, c) in enumerate(zip(compressions, accuracies, cycles)): ax.annotate(f'C{c}', (x, y), fontsize=9, ha='center')
        ax.set_xlabel('Compression Ratio (x)', fontsize=12); ax.set_ylabel('Accuracy (%)', fontsize=12); ax.set_title('Pareto Front: Compression vs Accuracy', fontsize=14, fontweight='bold'); ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax); cbar.set_label('Cycle', fontsize=11)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_compression_heatmap(configs: Dict[int, LayerConfig], save_path: Path):
        blocks = sorted(configs.keys()); kernels = ['Kernel 1', 'Kernel 2', 'Kernel 3', 'Kernel 4']; bitwidth_matrix = []; pruning_matrix = []
        for block_idx in blocks:
            config = configs[block_idx]; row_bits = []; row_prune = []
            for kernel_config in [config.kernel1_config, config.kernel2_config, config.kernel3_config, config.kernel4_config]:
                if kernel_config: row_bits.append(kernel_config.weight_bits); row_prune.append((1 - kernel_config.pruning_ratio) * 100)
                else: row_bits.append(8); row_prune.append(0)
            bitwidth_matrix.append(row_bits); pruning_matrix.append(row_prune)
        bitwidth_matrix = np.array(bitwidth_matrix).T; pruning_matrix = np.array(pruning_matrix).T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        im1 = ax1.imshow(bitwidth_matrix, cmap='viridis', aspect='auto')
        ax1.set_yticks(range(len(kernels))); ax1.set_yticklabels(kernels); ax1.set_xticks(range(len(blocks))); ax1.set_xticklabels([f'B{i}' for i in blocks]); ax1.set_title('Learned Bitwidth per Kernel', fontweight='bold'); plt.colorbar(im1, ax=ax1, label='Bits')
        im2 = ax2.imshow(pruning_matrix, cmap='Reds', aspect='auto')
        ax2.set_yticks(range(len(kernels))); ax2.set_yticklabels(kernels); ax2.set_xticks(range(len(blocks))); ax2.set_xticklabels([f'B{i}' for i in blocks]); ax2.set_title('Learned Pruning Ratio per Kernel', fontweight='bold'); plt.colorbar(im2, ax=ax2, label='Pruning %')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_cycle_progression(cycle_history: List[Dict], save_path: Path):
        if not cycle_history: return
        cycles = [c.get('cycle', i) for i, c in enumerate(cycle_history)]; accuracies = [c.get('accuracy', 0) for c in cycle_history]; compressions = [1.0/c.get('compression_ratio', 1) for c in cycle_history]; acc_drops = [c.get('accuracy_drop', 0) for c in cycle_history]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(cycles, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB'); axes[0].axhline(y=accuracies[0] if accuracies else 0, color='red', linestyle='--', label='Baseline', alpha=0.7); axes[0].set_ylabel('Accuracy (%)', fontsize=11); axes[0].set_title('Accuracy Over Cycles', fontweight='bold'); axes[0].grid(True, alpha=0.3); axes[0].legend()
        axes[1].plot(cycles, compressions, marker='s', linewidth=2, markersize=8, color='#A23B72'); axes[1].set_ylabel('Compression Ratio (x)', fontsize=11); axes[1].set_title('Compression Ratio Over Cycles', fontweight='bold'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(cycles, acc_drops, marker='^', linewidth=2, markersize=8, color='#F18F01'); axes[2].axhline(y=1.0, color='green', linestyle='--', label='Target (1%)', alpha=0.7); axes[2].set_xlabel('Cycle', fontsize=11); axes[2].set_ylabel('Accuracy Drop (%)', fontsize=11); axes[2].set_title('Accuracy Drop Over Cycles', fontweight='bold'); axes[2].grid(True, alpha=0.3); axes[2].legend()
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_budget_allocation(budgets: Dict[int, LayerBudget], sensitivity_scores: Dict[int, float], save_path: Path):
        blocks = sorted(budgets.keys()); target_compressions = [budgets[b].target_compression_ratio for b in blocks]; sensitivities = [sensitivity_scores.get(b, 0.5) for b in blocks]; strategies = [budgets[b].preferred_strategy for b in blocks]
        fig, axes = plt.subplots(2, 1, figsize=(14, 10)); x = np.arange(len(blocks)); width = 0.35
        axes[0].bar(x - width/2, [1.0/tc for tc in target_compressions], width, label='Target Compression (x)', color='#2E86AB', alpha=0.8); axes[0].bar(x + width/2, [s*10 for s in sensitivities], width, label='Sensitivity (x10)', color='#A23B72', alpha=0.8)
        axes[0].set_xlabel('Block Index', fontsize=12); axes[0].set_ylabel('Value', fontsize=12); axes[0].set_title('HLA Budget Allocation: Compression Targets vs Sensitivity', fontsize=14, fontweight='bold'); axes[0].set_xticks(x); axes[0].set_xticklabels([f'B{b}' for b in blocks]); axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3, axis='y')
        strategy_colors = {'quantization': '#00A878', 'pruning': '#E63946', 'auto': '#457B9D'}; strategy_counts = {'quantization': 0, 'pruning': 0, 'auto': 0}; colors = []
        for s in strategies: strategy_counts[s] += 1; colors.append(strategy_colors[s])
        axes[1].bar(blocks, [1]*len(blocks), color=colors, alpha=0.7, edgecolor='black'); axes[1].set_xlabel('Block Index', fontsize=12); axes[1].set_ylabel('Strategy', fontsize=12); axes[1].set_title('HLA Strategy Selection per Block', fontsize=14, fontweight='bold'); axes[1].set_yticks([])
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=strategy_colors['quantization'], label=f"Quantization ({strategy_counts['quantization']})"), Patch(facecolor=strategy_colors['pruning'], label=f"Pruning ({strategy_counts['pruning']})"), Patch(facecolor=strategy_colors['auto'], label=f"Auto ({strategy_counts['auto']})")]
        axes[1].legend(handles=legend_elements, fontsize=11, loc='upper right')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_solutions_comparison(cycle_history: List[Dict], save_path: Path):
        if not cycle_history: return
        df_data = [{'Cycle': e.get('cycle', 0), 'Accuracy (%)': e.get('accuracy', 0), 'Compression (x)': 1.0 / e.get('compression_ratio', 1), 'Acc Drop (%)': e.get('accuracy_drop', 0)} for e in cycle_history]; df = pd.DataFrame(df_data); fig, ax = plt.subplots(figsize=(12, 6)); x = df['Cycle']; width = 0.25; x_pos = np.arange(len(x))
        ax.bar(x_pos - width, df['Accuracy (%)'], width, label='Accuracy (%)', color='#2E86AB', alpha=0.8); ax.bar(x_pos, df['Compression (x)'], width, label='Compression (x)', color='#A23B72', alpha=0.8); ax.bar(x_pos + width, df['Acc Drop (%)'], width, label='Acc Drop (%)', color='#F18F01', alpha=0.8)
        ax.set_xlabel('Cycle', fontsize=12); ax.set_ylabel('Value', fontsize=12); ax.set_title('Solution Comparison Across Cycles', fontsize=14, fontweight='bold'); ax.set_xticks(x_pos); ax.set_xticklabels([f'C{c}' for c in x]); ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_surrogate_training(surrogate_history: List[Dict], save_path: Path):
        if not surrogate_history: return
        epochs = [h['epoch'] for h in surrogate_history]; losses = [h['loss'] for h in surrogate_history]; samples = [h['samples'] for h in surrogate_history]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(epochs, losses, marker='o', linewidth=2, markersize=6, color='#2E86AB'); ax1.set_xlabel('Epoch', fontsize=11); ax1.set_ylabel('Loss (MSE)', fontsize=11); ax1.set_title('Surrogate Model Training Loss', fontweight='bold', fontsize=14); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, samples, marker='s', linewidth=2, markersize=6, color='#A23B72'); ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Training Samples', fontsize=11); ax2.set_title('Surrogate Model Training Samples', fontweight='bold', fontsize=14); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_surrogate_predictions_vs_actual(predictions: List[Dict], save_path: Path):
        if not predictions: return
        cycles = [p['cycle'] for p in predictions]; predicted = [p['predicted_accuracy'] for p in predictions]; actual = [p['actual_accuracy'] for p in predictions]; errors = [p['error'] for p in predictions]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(cycles, predicted, marker='o', linewidth=2, markersize=8, label='Predicted', color='#2E86AB', alpha=0.7); ax1.plot(cycles, actual, marker='s', linewidth=2, markersize=8, label='Actual', color='#A23B72', alpha=0.7)
        ax1.set_xlabel('Cycle', fontsize=11); ax1.set_ylabel('Accuracy (%)', fontsize=11); ax1.set_title('Surrogate Predictions vs Actual Accuracy', fontweight='bold', fontsize=14); ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
        ax2.bar(cycles, errors, color='#F18F01', alpha=0.7, edgecolor='black'); ax2.set_xlabel('Cycle', fontsize=11); ax2.set_ylabel('Absolute Error (%)', fontsize=11); ax2.set_title('Surrogate Prediction Errors', fontweight='bold', fontsize=14); ax2.grid(True, alpha=0.3, axis='y')
        mean_error = np.mean(errors); ax2.axhline(y=mean_error, color='red', linestyle='--', label=f'Mean Error: {mean_error:.2f}%', linewidth=2); ax2.legend(fontsize=11)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    
    @staticmethod
    def plot_compression_strategy_distribution(configs: Dict[int, LayerConfig], save_path: Path):
        quant_only = 0; prune_only = 0; both = 0
        for config in configs.values():
            for kc in [config.kernel1_config, config.kernel2_config, config.kernel3_config, config.kernel4_config]:
                if kc is None: continue
                has_quant = kc.weight_bits < 8; has_prune = kc.pruning_ratio < 1.0
                if has_quant and has_prune: both += 1
                elif has_quant: quant_only += 1
                elif has_prune: prune_only += 1
        fig, ax = plt.subplots(figsize=(10, 6)); categories = ['Quantization\nOnly', 'Pruning\nOnly', 'Both Quant\n& Pruning']; values = [quant_only, prune_only, both]; colors = ['#2E86AB', '#E63946', '#00A878']
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        for bar, val in zip(bars, values): ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val}\n({val/(sum(values))*100:.1f}%)' if sum(values) > 0 else "0", ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Kernels', fontsize=12); ax.set_title('Compression Strategy Distribution Across All Kernels', fontsize=14, fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


# ============================================================================
# MAIN TRAINER (TRUE HRL ENABLED FOR CNN)
# ============================================================================

class CNNCompressionTrainer:
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        self.config = config; self.logger = logger; ReproducibilityManager.set_seed(42)
        if config.dataset == 'tinyimagenet': self.train_loader, self.test_loader = DataManager.get_tinyimagenet(config.batch_size)
        torch.cuda.empty_cache()
        self.model = timm.create_model(config.model_name, pretrained=True, num_classes=config.num_classes)
        self.model.to(config.device); self.original_model = copy.deepcopy(self.model).cpu()
        
        self.model_blocks = []; compressible_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if 'head' not in name and 'fc' not in name and 'classifier' not in name and 'downsample' not in name:
                    compressible_layers.append((name, module))
        for i in range(0, len(compressible_layers), 4):
            self.model_blocks.append(compressible_layers[i:i+4])
            
        self.num_blocks = len(self.model_blocks); self.criterion = nn.CrossEntropyLoss()
        
        if config.use_surrogate:
            self.surrogate = SurrogateModelTrainer(num_blocks=self.num_blocks, num_kernels_per_block=4, hidden_dims=config.surrogate_hidden_dims, device=config.device, baseline_accuracy=85.0, logger=logger)
            self.logger.log("Surrogate model initialized", level='SUCCESS')
        else: self.surrogate = None
        
        self.layer_agents = {}; self.hla = None; self.sensitivity_scores = {}; self.current_budgets = {}
        self.best_global_config = None; self.best_global_acc = 0.0; self.baseline_accuracy = 0.0; self.final_accuracy = 0.0
        self.visualizer = ComprehensiveVisualizer()

    def evaluate(self, model: nn.Module = None, max_batches: int = None) -> float:
        if model is None: model = self.model
        model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if max_batches and batch_idx >= max_batches: break
                inputs = inputs.to(self.config.device); targets = targets.to(self.config.device)
                outputs = model(inputs); _, predicted = outputs.max(1)
                total += targets.size(0); correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
    
    def finetune_with_early_stopping(self, model: nn.Module, max_epochs: int = 10, patience: int = 3, lr: float = 5e-5):
        self.logger.log(f"Fine-tuning (Max Epochs: {max_epochs})")
        masks = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name and param.dim() > 1:
                masks[name] = (param.data != 0).float().to(self.config.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        best_acc = 0.0; patience_counter = 0; best_model_state = copy.deepcopy(model.state_dict()); model.train()
        for epoch in range(max_epochs):
            running_loss = 0.0; pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                optimizer.zero_grad(); outputs = model(inputs); loss = self.criterion(outputs, targets)
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks: param.data *= masks[name]
                running_loss += loss.item(); pbar.set_postfix(loss=running_loss/(pbar.n+1))
            scheduler.step(); val_acc = self.evaluate(model, max_batches=20)
            self.logger.log(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
            if val_acc > best_acc: best_acc = val_acc; best_model_state = copy.deepcopy(model.state_dict()); patience_counter = 0
            else: patience_counter += 1
            if patience_counter >= patience: self.logger.log(f"Early stopping at epoch {epoch+1}"); break
        model.load_state_dict(best_model_state)
        return best_acc
    
    def compute_sensitivity(self):
        self.logger.log("Computing layer sensitivity...")
        self.sensitivity_estimator = CNNSensitivityEstimator(self.model, self.train_loader, self.model_blocks, self.config.device)
        self.sensitivity_estimator.compute_fisher_information(num_samples=50)
        self.sensitivity_scores = self.sensitivity_estimator.get_layer_sensitivity_scores()
    
    def pretrain_surrogate(self):
        if not self.surrogate: return
        self.logger.log("\n" + "="*80); self.logger.log("SURROGATE PRE-TRAINING PHASE"); self.logger.log("="*80)
        self.logger.log(f"Generating {self.config.surrogate_warmup_samples} random samples...")
        for sample_idx in tqdm(range(self.config.surrogate_warmup_samples), desc="Pre-training Surrogate", leave=False):
            random_configs = {}
            for block_idx in range(self.num_blocks):
                config = LayerConfig(block_idx=block_idx)
                for i in range(len(self.model_blocks[block_idx])):
                    kernel_name = f'kernel{i+1}'
                    kc = KernelConfig(name=kernel_name, weight_bits=np.random.randint(4, 9), act_bits=8, quant_type='INT', quant_mode='uniform', pruning_ratio=np.random.uniform(0.6, 1.0), shape=(64, 64))
                    if kernel_name == 'kernel1': config.kernel1_config = kc
                    elif kernel_name == 'kernel2': config.kernel2_config = kc
                    elif kernel_name == 'kernel3': config.kernel3_config = kc
                    elif kernel_name == 'kernel4': config.kernel4_config = kc
                config.update_aggregates(); random_configs[block_idx] = config
            temp_model = copy.deepcopy(self.model)
            for block_idx, config in random_configs.items():
                if block_idx >= len(self.model_blocks): continue
                k_configs = [config.kernel1_config, config.kernel2_config, config.kernel3_config, config.kernel4_config]
                for idx, (name, _) in enumerate(self.model_blocks[block_idx]):
                    kc = k_configs[idx]
                    if kc is None: continue
                    module = dict(temp_model.named_modules())[name]; weights = module.weight.data.clone()
                    if kc.pruning_ratio < 1.0:
                        mask = AdvancedPruner.create_neuron_mask(weights, kc.pruning_ratio)
                        weights = AdvancedPruner.apply_mask(weights, mask)
                    weights = AdvancedQuantizer.quantize(weights, kc.weight_bits); module.weight.data = weights
            acc = self.finetune_with_early_stopping(temp_model, max_epochs=1, patience=1, lr=1e-4)
            self.surrogate.add_sample(random_configs, acc); del temp_model; torch.cuda.empty_cache()
        self.logger.log("\nTraining surrogate on collected samples...")
        self.surrogate.train(epochs=100, batch_size=16)
        self.logger.log(f"Surrogate pre-training complete! Samples: {self.surrogate.get_buffer_size()}, Loss: {self.surrogate.best_loss:.4f}\n" + "="*80)
    
    def create_hla(self):
        self.logger.log("Creating High-Level Agent (Budget Allocator)...")
        self.hla = EnsembleHighLevelAgent(model=self.model, eval_dataloader=self.test_loader, sensitivity_scores=self.sensitivity_scores, global_goal=self.config.compression_goal, device=self.config.device, num_blocks=self.num_blocks, num_hla_agents=self.config.num_hla_agents, weights_config=self.config.hla_weights)
    
    def train_hla(self):
        self.logger.log("Training high-level agents")
        self.hla.train_hla_agents(total_timesteps=self.config.hla_timesteps)
    
    def get_budgets_from_hla(self, deterministic: bool = True) -> Dict[int, LayerBudget]:
        self.logger.log(f"HLA allocating budgets (Deterministic={deterministic})")
        budgets = self.hla.allocate_budgets(deterministic=deterministic)
        return budgets
    
    def create_lla_with_budget(self, block_idx: int, budget: LayerBudget):
        """Create LLA env with quantization_type and strategy from ExperimentConfig."""
        sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
        env = BudgetConstrainedCompressionEnv(
            self.model, self.train_loader, self.test_loader,
            block_idx, sensitivity, self.config.compression_goal,
            self.config.device, curriculum_stage=2,
            layer_budget=budget, surrogate_model=self.surrogate,
            model_blocks=self.model_blocks,
            quantization_type=self.config.quantization_type,
            default_strategy =self.config.strategy)
        env.cached_inputs, _ = DataManager.get_cached_batches(self.train_loader, self.config.device, num_batches=2)
        env.baseline_accuracy = self.baseline_accuracy
        lla = OptimizedEnsembleLowLevelAgent(env, num_agents=self.config.num_lla_agents, algorithms=self.config.rl_algorithms, weights_config=self.config.lla_weights)
        return lla
    
    def apply_configs_to_model(self, configs: Dict[int, LayerConfig]):
        for block_idx, config in configs.items():
            if block_idx >= len(self.model_blocks): continue
            block_layers = self.model_blocks[block_idx]
            k_configs = [config.kernel1_config, config.kernel2_config, config.kernel3_config, config.kernel4_config]
            for idx, (name, _) in enumerate(block_layers):
                kernel_config = k_configs[idx]
                if kernel_config is None: continue
                module = dict(self.model.named_modules())[name]
                weights = module.weight.data.clone()
                if kernel_config.pruning_ratio < 1.0:
                    mask = AdvancedPruner.create_neuron_mask(weights, kernel_config.pruning_ratio)
                    weights = AdvancedPruner.apply_mask(weights, mask)
                weights = AdvancedQuantizer.quantize(
                    weights, kernel_config.weight_bits,
                    mode=kernel_config.quant_mode, quant_type=kernel_config.quant_type)
                module.weight.data = weights
    
    def print_lla_voting_results(self, block_idx: int, lla: OptimizedEnsembleLowLevelAgent, config: LayerConfig):
        if not lla.voting_history: return
        self.logger.log(f"Block {block_idx}: LLA configuration finalized")
    
    def run_hierarchical_compression(self):
        self.logger.log("Hierarchical compression with dynamic learning enabled")
        self.logger.log(f"Quantization type: {self.config.quantization_type}")
        self.logger.log(f"Strategy: {self.config.strategy}")
        if self.surrogate: self.surrogate.update_baseline(self.baseline_accuracy)
        
        if self.surrogate and self.config.surrogate_warmup_samples > 0: self.pretrain_surrogate()
        
        self.compute_sensitivity()
        self.logger.log("Creating and training HLA"); self.create_hla(); self.train_hla()
        
        current_acc_drop = 0.0; current_compression = 1.0; num_cycles = 4
        
        for cycle in range(num_cycles):
            self.logger.log(f"\n{'='*80}"); self.logger.log(f"CYCLE {cycle+1}/{num_cycles}"); self.logger.log(f"{'='*80}")
            
            # TRUE HRL STEP 1: Provide real-world feedback to HLA environments
            self.hla.update_environments(current_acc_drop, current_compression, cycle, num_cycles)
            
            # TRUE HRL STEP 2: Let HLA train on the new state mapping inside the loop!
            if cycle > 0:
                self.logger.log("Adapting HLA policy based on previous cycle results")
                self.hla.train_hla_agents(total_timesteps=512)
            
            # TRUE HRL STEP 3: Explore in early cycles, Exploit in final cycle
            is_deterministic = (cycle >= num_cycles - 1)
            
            self.current_budgets = self.get_budgets_from_hla(deterministic=is_deterministic)
            save_dir = self.logger.get_run_dir()
            self.visualizer.plot_budget_allocation(self.current_budgets, self.sensitivity_scores, save_dir / f'budget_allocation_cycle_{cycle+1}.png')
            
            self.logger.log("Creating/Updating LLA agents with HLA budgets")
            for block_idx in range(self.num_blocks):
                budget = self.current_budgets.get(block_idx)
                if budget is None: continue
                if block_idx not in self.layer_agents: self.layer_agents[block_idx] = self.create_lla_with_budget(block_idx, budget)
                else: self.layer_agents[block_idx].update_budget(budget)
            
            self.logger.log("Training LLA agents")
            for block_idx, lla in self.layer_agents.items(): lla.train(total_timesteps=256)
            
            self.logger.log("Extracting configurations from LLA")
            configs = {}
            for block_idx, lla in self.layer_agents.items():
                config = lla.get_config(); configs[block_idx] = config; self.print_lla_voting_results(block_idx, lla, config)
            
            self.logger.log("\nApplying Compression and Fine-tuning")
            self.apply_configs_to_model(configs)
            
            predicted_acc = None
            if self.surrogate and self.surrogate.get_buffer_size() >= 3:
                predicted_acc = self.surrogate.predict(configs)
                print(f"\nSurrogate Prediction: {predicted_acc:.2f}%")
            
            post_finetune_acc = self.finetune_with_early_stopping(self.model, max_epochs=self.config.finetune_epochs, patience=3, lr=5e-5)
            
            if predicted_acc is not None:
                self.logger.log_surrogate_prediction(cycle + 1, predicted_acc, post_finetune_acc)
                print(f"Actual Accuracy: {post_finetune_acc:.2f}%\nPrediction Error: {abs(predicted_acc - post_finetune_acc):.2f}%\n")
            
            if self.surrogate:
                self.surrogate.add_sample(configs, post_finetune_acc)
                if self.surrogate.get_buffer_size() % self.config.surrogate_update_freq == 0:
                    self.logger.log("Training surrogate model...")
                    self.surrogate.train(epochs=50, batch_size=16)
            
            # TRUE HRL STEP 4: Store results for the next cycle's HLA state
            current_acc_drop = self.baseline_accuracy - post_finetune_acc
            avg_compression = np.mean([c.compression_ratio() for c in configs.values()])
            current_compression = avg_compression
            
            self.logger.log(f"\nCycle {cycle+1} Results:\n  Accuracy:     {post_finetune_acc:.2f}%\n  Acc Drop:     {current_acc_drop:.2f}%\n  Compression:  {1.0 / avg_compression if avg_compression > 0 else 1.0:.2f}x")
            
            self.logger.log_cycle_result(cycle + 1, {'accuracy': post_finetune_acc, 'accuracy_drop': current_acc_drop, 'compression_ratio': avg_compression, 'configs': {k: v.to_dict() for k, v in configs.items()}})
            
            if post_finetune_acc > self.best_global_acc:
                self.best_global_acc = post_finetune_acc; self.best_global_config = copy.deepcopy(configs); self.logger.save_checkpoint(self.model, f'best_cycle_{cycle+1}')
                self.logger.log("NEW GLOBAL BEST!", level='SUCCESS')
            torch.cuda.empty_cache(); gc.collect()
        
        self.final_accuracy = self.best_global_acc
        self.logger.log("\n" + "="*80); self.logger.log("FINAL RESULTS"); self.logger.log("="*80)
        self.logger.log(f"Baseline:   {self.baseline_accuracy:.2f}%\nFinal:      {self.final_accuracy:.2f}%\nDrop:       {self.baseline_accuracy - self.final_accuracy:.2f}%")
        if self.best_global_config: self.logger.log(f"Compression: {1.0/np.mean([c.compression_ratio() for c in self.best_global_config.values()]):.2f}x")
        self.generate_all_visualizations()
        return self.best_global_config
    
    def generate_all_visualizations(self):
        save_dir = self.logger.get_run_dir()
        cycle_history = self.logger.metrics.get('cycle_history', []); surrogate_history = self.logger.metrics.get('surrogate_history', []); surrogate_predictions = self.logger.metrics.get('surrogate_predictions', [])
        self.logger.log("\nGenerating Visualizations...")
        if self.best_global_config:
            self.visualizer.plot_compression_heatmap(self.best_global_config, save_dir / 'compression_heatmap.png')
            self.visualizer.plot_compression_strategy_distribution(self.best_global_config, save_dir / 'strategy_distribution.png')
        if cycle_history:
            self.visualizer.plot_pareto_front(cycle_history, save_dir / 'pareto_front.png')
            self.visualizer.plot_cycle_progression(cycle_history, save_dir / 'cycle_progression.png')
            self.visualizer.plot_solutions_comparison(cycle_history, save_dir / 'solutions_comparison.png')
        if surrogate_history:
            self.visualizer.plot_surrogate_training(surrogate_history, save_dir / 'surrogate_training.png')
        if surrogate_predictions:
            self.visualizer.plot_surrogate_predictions_vs_actual(surrogate_predictions, save_dir / 'surrogate_predictions.png')
        self.logger.log("Visualization generation complete")

# ============================================================================
# ENHANCED REPORTING
# ============================================================================

def generate_comprehensive_report(configs, baseline_acc, final_acc, logger, surrogate=None):
    print("\n" + "="*80); print("COMPREHENSIVE FINAL REPORT"); print("="*80)
    print(f"\nACCURACY METRICS"); print(f"{'Baseline Accuracy:':<25} {baseline_acc:.2f}%"); print(f"{'Final Accuracy:':<25} {final_acc:.2f}%"); print(f"{'Accuracy Drop:':<25} {baseline_acc - final_acc:.2f}%"); print(f"{'Relative Drop:':<25} {100*(baseline_acc - final_acc)/baseline_acc:.2f}%")
    both_strategies = 0; quant_only = 0; prune_only = 0; neither = 0; total_budget_compliance = 0; budget_compliant_count = 0; strategy_by_layer = []; compression_by_layer = []
    type_counts = {'INT': 0, 'FLOAT': 0}
    gran_counts  = {'uniform': 0, 'log': 0, 'per-channel': 0, 'learned': 0}

    for block_idx in sorted(configs.keys()):
        config = configs[block_idx]; budget = config.assigned_budget
        for name, k_config in [('kernel1', config.kernel1_config), ('kernel2', config.kernel2_config), ('kernel3', config.kernel3_config), ('kernel4', config.kernel4_config)]:
            if k_config is None: continue
            has_quant = k_config.weight_bits < 8; has_prune = k_config.pruning_ratio < 1.0
            if has_quant and has_prune: both_strategies += 1
            elif has_quant: quant_only += 1
            elif has_prune: prune_only += 1
            else: neither += 1
            type_counts[k_config.quant_type] = type_counts.get(k_config.quant_type, 0) + 1
            gran_counts[k_config.quant_mode] = gran_counts.get(k_config.quant_mode,  0) + 1
        if budget:
            actual = config.compression_ratio(); target = budget.target_compression_ratio; compliance = 1.0 - abs(actual - target) / target; total_budget_compliance += compliance
            if abs(actual - target) < 0.05: budget_compliant_count += 1
            strategy_by_layer.append(budget.preferred_strategy); compression_by_layer.append(1.0/actual)
    
    avg_budget_compliance = total_budget_compliance / len(configs) if configs else 0; total_kernels = both_strategies + quant_only + prune_only + neither
    print(f"\nCOMPRESSION STRATEGY DISTRIBUTION (BOTH ENABLED)")
    print(f"{'Both Quant + Prune:':<25} {both_strategies} ({100*both_strategies/total_kernels:.1f}%)" if total_kernels else f"{'Both Quant + Prune:':<25} 0 (0.0%)")
    print(f"{'Quantization Only:':<25} {quant_only} ({100*quant_only/total_kernels:.1f}%)" if total_kernels else f"{'Quantization Only:':<25} 0 (0.0%)")
    print(f"{'Pruning Only:':<25} {prune_only} ({100*prune_only/total_kernels:.1f}%)" if total_kernels else f"{'Pruning Only:':<25} 0 (0.0%)")
    print(f"{'Neither:':<25} {neither} ({100*neither/total_kernels:.1f}%)" if total_kernels else f"{'Neither:':<25} 0 (0.0%)")
    print(f"{'Total Kernels:':<25} {total_kernels}")

    print(f"\nQUANTIZATION TYPE DISTRIBUTION (τ_{{i,k}})")
    for t, count in type_counts.items():
        pct = 100*count/total_kernels if total_kernels else 0
        print(f"  {t:<12} {count} kernels ({pct:.1f}%)")
    print(f"\nGRANULARITY DISTRIBUTION (μ_{{i,k}})")
    for g, count in gran_counts.items():
        pct = 100*count/total_kernels if total_kernels else 0
        print(f"  {g:<14} {count} kernels ({pct:.1f}%)")

    if strategy_by_layer:
        from collections import Counter
        strategy_counts = Counter(strategy_by_layer); print(f"\nHLA STRATEGY SELECTION")
        for strategy, count in strategy_counts.items(): print(f"  {strategy.capitalize():<15} {count} layers ({100*count/len(strategy_by_layer):.1f}%)")
    
    compression = np.mean([c.compression_ratio() for c in configs.values()]); print(f"\nCOMPRESSION METRICS")
    print(f"{'Compression Ratio:':<25} {1.0/compression:.2f}x"); print(f"{'Model Size Reduction:':<25} {(1.0-compression)*100:.1f}%"); print(f"{'Budget Compliance:':<25} {avg_budget_compliance*100:.1f}%"); print(f"{'Budget Compliant Layers:':<25} {budget_compliant_count}/{len(configs)}")
    if compression_by_layer: print(f"{'Min Compression:':<25} {min(compression_by_layer):.2f}x"); print(f"{'Max Compression:':<25} {max(compression_by_layer):.2f}x"); print(f"{'Std Compression:':<25} {np.std(compression_by_layer):.2f}")
    
    if surrogate:
        stats = surrogate.get_training_stats(); print(f"\nSURROGATE MODEL PERFORMANCE"); print(f"{'Training Samples:':<25} {stats['total_samples']}"); print(f"{'Total Epochs:':<25} {stats['total_epochs']}"); print(f"{'Total Training Time:':<25} {stats['total_time']:.2f}s"); print(f"{'Avg Time/Epoch:':<25} {stats['avg_time_per_epoch']:.3f}s"); print(f"{'Training Sessions:':<25} {stats['training_sessions']}"); print(f"{'Final Loss (MSE):':<25} {stats['best_loss']:.6f}"); print(f"{'Model Parameters:':<25} {sum(p.numel() for p in surrogate.model.parameters())}")
        if logger.metrics.get('surrogate_predictions'):
            predictions = logger.metrics['surrogate_predictions']; errors = [p['error'] for p in predictions]; mean_error = np.mean(errors); std_error = np.std(errors); max_error = max(errors); min_error = min(errors)
            print(f"\nSURROGATE PREDICTION ACCURACY"); print(f"{'Mean Absolute Error:':<25} {mean_error:.2f}%"); print(f"{'Std Deviation:':<25} {std_error:.2f}%"); print(f"{'Min Error:':<25} {min_error:.2f}%"); print(f"{'Max Error:':<25} {max_error:.2f}%"); print(f"{'Predictions Made:':<25} {len(predictions)}")
    
    print(f"\nOUTPUT DIRECTORY\n  {logger.get_run_dir()}\n" + "="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    compression_goal = CompressionGoal(
        target_accuracy_drop=1.0, target_compression_ratio=0.25,
        min_layer_bits=3, max_layer_bits=8,
        min_layer_pruning=0.0, max_layer_pruning=0.6,
        alpha=50.0, beta=2.0, gamma=1.0
    )
    
    config = ExperimentConfig(
        model_name="mobilenetv2_100",        # Defaulting to ResNet18 (or use mobilenetv2_100)
        dataset="tinyimagenet",       # Switched to TinyImageNet
        num_classes=200,              # TinyImageNet has 200 classes
        do_finetune=True, finetune_epochs=15, batch_size=64,
        num_lla_agents=3, num_hla_agents=3, lla_timesteps=256, hla_timesteps=256,
        use_surrogate=True, hla_budget_update_freq=1,
        compression_goal=compression_goal, experiment_name='hrl_cnn_configurable',
        quantization_type='mixed',   # 'mixed' | 'int' | 'float'
        strategy=None,              # 'uniform' | 'log' | 'per-channel' | 'learned'
    )

    logger = ExperimentLogger(config.experiment_name, config.output_dir)
    logger.log_config(asdict(config))
    
    trainer = CNNCompressionTrainer(config, logger)
    configs = trainer.run_hierarchical_compression()
    
    logger.log_metric('baseline_accuracy', trainer.baseline_accuracy)
    logger.log_metric('final_accuracy', trainer.final_accuracy)
    if trainer.surrogate:
        stats = trainer.surrogate.get_training_stats()
        logger.log_metric('surrogate_samples', stats['total_samples'])
        logger.log_metric('surrogate_epochs', stats['total_epochs'])
        logger.log_metric('surrogate_loss', stats['best_loss'])
        logger.log_metric('surrogate_time', stats['total_time'])
    logger.save_metrics()
    
    generate_comprehensive_report(configs, trainer.baseline_accuracy, trainer.final_accuracy, logger, trainer.surrogate)


if __name__ == "__main__":
    main()