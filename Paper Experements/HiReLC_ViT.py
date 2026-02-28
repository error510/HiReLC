#!pip install torch torchvision timm pandas numpy matplotlib seaborn scikit-learn tqdm gymnasium stable_baselines3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch

# Set style
sns.set_theme(style="white", context="paper")
plt.rcParams['figure.dpi'] = 150

# ── granularity label map (shared by stats + plotting) ───────────────────────
GRAN_MAP_INT = {'uniform': 0, 'log': 1, 'per-channel': 2, 'learned': 3}
GRAN_MAP_STR = {0: 'UNI', 1: 'LOG', 2: 'PCH', 3: 'LRN'}

def get_kernel_stats(configs, num_blocks=12):
    kernels = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
    bits_mat  = np.zeros((len(kernels), num_blocks))
    prune_mat = np.zeros((len(kernels), num_blocks))
    type_mat  = np.zeros((len(kernels), num_blocks))
    gran_mat  = np.zeros((len(kernels), num_blocks))
    for block_idx in range(num_blocks):
        if block_idx not in configs: continue
        c = configs[block_idx]
        k_configs = [c.qkv_config, c.attn_proj_config, c.mlp_fc1_config, c.mlp_fc2_config]
        for k_i, k in enumerate(k_configs):
            if k:
                bits_mat [k_i, block_idx] = k.weight_bits
                prune_mat[k_i, block_idx] = (1.0 - k.pruning_ratio) * 100
                type_mat [k_i, block_idx] = 1 if k.quant_type == 'FLOAT' else 0
                gran_mat [k_i, block_idx] = GRAN_MAP_INT.get(k.quant_mode, 0)
            else:
                bits_mat [k_i, block_idx] = 32
                prune_mat[k_i, block_idx] = 0
                type_mat [k_i, block_idx] = 0
                gran_mat [k_i, block_idx] = 0
    return bits_mat, prune_mat, type_mat, gran_mat, kernels

def plot_per_kernel_decisions(configs, save_path=None):
    bits, prune, types, gran, y_labels = get_kernel_stats(configs)
    x_labels = [f"B{i}" for i in range(bits.shape[1])]
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    sns.heatmap(bits, annot=True, fmt='g', cmap="YlGnBu", ax=axes[0],
                xticklabels=x_labels, yticklabels=y_labels, cbar_kws={'label': 'Bits'})
    axes[0].set_title("Quantization: Weight Bits per Kernel", fontsize=12, fontweight='bold')
    sns.heatmap(prune, annot=True, fmt='.0f', cmap="Reds", ax=axes[1],
                xticklabels=x_labels, yticklabels=y_labels, cbar_kws={'label': '% Pruned'})
    axes[1].set_title("Sparsity: Percentage of Neurons Pruned", fontsize=12, fontweight='bold')
    cmap_type = sns.color_palette(["#3498db", "#e67e22"], as_cmap=True)
    sns.heatmap(types, annot=np.where(types == 1, "FP", "INT"), fmt="", cmap=cmap_type,
                ax=axes[2], cbar=False, xticklabels=x_labels, yticklabels=y_labels)
    axes[2].set_title("Data Type: INT vs FLOAT Decision (τ_{i,k})", fontsize=12, fontweight='bold')
    gran_annot = np.vectorize(GRAN_MAP_STR.get)(gran.astype(int))
    cmap_gran = sns.color_palette(["#27ae60", "#8e44ad", "#e74c3c", "#f39c12"], as_cmap=True)
    sns.heatmap(gran, annot=gran_annot, fmt="", cmap=cmap_gran, ax=axes[3],
                cbar=False, xticklabels=x_labels, yticklabels=y_labels, vmin=0, vmax=3)
    axes[3].set_title(
        "Granularity: UNI=uniform / LOG=log / PCH=per-channel / LRN=learned (μ_{i,k})",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_acc_vs_reduction(trainer, configs, save_path=None):
    def get_size_mb(model):
        return sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024**2)
    def get_compressed_size_mb(configs, model):
        total_bits = 0
        for name, param in model.named_parameters():
            bits = 32
            for b_idx, conf in configs.items():
                if (f"blocks.{b_idx}" in name and "weight" in name and param.dim() > 1):
                    if   "qkv"  in name and conf.qkv_config:       bits = conf.qkv_config.weight_bits * conf.qkv_config.pruning_ratio
                    elif "proj" in name and conf.attn_proj_config:  bits = conf.attn_proj_config.weight_bits * conf.attn_proj_config.pruning_ratio
                    elif "fc1"  in name and conf.mlp_fc1_config:    bits = conf.mlp_fc1_config.weight_bits * conf.mlp_fc1_config.pruning_ratio
                    elif "fc2"  in name and conf.mlp_fc2_config:    bits = conf.mlp_fc2_config.weight_bits * conf.mlp_fc2_config.pruning_ratio
                    break
            total_bits += param.numel() * bits
        return total_bits / (8 * 1024 * 1024)
    orig_size    = get_size_mb(trainer.original_model)
    comp_size    = get_compressed_size_mb(configs, trainer.original_model)
    baseline_acc = trainer.baseline_accuracy
    final_acc    = trainer.final_accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([orig_size], [baseline_acc], color='gray', s=100, label='Baseline', marker='o')
    ax.scatter([comp_size], [final_acc], color='green', s=150, label='HRL Agent Result', marker='*')
    txt = (f" -{(1 - comp_size/orig_size)*100:.1f}% Size\n {(final_acc - baseline_acc):.2f}% Acc")
    ax.annotate(txt, (comp_size, final_acc), xytext=(10, -20), textcoords='offset points')
    ax.set_title("Pareto Frontier: Accuracy vs Model Size", fontsize=12, fontweight='bold')
    ax.set_xlabel("Model Size (MB)"); ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
    ax.set_xlim(0, orig_size * 1.2)
    if save_path: plt.savefig(save_path)
    plt.show()

def plot_sensitivity_analysis(trainer, configs):
    sens_scores = []; comp_ratios = []; blocks = []
    for b_idx, score in trainer.sensitivity_scores.items():
        if b_idx in configs:
            sens_scores.append(score)
            comp_ratios.append(configs[b_idx].compression_ratio())
            blocks.append(b_idx)
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(sens_scores, comp_ratios, c=blocks, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Block Index')
    if len(sens_scores) > 1:
        z = np.polyfit(sens_scores, comp_ratios, 1); p = np.poly1d(z)
        plt.plot(sens_scores, p(sens_scores), "r--", alpha=0.5, label="Trend")
    plt.xlabel("Fisher Sensitivity Score (Higher = More Important)")
    plt.ylabel("Compression Ratio (Lower = More Compressed)")
    plt.title("Sensitivity vs Compression Analysis")
    plt.grid(True, alpha=0.3); plt.legend(); plt.show()

def print_comprehensive_summary(trainer, configs):
    print("\n" + "=" * 60); print(" FINAL HRL COMPRESSION REPORT"); print("=" * 60)
    acc_drop = trainer.baseline_accuracy - trainer.final_accuracy
    print(f"\n[1] PERFORMANCE")
    print(f"  Baseline Accuracy: {trainer.baseline_accuracy:.2f}%")
    print(f"  Final Accuracy:    {trainer.final_accuracy:.2f}%")
    print(f"  Delta:             {'-' if acc_drop > 0 else '+'}{abs(acc_drop):.2f}%")
    total_params = 0; total_kept = 0; total_bits_orig = 0; total_bits_comp = 0; quant_modes = []
    for b_idx, c in configs.items():
        for k in [c.qkv_config, c.attn_proj_config, c.mlp_fc1_config, c.mlp_fc2_config]:
            if not k: continue
            if k.shape:
                elems = np.prod(k.shape); kept = int(elems * k.pruning_ratio)
                total_params += elems; total_kept += kept
                total_bits_orig += elems * 32; total_bits_comp += kept * k.weight_bits
                quant_modes.append(f"{k.weight_bits}-bit {k.quant_type} {k.quant_mode}")
    sparsity       = 1.0 - (total_kept / total_params)   if total_params   else 0
    size_reduction = 1.0 - (total_bits_comp / total_bits_orig) if total_bits_orig else 0
    print(f"\n[2] COMPRESSION STATS")
    print(f"  Global Sparsity:  {sparsity*100:.2f}% (Neurons Pruned)")
    print(f"  Bitrate Reduction:{size_reduction*100:.2f}%")
    print(f"  Compression Ratio:{1/(1-size_reduction):.2f}x")
    print(f"\n[3] AGENT BEHAVIOR")
    from collections import Counter
    mode_counts = Counter(quant_modes)
    print("  Selected Configurations (Frequency):")
    for mode, count in mode_counts.most_common(5): print(f"    - {mode}: {count} kernels")
    print("\n" + "=" * 60)

# ─────────────────────────────────────────────────────────────────────────────

import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
torch.set_num_threads(10)

import sys, copy, json, time, warnings, platform
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision, torchvision.transforms as transforms
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
import timm
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

QUANT_TYPES = ['INT', 'FLOAT']
QUANT_MODES = ['uniform', 'log', 'per-channel', 'learned']

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CompressionGoal:
    target_accuracy_drop:    float = 1.0
    target_compression_ratio:float = 0.25
    target_flops_reduction:  float = 0.30
    min_layer_bits:  int   = 2
    max_layer_bits:  int   = 8
    min_layer_pruning: float = 0.0
    max_layer_pruning: float = 0.8
    alpha: float = 50.0
    beta:  float = 2.0
    gamma: float = 1.0
    delta: float = 1.0

@dataclass
class LayerBudget:
    block_idx:                int
    target_compression_ratio: float
    max_accuracy_drop:        float
    priority:                 float
    sensitivity:              float
    preferred_strategy: str   = 'auto'
    min_bits:           int   = 4
    max_pruning:        float = 0.4
    global_min_bits:    int   = 2
    global_max_bits:    int   = 8
    global_min_pruning: float = 0.0
    global_max_pruning: float = 0.8

@dataclass
class ExperimentConfig:
    model_name:   str = "vit_tiny_patch16_224"
    dataset:      str = "cifar10"
    num_classes:  int = 10
    do_finetune:  bool = True
    finetune_epochs: int = 10
    batch_size:   int = 128
    lr:           float = 3e-4
    weight_decay: float = 0.01

    num_lla_agents: int = 3
    num_hla_agents: int = 3
    lla_timesteps:  int = 2048
    hla_timesteps:  int = 2048
    rl_algorithms: List[str] = field(default_factory=lambda: ['PPO', 'A2C', 'PPO'])

    curriculum_stages:  int  = 3
    hla_budget_update_freq: int = 1

    use_surrogate: bool = True
    surrogate_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    surrogate_warmup_samples: int = 50
    surrogate_update_freq:    int = 1

    lla_weights: List[Dict[str, float]] = field(default_factory=lambda: [
        {'alpha': 30.0, 'beta': 1.5, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 5.0, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 8.0, 'gamma': 1.0},
    ])
    hla_weights: List[Dict[str, float]] = field(default_factory=lambda: [
        {'alpha': 30.0, 'beta':  6.0, 'gamma': 1.0},
        {'alpha': 30.0, 'beta': 10.0, 'gamma': 1.0},
        {'alpha': 30.0, 'beta':  4.0, 'gamma': 1.0},
    ])
    compression_goal: CompressionGoal = field(default_factory=CompressionGoal)

    quantization_type: str = 'log'
    strategy: str = None

    enable_pruning:       bool = True
    enable_quantization:  bool = True
    num_seeds:  int = 3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir: str = './'
    experiment_name: str = 'hrl_vit_complete'
    verbose: int = 1
    save_checkpoints: bool = True


class ReproducibilityManager:
    @staticmethod
    def set_seed(seed: int = 42):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random; random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = True

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        return {
            'timestamp':      datetime.now().isoformat(),
            'python_version': platform.python_version(),
            'pytorch_version':torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version':   torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'device_name':    torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'numpy_version':  np.__version__,
            'platform':       platform.platform(),
        }


class ExperimentLogger:
    def __init__(self, experiment_name: str, output_dir: str = './outputs'):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.run_dir = (self.output_dir / experiment_name /
                        datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file     = self.run_dir / 'experiment.log'
        self.metrics_file = self.run_dir / 'metrics.json'
        self.metrics = {
            'environment': ReproducibilityManager.get_environment_info(),
            'config': {}, 'results': {}, 'timing': {},
            'checkpoints': [], 'cycle_history': [],
            'surrogate_history': [], 'surrogate_predictions': []
        }
        self.start_time = time.time()
        self._init_log_file()

    def _init_log_file(self):
        with open(self.log_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"EXPERIMENT: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, level: str = 'INFO', print_console: bool = True):
        timestamp   = datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] [{level:5s}] {message}"
        if print_console:
            if   level == 'ERROR':   print(f"\033[91m{log_message}\033[0m")
            elif level == 'WARN':    print(f"\033[93m{log_message}\033[0m")
            elif level == 'SUCCESS': print(f"\033[92m{log_message}\033[0m")
            else:                    print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_config(self, config: Dict[str, Any]):
        self.metrics['config'] = config
        self.log("=" * 80); self.log("EXPERIMENT CONFIGURATION"); self.log("=" * 80)
        self.log(json.dumps(config, indent=2, default=str)); self.log("=" * 80)

    def log_metric(self, key: str, value: Any, step: Optional[int] = None):
        if step is not None:
            if key not in self.metrics['results']:
                self.metrics['results'][key] = {}
            self.metrics['results'][key][step] = value
        else:
            self.metrics['results'][key] = value

    def log_cycle_result(self, cycle: int, data: Dict[str, Any]):
        self.metrics['cycle_history'].append(
            {'cycle': cycle, 'timestamp': datetime.now().isoformat(), **data})

    def log_surrogate_training(self, epoch: int, loss: float, samples: int):
        self.metrics['surrogate_history'].append(
            {'epoch': epoch, 'loss': loss, 'samples': samples,
             'timestamp': datetime.now().isoformat()})

    def log_surrogate_prediction(self, cycle: int, predicted: float, actual: float):
        self.metrics['surrogate_predictions'].append({
            'cycle': cycle, 'predicted_accuracy': predicted,
            'actual_accuracy': actual, 'error': abs(predicted - actual),
            'timestamp': datetime.now().isoformat()})

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


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
class DataManager:
    @staticmethod
    def get_cifar10(batch_size=64, num_workers=10):
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        trainset = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=transform_train)
        testset  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        return (DataLoader(trainset, batch_size=batch_size, shuffle=True,  **kw),
                DataLoader(testset,  batch_size=batch_size, shuffle=False, **kw))

    @staticmethod
    def get_cifar100(batch_size=64, num_workers=10):
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))])
        trainset = torchvision.datasets.CIFAR100('./data', train=True,  download=True, transform=transform_train)
        testset  = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
        kw = dict(num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        return (DataLoader(trainset, batch_size=batch_size, shuffle=True,  **kw),
                DataLoader(testset,  batch_size=batch_size, shuffle=False, **kw))

    @staticmethod
    def get_cached_batches(dataloader, device, num_batches=5):
        cached_inputs, cached_targets = [], []
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches: break
            cached_inputs.append(x.to(device, non_blocking=True))
            cached_targets.append(y.to(device, non_blocking=True))
        return torch.cat(cached_inputs), torch.cat(cached_targets)


# ─────────────────────────────────────────────────────────────────────────────
# QUANTIZER
# ─────────────────────────────────────────────────────────────────────────────
@torch.jit.script
def _quantize_uniform_jit(tensor: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    if bits >= 32: return tensor
    if symmetric:
        qmax = float(2 ** (bits - 1) - 1); qmin = float(-(2 ** (bits - 1)))
        max_val = tensor.abs().max()
        scale = max_val / qmax if qmax > 0 else torch.tensor(1.0, device=tensor.device)
        zero_point = torch.tensor(0.0, device=tensor.device)
    else:
        qmin = 0.0; qmax = float(2 ** bits - 1)
        min_val, max_val = tensor.min(), tensor.max()
        scale = ((max_val - min_val) / (qmax - qmin)
                 if qmax > qmin else torch.tensor(1.0, device=tensor.device))
        zero_point = qmin - torch.round(min_val / scale)
    scale = torch.clamp(scale, min=1e-8)
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    return (quantized - zero_point) * scale

class AdvancedQuantizer:
    @staticmethod
    def quantize_uniform(tensor, bits, symmetric=True):
        return _quantize_uniform_jit(tensor, bits, symmetric)

    @staticmethod
    def quantize_log(tensor, bits):
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


# ─────────────────────────────────────────────────────────────────────────────
# PRUNER
# ─────────────────────────────────────────────────────────────────────────────
class AdvancedPruner:
    @staticmethod
    def compute_importance_scores(weight, method='l2', gradient=None):
        if method == 'l2':    return torch.norm(weight, p=2, dim=1)
        elif method == 'l1':  return torch.norm(weight, p=1, dim=1)
        elif method == 'fisher':
            if gradient is not None: return ((weight * gradient) ** 2).sum(dim=1)
            return torch.norm(weight, p=2, dim=1)
        return torch.norm(weight, p=2, dim=1)

    @staticmethod
    def create_neuron_mask(weight, keep_ratio, importance_method='l2', gradient=None):
        out_features = weight.shape[0]
        n_keep = int(out_features * keep_ratio)
        if n_keep == 0: return torch.zeros(out_features, device=weight.device)
        importance = AdvancedPruner.compute_importance_scores(weight, method=importance_method, gradient=gradient)
        _, indices = torch.topk(importance, n_keep, largest=True)
        mask = torch.zeros(out_features, device=weight.device)
        mask[indices] = 1.0
        return mask

    @staticmethod
    def apply_mask(tensor, mask):
        if len(tensor.shape) == 2: return tensor * mask.unsqueeze(1)
        elif len(tensor.shape) == 1: return tensor * mask
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
class ViTSensitivityEstimator:
    def __init__(self, model, dataloader, device='cuda', num_blocks=12):
        self.model = model; self.dataloader = dataloader
        self.device = device; self.num_blocks = num_blocks
        self.fisher_dict = {}; self.layer_sensitivity = {}

    def compute_fisher_information(self, num_samples=50):
        self.model.eval()
        fisher_dict = {n: torch.zeros_like(p.data)
                       for n, p in self.model.named_parameters() if p.requires_grad}
        criterion = nn.CrossEntropyLoss()
        inputs_list, targets_list, samples_collected = [], [], 0
        for x, y in self.dataloader:
            if samples_collected >= num_samples: break
            inputs_list.append(x); targets_list.append(y)
            samples_collected += x.size(0)
        if not inputs_list: return {}
        inputs  = torch.cat(inputs_list).to(self.device)
        targets = torch.cat(targets_list).to(self.device)
        batch_size  = self.dataloader.batch_size
        num_batches = (inputs.size(0) + batch_size - 1) // batch_size
        count = 0
        for i in tqdm(range(num_batches), desc="Computing Fisher Info", leave=False):
            s = i * batch_size; e = min(s + batch_size, inputs.size(0))
            self.model.zero_grad()
            loss = criterion(self.model(inputs[s:e]), targets[s:e])
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_dict[n] += p.grad.data ** 2
            count += (e - s)
        for n in fisher_dict: fisher_dict[n] /= count
        self.fisher_dict = fisher_dict
        return fisher_dict

    def get_layer_sensitivity_scores(self):
        layer_scores = defaultdict(float)
        for name, fisher in self.fisher_dict.items():
            if 'blocks.' in name:
                try:
                    block_idx = int(name.split('.')[1])
                    layer_scores[block_idx] += fisher.sum().item()
                except (IndexError, ValueError):
                    continue
        if layer_scores:
            max_score = max(layer_scores.values())
            if max_score > 0:
                layer_scores = {k: v / max_score for k, v in layer_scores.items()}
        self.layer_sensitivity = dict(layer_scores)
        return self.layer_sensitivity


# ─────────────────────────────────────────────────────────────────────────────
# SURROGATE MODEL
# ─────────────────────────────────────────────────────────────────────────────
class AccuracySurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()
        layers = []; prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.3)]
            prev_dim = h
        layers += [nn.Linear(prev_dim, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, x): return self.network(x) * 100.0


class SurrogateModelTrainer:
    _TYPE_NORM = {'INT': 0.0, 'FLOAT': 1.0}
    _GRAN_NORM = {'uniform': 0.00, 'log': 0.33, 'per-channel': 0.67, 'learned': 1.00}
    _FEATURES_PER_KERNEL = 5

    def __init__(self, num_blocks, num_kernels_per_block=4,
                 hidden_dims=[64, 32], device='cuda',
                 baseline_accuracy=85.0, logger=None):
        self.device = device; self.num_blocks = num_blocks
        self.num_kernels_per_block = num_kernels_per_block
        self.baseline_accuracy = baseline_accuracy; self.logger = logger
        self.input_dim = num_blocks * num_kernels_per_block * self._FEATURES_PER_KERNEL
        self.model     = AccuracySurrogateModel(self.input_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.MSELoss()
        self.config_buffer = []; self.accuracy_buffer = []
        self.best_loss = float('inf'); self.best_state = None
        self.training_history = {'epochs': [], 'losses': [], 'samples': [], 'train_times': []}
        self.total_training_epochs = 0; self.total_training_time = 0.0

    def encode_config(self, configs):
        features = []
        for block_idx in range(self.num_blocks):
            if block_idx not in configs:
                features.extend([8.0, 1.0, 0.0, 0.0, 0.0] * self.num_kernels_per_block)
                continue
            config = configs[block_idx]
            for kc in [config.qkv_config, config.attn_proj_config,
                       config.mlp_fc1_config, config.mlp_fc2_config]:
                if kc is None:
                    features.extend([8.0, 1.0, 0.0, 0.0, 0.0])
                else:
                    features.extend([
                        kc.weight_bits / 10.0,
                        kc.pruning_ratio,
                        self._TYPE_NORM.get(kc.quant_type, 0.0),
                        self._GRAN_NORM.get(kc.quant_mode, 0.0),
                        1.0 if (kc.pruning_ratio < 1.0 and kc.weight_bits < 8) else 0.5
                    ])
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def add_sample(self, configs, post_finetune_accuracy):
        self.config_buffer.append(self.encode_config(configs))
        self.accuracy_buffer.append(post_finetune_accuracy)

    def train(self, epochs=50, batch_size=32):
        if len(self.config_buffer) < 3: return
        start = time.time()
        X = torch.stack(self.config_buffer)
        y = torch.tensor(self.accuracy_buffer, dtype=torch.float32,
                         device=self.device).unsqueeze(1)
        dl = DataLoader(TensorDataset(X, y),
                        batch_size=min(batch_size, len(self.config_buffer)), shuffle=True)
        self.model.train(); best_epoch_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0.0
            for bX, by in dl:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(bX), by)
                loss.backward(); self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dl)
            self.training_history['epochs'].append(self.total_training_epochs + epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['samples'].append(len(self.config_buffer))
            if self.logger:
                self.logger.log_surrogate_training(
                    self.total_training_epochs + epoch, avg_loss, len(self.config_buffer))
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
        if self.best_state: self.model.load_state_dict(self.best_state)
        self.best_loss = best_epoch_loss
        self.total_training_epochs += epochs
        elapsed = time.time() - start
        self.total_training_time += elapsed
        self.training_history['train_times'].append(elapsed)

    def predict(self, configs):
        if len(self.config_buffer) < 3: return self.baseline_accuracy - 2.0
        self.model.eval()
        with torch.no_grad():
            return self.model(self.encode_config(configs).unsqueeze(0)).item()

    def get_buffer_size(self): return len(self.config_buffer)
    def update_baseline(self, b): self.baseline_accuracy = b

    def get_training_stats(self):
        return {
            'total_samples':    len(self.config_buffer),
            'total_epochs':     self.total_training_epochs,
            'total_time':       self.total_training_time,
            'best_loss':        self.best_loss,
            'avg_time_per_epoch': (self.total_training_time / self.total_training_epochs
                                   if self.total_training_epochs > 0 else 0),
            'training_sessions': len(self.training_history['train_times']),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LAYER CONFIGURATION DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class KernelConfig:
    name:             str
    weight_bits:      int   = 8
    act_bits:         int   = 8
    quant_type:       str   = 'INT'
    quant_mode:       str   = 'uniform'
    pruning_ratio:    float = 1.0
    importance_method:str   = 'l2'
    shape: Tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self): return asdict(self)
    def compression_ratio(self): return (self.weight_bits / 32.0) * self.pruning_ratio

@dataclass
class LayerConfig:
    block_idx:       int
    qkv_config:      Optional[KernelConfig] = None
    attn_proj_config:Optional[KernelConfig] = None
    mlp_fc1_config:  Optional[KernelConfig] = None
    mlp_fc2_config:  Optional[KernelConfig] = None
    avg_weight_bits: int   = 8
    avg_act_bits:    int   = 8
    avg_pruning_ratio:float= 1.0
    selected_by_agent:str  = ''
    agent_confidence: float= 0.0
    assigned_budget:  Optional[LayerBudget] = None

    def to_dict(self):
        return {
            'block_idx':       self.block_idx,
            'qkv':             self.qkv_config.to_dict()       if self.qkv_config       else None,
            'attn_proj':       self.attn_proj_config.to_dict() if self.attn_proj_config else None,
            'mlp_fc1':         self.mlp_fc1_config.to_dict()   if self.mlp_fc1_config   else None,
            'mlp_fc2':         self.mlp_fc2_config.to_dict()   if self.mlp_fc2_config   else None,
            'avg_weight_bits': self.avg_weight_bits,
            'avg_act_bits':    self.avg_act_bits,
            'avg_pruning_ratio':self.avg_pruning_ratio,
            'selected_by_agent':self.selected_by_agent,
            'agent_confidence': self.agent_confidence,
        }

    def update_aggregates(self):
        configs = [c for c in [self.qkv_config, self.attn_proj_config,
                               self.mlp_fc1_config, self.mlp_fc2_config] if c is not None]
        if configs:
            self.avg_weight_bits  = int(np.mean([c.weight_bits    for c in configs]))
            self.avg_act_bits     = int(np.mean([c.act_bits        for c in configs]))
            self.avg_pruning_ratio= float(np.mean([c.pruning_ratio for c in configs]))

    def compression_ratio(self):
        configs = [c for c in [self.qkv_config, self.attn_proj_config,
                               self.mlp_fc1_config, self.mlp_fc2_config] if c is not None]
        return float(np.mean([c.compression_ratio() for c in configs])) if configs else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LLA ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class BudgetConstrainedCompressionEnv(gym.Env):
    def __init__(self, model, dataloader, eval_dataloader,
                 block_idx, sensitivity_score, global_goal,
                 device='cuda', curriculum_stage=0,
                 layer_budget=None, surrogate_model=None,
                 quantization_type: str = 'mixed',
                 default_strategy:  Optional[str] = None):
        super().__init__()
        self.model           = model
        self.dataloader      = dataloader
        self.eval_dataloader = eval_dataloader
        self.block_idx       = block_idx
        self.sensitivity_score = sensitivity_score
        self.global_goal     = global_goal
        self.device          = device
        self.curriculum_stage= curriculum_stage
        self.surrogate_model = surrogate_model
        self.quantization_type = quantization_type.lower()
        self.default_strategy  = default_strategy

        self.layer_budget = layer_budget or LayerBudget(
            block_idx=block_idx, target_compression_ratio=0.25,
            max_accuracy_drop=1.0, priority=0.5, sensitivity=sensitivity_score,
            global_min_bits=global_goal.min_layer_bits,
            global_max_bits=global_goal.max_layer_bits,
            global_min_pruning=global_goal.min_layer_pruning,
            global_max_pruning=global_goal.max_layer_pruning)

        self.kernel_modules = self._identify_kernels()
        num_kernels = len(self.kernel_modules)
        self.action_space      = spaces.MultiDiscrete([15, 15, 2, 4] * num_kernels)
        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(12 + num_kernels,), dtype=np.float32)
        self.cached_inputs  = None
        self.baseline_outputs = None
        self.baseline_accuracy= None
        self.original_weights = {}
        self.step_count       = 0
        self.reward_history   = []
        if self.cached_inputs is None:
            self.cached_inputs, _ = DataManager.get_cached_batches(
                self.dataloader, self.device, num_batches=3)

    def _identify_kernels(self):
        kernels = {}
        prefix  = f'blocks.{self.block_idx}.'
        for name, module in self.model.named_modules():
            if 'head' in name or 'patch_embed' in name: continue
            if name.startswith(prefix) and isinstance(module, nn.Linear):
                if   'qkv'  in name:                                 kernels['qkv']       = module
                elif 'attn' in name and 'proj' in name:              kernels['attn_proj']  = module
                elif 'mlp'  in name and 'fc1' in name:               kernels['mlp_fc1']    = module
                elif 'mlp'  in name and 'fc2' in name:               kernels['mlp_fc2']    = module
        return kernels

    def update_budget(self, new_budget): self.layer_budget = new_budget

    def _decode_action(self, action: np.ndarray) -> LayerConfig:
        layer_config = LayerConfig(block_idx=self.block_idx)
        layer_config.assigned_budget = self.layer_budget
        budget      = self.layer_budget
        min_bits    = max(budget.min_bits,     budget.global_min_bits)
        max_bits    = budget.global_max_bits
        max_pruning = min(budget.max_pruning,  budget.global_max_pruning)
        min_pruning = budget.global_min_pruning

        for kernel_idx, kernel_name in enumerate(self.kernel_modules.keys()):
            base_idx    = kernel_idx * 4
            bits_idx    = action[base_idx]
            pruning_idx = action[base_idx + 1]
            type_idx    = int(action[base_idx + 2])
            gran_idx    = int(action[base_idx + 3])
            module      = self.kernel_modules[kernel_name]

            bits = int(np.clip(
                min_bits + int(bits_idx * (max_bits - min_bits) / 14.0),
                min_bits, max_bits))

            prune_amount = min_pruning + (pruning_idx / 14.0) * (max_pruning - min_pruning)
            keep_ratio   = float(np.clip(
                1.0 - prune_amount, 1.0 - max_pruning, 1.0 - min_pruning))

            if   self.quantization_type == 'int':   quant_type = 'INT'
            elif self.quantization_type == 'float':  quant_type = 'FLOAT'
            else:                                    quant_type = QUANT_TYPES[type_idx % len(QUANT_TYPES)]

            if self.default_strategy and self.default_strategy in QUANT_MODES:
                quant_mode = self.default_strategy
            else:
                quant_mode = QUANT_MODES[gran_idx % len(QUANT_MODES)]

            kc = KernelConfig(
                name=kernel_name, weight_bits=bits, act_bits=bits,
                quant_type=quant_type, quant_mode=quant_mode,
                pruning_ratio=keep_ratio, importance_method='l2',
                shape=tuple(module.weight.shape))

            if   kernel_name == 'qkv':       layer_config.qkv_config       = kc
            elif kernel_name == 'attn_proj': layer_config.attn_proj_config  = kc
            elif kernel_name == 'mlp_fc1':   layer_config.mlp_fc1_config    = kc
            elif kernel_name == 'mlp_fc2':   layer_config.mlp_fc2_config    = kc

        layer_config.update_aggregates()
        return layer_config

    def _compute_baseline_outputs(self):
        if self.baseline_outputs is not None: return
        self.model.eval()
        with torch.no_grad():
            self.baseline_outputs = self.model(self.cached_inputs).detach()

    def _apply_compression(self, config: LayerConfig):
        for kernel_name, module in self.kernel_modules.items():
            if   kernel_name == 'qkv':       kc = config.qkv_config
            elif kernel_name == 'attn_proj': kc = config.attn_proj_config
            elif kernel_name == 'mlp_fc1':   kc = config.mlp_fc1_config
            elif kernel_name == 'mlp_fc2':   kc = config.mlp_fc2_config
            else: continue
            if kc is None: continue
            if kernel_name not in self.original_weights:
                self.original_weights[kernel_name] = module.weight.data.clone()
            w = self.original_weights[kernel_name].clone()
            if kc.pruning_ratio < 1.0:
                mask = AdvancedPruner.create_neuron_mask(
                    w, kc.pruning_ratio, importance_method=kc.importance_method)
                w = AdvancedPruner.apply_mask(w, mask)
            w = AdvancedQuantizer.quantize(
                w, kc.weight_bits, mode=kc.quant_mode, quant_type=kc.quant_type)
            module.weight.data = w

    def _compute_reward(self, config, full_model_configs=None):
        compression_ratio = config.compression_ratio()
        if (self.surrogate_model and self.surrogate_model.get_buffer_size() >= 3):
            fmc = {**(full_model_configs or {}), self.block_idx: config}
            predicted_accuracy = self.surrogate_model.predict(fmc)
            baseline    = self.baseline_accuracy or 85.0
            accuracy_drop = baseline - predicted_accuracy
            use_surrogate = True
        else:
            self.model.eval()
            with torch.no_grad():
                compressed_outputs = self.model(self.cached_inputs)
            mse = F.mse_loss(compressed_outputs, self.baseline_outputs)
            accuracy_drop     = min(mse.item() * 20, 50.0)
            predicted_accuracy= 85.0 - accuracy_drop
            use_surrogate     = False

        if   accuracy_drop < 1.0: accuracy_reward = 100.0
        elif accuracy_drop < 2.0: accuracy_reward = 95.0 - (accuracy_drop - 1.0) * 5
        elif accuracy_drop < 3.0: accuracy_reward = 85.0 - (accuracy_drop - 2.0) * 10
        else:                     accuracy_reward = max(0, 70.0 - (accuracy_drop - 3.0) * 10)

        compression_reward = (1.0 - compression_ratio) * 60
        target_compression = self.layer_budget.target_compression_ratio
        compression_gap    = abs(compression_ratio - target_compression)
        budget_compliance_reward = np.exp(-10 * compression_gap) * 20
        sensitivity_penalty      = (1.0 - compression_ratio) * self.sensitivity_score * 12
        stability_component      = (-np.std(self.reward_history[-5:]) * 2
                                    if len(self.reward_history) > 5 else 0.0)
        reward = (self.global_goal.alpha   * accuracy_reward
                + self.global_goal.beta    * compression_reward
                + budget_compliance_reward
                + self.global_goal.gamma   * stability_component
                - sensitivity_penalty)
        if np.isnan(reward) or np.isinf(reward): reward = -10.0

        return float(reward), {
            'predicted_accuracy': predicted_accuracy,
            'baseline_accuracy':  self.baseline_accuracy or 85.0,
            'accuracy_drop':      accuracy_drop,
            'compression_ratio':  compression_ratio,
            'accuracy_reward':    accuracy_reward,
            'compression_reward': compression_reward,
            'budget_compliance':  budget_compliance_reward,
            'compression_gap':    compression_gap,
            'target_compression': target_compression,
            'actual_compression': compression_ratio,
            'reward':             reward,
            'stability':          stability_component,
            'used_surrogate':     use_surrogate,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for kn, ow in self.original_weights.items():
            if kn in self.kernel_modules:
                self.kernel_modules[kn].weight.data = ow.clone()
        if self.cached_inputs is None:
            self.cached_inputs, _ = DataManager.get_cached_batches(
                self.dataloader, self.device, num_batches=3)
        self.model.eval()
        with torch.no_grad():
            self.baseline_outputs = self.model(self.cached_inputs).detach()
        num_kernels = len(self.kernel_modules)
        state = np.zeros(12 + num_kernels, dtype=np.float32)
        state[0]  = self.block_idx / 12.0
        state[1]  = self.sensitivity_score
        state[2]  = 1.0; state[5] = self.curriculum_stage / 2.0
        state[6]  = self.layer_budget.target_compression_ratio
        state[7]  = self.layer_budget.priority
        state[8]  = self.layer_budget.max_accuracy_drop / 10.0; state[9] = 1.0
        state[10] = (1.0 if self.surrogate_model and
                     self.surrogate_model.get_buffer_size() >= 3 else 0.0)
        for i in range(num_kernels): state[12 + i] = 0.5
        self.step_count = 0; self.reward_history = []
        return state, {}

    def step(self, action):
        config = self._decode_action(action)
        self._apply_compression(config)
        reward, components = self._compute_reward(config)
        self.reward_history.append(reward)
        num_kernels = len(self.kernel_modules)
        state = np.zeros(12 + num_kernels, dtype=np.float32)
        state[0]  = self.block_idx / 12.0; state[1] = self.sensitivity_score
        state[2]  = components['actual_compression']
        state[3]  = components['compression_gap']
        state[4]  = components['accuracy_drop'] / 10.0
        state[5]  = self.curriculum_stage / 2.0
        state[6]  = self.layer_budget.target_compression_ratio
        state[7]  = self.layer_budget.priority
        state[8]  = self.layer_budget.max_accuracy_drop / 10.0; state[9] = 1.0
        state[10] = 1.0 if components['used_surrogate'] else 0.0
        state[11] = reward / 100.0
        for i in range(num_kernels): state[12 + i] = 0.5
        self.step_count += 1
        done = self.step_count >= 20
        return state, reward, done, False, {'config': config, 'reward_components': components}

    def __deepcopy__(self, memo):
        cls = self.__class__; result = cls.__new__(cls); memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['dataloader','eval_dataloader','cached_inputs',
                     'baseline_outputs','surrogate_model']:
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# LLA ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
class OptimizedEnsembleLowLevelAgent:
    def __init__(self, env, num_agents=3, algorithms=None, weights_config=None):
        self.env        = env
        self.num_agents = num_agents
        if algorithms is None: algorithms = ['PPO', 'A2C', 'PPO'][:num_agents]
        self.algorithms = algorithms[:num_agents]
        self.agents         = []
        self.agent_weights  = np.ones(num_agents) / num_agents
        self.voting_history = []

        for i, algo in enumerate(self.algorithms):
            def make_env_thunk(agent_index, w_config):
                def _thunk():
                    e = copy.copy(env)
                    e.model = env.model; e.cached_inputs = env.cached_inputs
                    e.baseline_outputs = env.baseline_outputs
                    e.layer_budget     = env.layer_budget
                    e.surrogate_model  = env.surrogate_model
                    if w_config and agent_index < len(w_config):
                        w = w_config[agent_index]
                        e.global_goal = copy.deepcopy(env.global_goal)
                        e.global_goal.alpha = w['alpha']
                        e.global_goal.beta  = w['beta']
                        e.global_goal.gamma = w['gamma']
                    return e
                return _thunk
            vec_env = DummyVecEnv([make_env_thunk(i, weights_config) for _ in range(2)])
            if   algo == 'PPO': agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4,
                                            n_steps=128, batch_size=32, ent_coef=0.01,
                                            seed=42+i, max_grad_norm=0.5, device='cpu')
            elif algo == 'A2C': agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=7e-4,
                                            n_steps=64, ent_coef=0.01, seed=42+i,
                                            max_grad_norm=0.5, device='cpu')
            else:               agent = PPO('MlpPolicy', vec_env, verbose=0,
                                            seed=42+i, max_grad_norm=0.5, device='cpu')
            self.agents.append({'agent': agent, 'algorithm': algo,
                                'env': vec_env, 'performance_history': []})
        self.voting_method = 'weighted'

    def update_budget(self, new_budget):
        self.env.update_budget(new_budget)
        for ad in self.agents:
            for e in ad['env'].envs: e.update_budget(new_budget)

    def train(self, total_timesteps=1000):
        for ad in self.agents: ad['agent'].learn(total_timesteps=total_timesteps)

    def _vote_on_action(self, actions, method='weighted'):
        if method == 'weighted':
            voted = np.zeros(actions[0].shape, dtype=float)
            for i, a in enumerate(actions): voted += self.agent_weights[i] * a
            return np.round(voted).astype(int)
        return actions[0]

    def predict(self, state):
        actions = [ad['agent'].predict(state, deterministic=True)[0] for ad in self.agents]
        voted   = self._vote_on_action(actions, method=self.voting_method)
        config  = self.env._decode_action(voted)
        self.voting_history.append({'actions': [a.tolist() for a in actions],
                                    'voted_action': voted.tolist(),
                                    'weights': self.agent_weights.tolist()})
        return voted, config

    def get_config(self):
        obs = self.env.reset()
        _, config = self.predict(obs[0])
        best = np.argmax(self.agent_weights)
        config.selected_by_agent = f"{self.algorithms[best]}_agent_{best}"
        config.agent_confidence  = float(self.agent_weights[best])
        return config


# ─────────────────────────────────────────────────────────────────────────────
# HLA ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class BudgetAllocationEnvironment(gym.Env):
    def __init__(self, model, eval_dataloader, sensitivity_scores,
                 global_goal, device='cuda', num_blocks=12):
        super().__init__()
        self.model = model; self.eval_dataloader = eval_dataloader
        self.sensitivity_scores = sensitivity_scores
        self.global_goal = global_goal; self.device = device; self.num_blocks = num_blocks
        self.action_space      = spaces.MultiDiscrete([5, 3] * num_blocks)
        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(5 + num_blocks * 2,), dtype=np.float32)
        self.baseline_accuracy = None; self.step_count = 0
        self.current_budgets   = {}
        self.last_acc_drop = 0.0; self.last_compression = 1.0; self.cycle_progress = 0.0

    def update_feedback(self, acc_drop, comp_ratio, cycle_prog):
        self.last_acc_drop = acc_drop; self.last_compression = comp_ratio
        self.cycle_progress = cycle_prog

    def _evaluate_model(self):
        self.model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for i, (inp, tgt) in enumerate(self.eval_dataloader):
                if i >= 10: break
                inp, tgt = inp.to(self.device), tgt.to(self.device)
                _, pred = self.model(inp).max(1)
                total += tgt.size(0); correct += pred.eq(tgt).sum().item()
        return 100. * correct / total

    def _decode_action(self, action):
        compression_levels = {0: 0.35, 1: 0.30, 2: 0.25, 3: 0.20, 4: 0.15}
        pruning_limits     = {k: min(v, self.global_goal.max_layer_pruning)
                              for k, v in {0:0.20,1:0.30,2:0.40,3:0.60,4:0.80}.items()}
        strategies = ['quantization', 'pruning', 'auto']
        budgets    = {}
        for block_idx in range(self.num_blocks):
            base_idx    = block_idx * 2
            comp_level  = action[base_idx]
            strategy_idx= action[base_idx + 1]
            sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
            if sensitivity > 0.7: comp_level = max(0, comp_level - 1)
            elif sensitivity < 0.3: comp_level = min(4, comp_level + 1)
            target_compression = compression_levels[comp_level]
            max_pruning        = pruning_limits[comp_level]
            preferred_strategy = strategies[strategy_idx]
            min_bits = (max(6, self.global_goal.min_layer_bits) if comp_level <= 1 else
                        max(5, self.global_goal.min_layer_bits) if comp_level <= 2 else
                        max(4, self.global_goal.min_layer_bits))
            max_acc_drop = 0.5 if sensitivity > 0.7 else (1.0 if sensitivity > 0.5 else 2.0)
            budgets[block_idx] = LayerBudget(
                block_idx=block_idx, target_compression_ratio=target_compression,
                max_accuracy_drop=max_acc_drop, priority=sensitivity,
                sensitivity=sensitivity, preferred_strategy=preferred_strategy,
                min_bits=min_bits, max_pruning=max_pruning,
                global_min_bits=self.global_goal.min_layer_bits,
                global_max_bits=self.global_goal.max_layer_bits,
                global_min_pruning=self.global_goal.min_layer_pruning,
                global_max_pruning=self.global_goal.max_layer_pruning)
        return budgets

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.baseline_accuracy is None:
            self.baseline_accuracy = self._evaluate_model()
        state    = np.zeros(5 + self.num_blocks * 2, dtype=np.float32)
        state[0] = self.last_acc_drop / 10.0; state[1] = self.last_compression
        state[2] = self.cycle_progress
        state[3] = self.global_goal.target_compression_ratio
        state[4] = self.global_goal.target_accuracy_drop / 10.0
        for i in range(self.num_blocks):
            state[5 + i*2]     = self.sensitivity_scores.get(i, 0.5)
            state[5 + i*2 + 1] = 0.5
        self.step_count = 0; self.current_budgets = {}
        return state, {}

    def step(self, action):
        budgets = self._decode_action(action); self.current_budgets = budgets
        avg_tc  = np.mean([b.target_compression_ratio for b in budgets.values()])
        comp_var= np.var( [b.target_compression_ratio for b in budgets.values()])
        balance_reward = -comp_var * 10
        sensitivity_aware_reward = 0.0
        for b_idx, budget in budgets.items():
            s = self.sensitivity_scores.get(b_idx, 0.5)
            if   s > 0.7 and budget.target_compression_ratio > 0.3:
                sensitivity_aware_reward += (budget.target_compression_ratio - 0.3) * 10
            elif s < 0.3 and budget.target_compression_ratio < 0.2:
                sensitivity_aware_reward += (0.2 - budget.target_compression_ratio) * 5
        global_gap        = abs(avg_tc - self.global_goal.target_compression_ratio)
        compression_reward= -global_gap * 50
        dynamic_reward    = 0.0
        if self.last_acc_drop > self.global_goal.target_accuracy_drop:
            if avg_tc < self.last_compression:
                dynamic_reward -= 50.0 * (self.last_acc_drop - self.global_goal.target_accuracy_drop)
        elif self.last_compression > self.global_goal.target_compression_ratio:
            if avg_tc < self.last_compression: dynamic_reward += 20.0
        reward = balance_reward + sensitivity_aware_reward + compression_reward + dynamic_reward
        state    = np.zeros(5 + self.num_blocks * 2, dtype=np.float32)
        state[0] = self.last_acc_drop / 10.0; state[1] = self.last_compression
        state[2] = self.cycle_progress; state[3] = global_gap
        state[4] = (balance_reward + sensitivity_aware_reward + dynamic_reward) / 20.0
        for i in range(self.num_blocks):
            state[5 + i*2] = self.sensitivity_scores.get(i, 0.5)
            if i in budgets: state[5 + i*2 + 1] = budgets[i].target_compression_ratio
        self.step_count += 1
        return state, reward, self.step_count >= 10, False, {
            'budgets': budgets, 'avg_compression': avg_tc}


# ─────────────────────────────────────────────────────────────────────────────
# HLA ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
class EnsembleHighLevelAgent:
    def __init__(self, model, eval_dataloader, sensitivity_scores,
                 global_goal, device='cuda', num_blocks=12,
                 num_hla_agents=3, weights_config=None):
        self.model = model; self.eval_dataloader = eval_dataloader
        self.sensitivity_scores = sensitivity_scores; self.global_goal = global_goal
        self.device = device; self.num_blocks = num_blocks
        self.num_hla_agents = num_hla_agents; self.hla_agents = []
        algorithms = ['PPO', 'A2C', 'PPO'][:num_hla_agents]
        self.budget_proposals = []
        for i in range(num_hla_agents):
            sg = copy.deepcopy(global_goal)
            if weights_config and i < len(weights_config):
                w = weights_config[i]; sg.alpha = w['alpha']; sg.beta = w['beta']; sg.gamma = w['gamma']
            env     = BudgetAllocationEnvironment(model, eval_dataloader,
                          sensitivity_scores, sg, device, num_blocks)
            vec_env = DummyVecEnv([lambda e=env: e])
            algo    = algorithms[i]
            if   algo == 'PPO': agent = PPO('MlpPolicy', vec_env, verbose=0, learning_rate=1e-4,
                                            n_steps=64, batch_size=32, ent_coef=0.01,
                                            seed=42+i, max_grad_norm=0.5, device='cpu')
            elif algo == 'A2C': agent = A2C('MlpPolicy', vec_env, verbose=0, learning_rate=3e-4,
                                            n_steps=64, ent_coef=0.01, seed=42+i,
                                            max_grad_norm=0.5, device='cpu')
            else:               agent = PPO('MlpPolicy', vec_env, verbose=0,
                                            seed=42+i, max_grad_norm=0.5, device='cpu')
            self.hla_agents.append({'agent': agent, 'env': vec_env,
                                    'algorithm': algo, 'weight': 1.0/num_hla_agents})

    def update_environments(self, acc_drop, comp_ratio, cycle, max_cycles):
        progress = cycle / float(max_cycles)
        for hd in self.hla_agents:
            for env in hd['env'].envs: env.update_feedback(acc_drop, comp_ratio, progress)

    def train_hla_agents(self, total_timesteps=2000):
        for hd in self.hla_agents: hd['agent'].learn(total_timesteps=total_timesteps)

    def allocate_budgets(self, deterministic=True):
        all_budgets = []
        for hd in self.hla_agents:
            obs    = hd['env'].reset()
            action, _ = hd['agent'].predict(obs, deterministic=deterministic)
            budgets= hd['env'].envs[0]._decode_action(action[0])
            all_budgets.append(budgets)
        final_budgets = {}
        for block_idx in range(self.num_blocks):
            props = [b[block_idx] for b in all_budgets if block_idx in b]
            if not props: continue
            strategies   = [b.preferred_strategy for b in props]
            most_common  = max(set(strategies), key=strategies.count)
            final_budgets[block_idx] = LayerBudget(
                block_idx=block_idx,
                target_compression_ratio=float(np.mean([b.target_compression_ratio for b in props])),
                max_accuracy_drop       =float(np.mean([b.max_accuracy_drop        for b in props])),
                priority                =float(np.mean([b.priority                 for b in props])),
                sensitivity             =self.sensitivity_scores.get(block_idx, 0.5),
                preferred_strategy      =most_common,
                min_bits                =int(np.round(np.mean([b.min_bits    for b in props]))),
                max_pruning             =float(np.mean([b.max_pruning for b in props])))
        return final_budgets


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────
class ComprehensiveVisualizer:
    @staticmethod
    def plot_pareto_front(cycle_history, save_path):
        if not cycle_history: return
        accuracies   = [c.get('accuracy', 0) for c in cycle_history]
        compressions = [1.0/c.get('compression_ratio', 1) for c in cycle_history]
        cycles = list(range(1, len(cycle_history)+1))
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(compressions, accuracies, c=cycles, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        for i, (x, y, c) in enumerate(zip(compressions, accuracies, cycles)):
            ax.annotate(f'C{c}', (x, y), fontsize=9, ha='center')
        ax.set_xlabel('Compression Ratio (x)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Pareto Front: Compression vs Accuracy', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax).set_label('Cycle', fontsize=11)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_compression_heatmap(configs, save_path):
        blocks = sorted(configs.keys())
        kernels = ['QKV', 'Attn Proj', 'MLP FC1', 'MLP FC2']
        bm_rows = []; pm_rows = []
        for b in blocks:
            c = configs[b]
            bm_rows.append([kc.weight_bits    if kc else 8 for kc in
                            [c.qkv_config, c.attn_proj_config, c.mlp_fc1_config, c.mlp_fc2_config]])
            pm_rows.append([(1-kc.pruning_ratio)*100 if kc else 0 for kc in
                            [c.qkv_config, c.attn_proj_config, c.mlp_fc1_config, c.mlp_fc2_config]])
        bm = np.array(bm_rows).T; pm = np.array(pm_rows).T
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        im1 = ax1.imshow(bm, cmap='viridis', aspect='auto')
        ax1.set_yticks(range(len(kernels))); ax1.set_yticklabels(kernels)
        ax1.set_xticks(range(len(blocks))); ax1.set_xticklabels([f'B{i}' for i in blocks])
        ax1.set_title('Learned Bitwidth per Kernel', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Bits')
        im2 = ax2.imshow(pm, cmap='Reds', aspect='auto')
        ax2.set_yticks(range(len(kernels))); ax2.set_yticklabels(kernels)
        ax2.set_xticks(range(len(blocks))); ax2.set_xticklabels([f'B{i}' for i in blocks])
        ax2.set_title('Learned Pruning Ratio per Kernel', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Pruning %')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_cycle_progression(cycle_history, save_path):
        if not cycle_history: return
        cycles       = [c.get('cycle', i) for i, c in enumerate(cycle_history)]
        accuracies   = [c.get('accuracy', 0) for c in cycle_history]
        compressions = [1.0/c.get('compression_ratio', 1) for c in cycle_history]
        acc_drops    = [c.get('accuracy_drop', 0) for c in cycle_history]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(cycles, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        if accuracies: axes[0].axhline(y=accuracies[0], color='red', linestyle='--', label='Baseline', alpha=0.7)
        axes[0].set_ylabel('Accuracy (%)', fontsize=11); axes[0].set_title('Accuracy Over Cycles', fontweight='bold')
        axes[0].grid(True, alpha=0.3); axes[0].legend()
        axes[1].plot(cycles, compressions, marker='s', linewidth=2, markersize=8, color='#A23B72')
        axes[1].set_ylabel('Compression Ratio (x)', fontsize=11)
        axes[1].set_title('Compression Ratio Over Cycles', fontweight='bold'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(cycles, acc_drops, marker='^', linewidth=2, markersize=8, color='#F18F01')
        axes[2].axhline(y=1.0, color='green', linestyle='--', label='Target (1%)', alpha=0.7)
        axes[2].set_xlabel('Cycle', fontsize=11); axes[2].set_ylabel('Accuracy Drop (%)', fontsize=11)
        axes[2].set_title('Accuracy Drop Over Cycles', fontweight='bold'); axes[2].grid(True, alpha=0.3); axes[2].legend()
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_budget_allocation(budgets, sensitivity_scores, save_path):
        blocks    = sorted(budgets.keys())
        tgt_comps = [budgets[b].target_compression_ratio for b in blocks]
        sensitivities = [sensitivity_scores.get(b, 0.5) for b in blocks]
        strategies    = [budgets[b].preferred_strategy for b in blocks]
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        x = np.arange(len(blocks)); width = 0.35
        axes[0].bar(x - width/2, [1.0/tc for tc in tgt_comps], width, label='Target Compression (x)', color='#2E86AB', alpha=0.8)
        axes[0].bar(x + width/2, [s*10  for s  in sensitivities], width, label='Sensitivity (x10)', color='#A23B72', alpha=0.8)
        axes[0].set_xlabel('Block Index', fontsize=12); axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('HLA Budget Allocation: Compression Targets vs Sensitivity', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x); axes[0].set_xticklabels([f'B{b}' for b in blocks])
        axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3, axis='y')
        strategy_colors = {'quantization': '#00A878', 'pruning': '#E63946', 'auto': '#457B9D'}
        strategy_counts = {'quantization': 0, 'pruning': 0, 'auto': 0}
        colors = []
        for s in strategies: strategy_counts[s] += 1; colors.append(strategy_colors[s])
        axes[1].bar(blocks, [1]*len(blocks), color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Block Index', fontsize=12); axes[1].set_ylabel('Strategy', fontsize=12)
        axes[1].set_title('HLA Strategy Selection per Block', fontsize=14, fontweight='bold')
        axes[1].set_yticks([])
        from matplotlib.patches import Patch
        axes[1].legend(handles=[
            Patch(facecolor=strategy_colors['quantization'], label=f"Quantization ({strategy_counts['quantization']})"),
            Patch(facecolor=strategy_colors['pruning'],      label=f"Pruning ({strategy_counts['pruning']})"),
            Patch(facecolor=strategy_colors['auto'],         label=f"Auto ({strategy_counts['auto']})"),
        ], fontsize=11, loc='upper right')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_solutions_comparison(cycle_history, save_path):
        if not cycle_history: return
        df = pd.DataFrame([{'Cycle': e.get('cycle',0), 'Accuracy (%)': e.get('accuracy',0),
                             'Compression (x)': 1.0/e.get('compression_ratio',1),
                             'Acc Drop (%)': e.get('accuracy_drop',0)} for e in cycle_history])
        fig, ax = plt.subplots(figsize=(12, 6))
        x = df['Cycle']; width = 0.25; x_pos = np.arange(len(x))
        ax.bar(x_pos - width, df['Accuracy (%)'],   width, label='Accuracy (%)',   color='#2E86AB', alpha=0.8)
        ax.bar(x_pos,         df['Compression (x)'],width, label='Compression (x)',color='#A23B72', alpha=0.8)
        ax.bar(x_pos + width, df['Acc Drop (%)'],   width, label='Acc Drop (%)',   color='#F18F01', alpha=0.8)
        ax.set_xlabel('Cycle', fontsize=12); ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Solution Comparison Across Cycles', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos); ax.set_xticklabels([f'C{c}' for c in x])
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_surrogate_training(surrogate_history, save_path):
        if not surrogate_history: return
        epochs = [h['epoch'] for h in surrogate_history]
        losses = [h['loss']  for h in surrogate_history]
        samples= [h['samples']for h in surrogate_history]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.plot(epochs, losses, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_xlabel('Epoch', fontsize=11); ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Surrogate Model Training Loss', fontweight='bold', fontsize=14); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, samples, marker='s', linewidth=2, markersize=6, color='#A23B72')
        ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Training Samples', fontsize=11)
        ax2.set_title('Surrogate Model Training Samples', fontweight='bold', fontsize=14); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_surrogate_predictions_vs_actual(predictions, save_path):
        if not predictions: return
        cycles    = [p['cycle']              for p in predictions]
        predicted = [p['predicted_accuracy'] for p in predictions]
        actual    = [p['actual_accuracy']    for p in predictions]
        errors    = [p['error']              for p in predictions]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(cycles, predicted, marker='o', linewidth=2, markersize=8, label='Predicted', color='#2E86AB', alpha=0.7)
        ax1.plot(cycles, actual,    marker='s', linewidth=2, markersize=8, label='Actual',    color='#A23B72', alpha=0.7)
        ax1.set_xlabel('Cycle', fontsize=11); ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('Surrogate Predictions vs Actual Accuracy', fontweight='bold', fontsize=14)
        ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)
        ax2.bar(cycles, errors, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cycle', fontsize=11); ax2.set_ylabel('Absolute Error (%)', fontsize=11)
        ax2.set_title('Surrogate Prediction Errors', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        mean_error = np.mean(errors)
        ax2.axhline(y=mean_error, color='red', linestyle='--',
                    label=f'Mean Error: {mean_error:.2f}%', linewidth=2)
        ax2.legend(fontsize=11)
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()

    @staticmethod
    def plot_compression_strategy_distribution(configs, save_path):
        quant_only = 0; prune_only = 0; both = 0
        for config in configs.values():
            for kc in [config.qkv_config, config.attn_proj_config,
                       config.mlp_fc1_config, config.mlp_fc2_config]:
                if kc is None: continue
                hq = kc.weight_bits < 8; hp = kc.pruning_ratio < 1.0
                if hq and hp: both += 1
                elif hq: quant_only += 1
                elif hp: prune_only += 1
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Quantization\nOnly', 'Pruning\nOnly', 'Both Quant\n& Pruning']
        values     = [quant_only, prune_only, both]
        colors     = ['#2E86AB', '#E63946', '#00A878']
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        total = sum(values)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val}\n({val/total*100:.1f}%)' if total else "0",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Kernels', fontsize=12)
        ax.set_title('Compression Strategy Distribution Across All Kernels', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class ViTCompressionTrainer:
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        self.config = config; self.logger = logger
        ReproducibilityManager.set_seed(42)
        if config.dataset == 'cifar10':
            self.train_loader, self.test_loader = DataManager.get_cifar10(config.batch_size)
        elif config.dataset == 'cifar100':
            self.train_loader, self.test_loader = DataManager.get_cifar100(config.batch_size)
        torch.cuda.empty_cache()
        self.model = timm.create_model(config.model_name, pretrained=True, num_classes=config.num_classes)
        self.model.to(config.device)
        self.original_model = copy.deepcopy(self.model).cpu()
        self.num_blocks     = len(self.model.blocks)
        self.criterion      = nn.CrossEntropyLoss()
        if config.use_surrogate:
            self.surrogate = SurrogateModelTrainer(
                num_blocks=self.num_blocks, num_kernels_per_block=4,
                hidden_dims=config.surrogate_hidden_dims,
                device=config.device, baseline_accuracy=85.0, logger=logger)
            self.logger.log("Surrogate model initialized", level='SUCCESS')
        else:
            self.surrogate = None
        self.layer_agents = {}; self.hla = None
        self.sensitivity_scores = {}; self.current_budgets = {}
        self.best_global_config = None; self.best_global_acc = 0.0
        self.baseline_accuracy  = 0.0; self.final_accuracy   = 0.0
        self.visualizer         = ComprehensiveVisualizer()

    def evaluate(self, model=None, max_batches=None):
        if model is None: model = self.model
        model.eval(); correct = 0; total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                if max_batches and batch_idx >= max_batches: break
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                _, predicted = model(inputs).max(1)
                total += targets.size(0); correct += predicted.eq(targets).sum().item()
        return 100. * correct / total

    def finetune_with_early_stopping(self, model, max_epochs=10, patience=3, lr=5e-5):
        self.logger.log(f"Fine-tuning (Max Epochs: {max_epochs})")
        masks = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name and param.dim() > 1:
                masks[name] = (param.data != 0).float().to(self.config.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        best_acc  = 0.0; patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        model.train()
        for epoch in range(max_epochs):
            running_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                optimizer.zero_grad()
                loss = self.criterion(model(inputs), targets)
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks: param.data *= masks[name]
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss/(pbar.n+1))
            scheduler.step()
            val_acc = self.evaluate(model, max_batches=20)
            self.logger.log(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc; best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.log(f"Early stopping at epoch {epoch+1}"); break
        model.load_state_dict(best_model_state)
        return best_acc

    def compute_sensitivity(self):
        self.logger.log("Computing layer sensitivity...")
        self.sensitivity_estimator = ViTSensitivityEstimator(
            self.model, self.train_loader, self.config.device, self.num_blocks)
        self.sensitivity_estimator.compute_fisher_information(num_samples=50)
        self.sensitivity_scores = self.sensitivity_estimator.get_layer_sensitivity_scores()

    # ─── NEW: Surrogate warm-up (ported from CNN trainer) ────────────────────
    def pretrain_surrogate(self):
        """Populate the surrogate with random compression samples before the RL loop."""
        if not self.surrogate:
            return
        self.logger.log("\n" + "=" * 80)
        self.logger.log("SURROGATE PRE-TRAINING PHASE")
        self.logger.log("=" * 80)
        self.logger.log(f"Generating {self.config.surrogate_warmup_samples} random samples...")

        kernel_names = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']

        for _ in tqdm(range(self.config.surrogate_warmup_samples),
                      desc="Pre-training Surrogate", leave=False):
            random_configs = {}
            for block_idx in range(self.num_blocks):
                layer_config = LayerConfig(block_idx=block_idx)
                for kname in kernel_names:
                    kc = KernelConfig(
                        name=kname,
                        weight_bits=int(np.random.randint(4, 9)),
                        act_bits=8,
                        quant_type=np.random.choice(QUANT_TYPES),
                        quant_mode=np.random.choice(QUANT_MODES),
                        pruning_ratio=float(np.random.uniform(0.6, 1.0)),
                        shape=(192, 192),   # placeholder – surrogate ignores shape
                    )
                    if   kname == 'qkv':       layer_config.qkv_config       = kc
                    elif kname == 'attn_proj': layer_config.attn_proj_config  = kc
                    elif kname == 'mlp_fc1':   layer_config.mlp_fc1_config    = kc
                    elif kname == 'mlp_fc2':   layer_config.mlp_fc2_config    = kc
                layer_config.update_aggregates()
                random_configs[block_idx] = layer_config

            # Apply to a fresh copy so self.model stays clean
            temp_model = copy.deepcopy(self.model)
            self._apply_configs_to_model_instance(temp_model, random_configs)

            acc = self.finetune_with_early_stopping(
                temp_model, max_epochs=1, patience=1, lr=1e-4)

            self.surrogate.add_sample(random_configs, acc)
            del temp_model
            torch.cuda.empty_cache()

        self.logger.log("\nTraining surrogate on collected samples...")
        self.surrogate.train(epochs=100, batch_size=16)
        self.logger.log(
            f"Surrogate pre-training complete! "
            f"Samples: {self.surrogate.get_buffer_size()}, "
            f"Loss: {self.surrogate.best_loss:.4f}\n" + "=" * 80,
            level='SUCCESS')

    def _apply_configs_to_model_instance(self, model: nn.Module,
                                          configs: Dict[int, LayerConfig]):
        """Apply compression configs to an arbitrary model (not necessarily self.model)."""
        for block_idx, config in configs.items():
            prefix = f'blocks.{block_idx}.'
            for name, module in model.named_modules():
                if not name.startswith(prefix) or not isinstance(module, nn.Linear):
                    continue
                if any(kw in name.lower() for kw in ['head', 'patch_embed']):
                    continue
                if   'qkv'  in name:                        kc = config.qkv_config
                elif 'attn' in name and 'proj' in name:     kc = config.attn_proj_config
                elif 'mlp'  in name and 'fc1'  in name:     kc = config.mlp_fc1_config
                elif 'mlp'  in name and 'fc2'  in name:     kc = config.mlp_fc2_config
                else:                                        continue
                if kc is None:
                    continue
                w = module.weight.data.clone()
                if kc.pruning_ratio < 1.0:
                    mask = AdvancedPruner.create_neuron_mask(w, kc.pruning_ratio)
                    w    = AdvancedPruner.apply_mask(w, mask)
                w = AdvancedQuantizer.quantize(
                    w, kc.weight_bits, mode=kc.quant_mode, quant_type=kc.quant_type)
                module.weight.data = w
    # ─────────────────────────────────────────────────────────────────────────

    def create_hla(self):
        self.logger.log("Creating High-Level Agent (Budget Allocator)...")
        self.hla = EnsembleHighLevelAgent(
            model=self.model, eval_dataloader=self.test_loader,
            sensitivity_scores=self.sensitivity_scores, global_goal=self.config.compression_goal,
            device=self.config.device, num_blocks=self.num_blocks,
            num_hla_agents=self.config.num_hla_agents, weights_config=self.config.hla_weights)

    def train_hla(self):
        self.logger.log("TRAINING HIGH-LEVEL AGENTS (Phase 1 pre-training)")
        self.hla.train_hla_agents(total_timesteps=self.config.hla_timesteps)
        self.logger.log("HLA training complete", level='SUCCESS')

    def get_budgets_from_hla(self, deterministic=True):
        self.logger.log(f"HLA allocating budgets (Deterministic={deterministic})")
        budgets = self.hla.allocate_budgets(deterministic=deterministic)
        return budgets

    def create_lla_with_budget(self, block_idx, budget):
        sensitivity = self.sensitivity_scores.get(block_idx, 0.5)
        env = BudgetConstrainedCompressionEnv(
            self.model, self.train_loader, self.test_loader,
            block_idx, sensitivity, self.config.compression_goal,
            self.config.device, curriculum_stage=2,
            layer_budget=budget, surrogate_model=self.surrogate,
            quantization_type=self.config.quantization_type,
            default_strategy =self.config.strategy)
        env.cached_inputs, _ = DataManager.get_cached_batches(
            self.train_loader, self.config.device, num_batches=2)
        env.baseline_accuracy = self.baseline_accuracy
        return OptimizedEnsembleLowLevelAgent(
            env, num_agents=self.config.num_lla_agents,
            algorithms=self.config.rl_algorithms, weights_config=self.config.lla_weights)

    def apply_configs_to_model(self, configs):
        for block_idx, config in configs.items():
            prefix = f'blocks.{block_idx}.'
            for name, module in self.model.named_modules():
                if not name.startswith(prefix) or not isinstance(module, nn.Linear): continue
                if any(kw in name.lower() for kw in ['head', 'patch_embed']): continue
                if   'qkv'  in name:                kc = config.qkv_config
                elif 'attn' in name and 'proj' in name: kc = config.attn_proj_config
                elif 'mlp'  in name and 'fc1' in name:  kc = config.mlp_fc1_config
                elif 'mlp'  in name and 'fc2' in name:  kc = config.mlp_fc2_config
                else: continue
                if kc is None: continue
                w = module.weight.data.clone()
                if kc.pruning_ratio < 1.0:
                    mask = AdvancedPruner.create_neuron_mask(w, kc.pruning_ratio)
                    w = AdvancedPruner.apply_mask(w, mask)
                w = AdvancedQuantizer.quantize(
                    w, kc.weight_bits, mode=kc.quant_mode, quant_type=kc.quant_type)
                module.weight.data = w

    def print_lla_voting_results(self, block_idx, lla, config):
        if not lla.voting_history: return
        self.logger.log(f"Block {block_idx}: LLA configuration selected by {config.selected_by_agent}")
        self.logger.log(f"  Average compression ratio: {1.0/config.compression_ratio():.2f}x")

    def run_hierarchical_compression(self):
        self.logger.log("="*80)
        self.logger.log("HIERARCHICAL COMPRESSION - FULL DYNAMIC LEARNING ENABLED")
        self.logger.log(f"  quantization_type : {self.config.quantization_type}")
        self.logger.log(f"  strategy          : {self.config.strategy}")
        self.logger.log("="*80)

        self.logger.log("\nPhase 0: Establishing Baseline")
        self.baseline_accuracy = self.finetune_with_early_stopping(
            self.model, max_epochs=3, patience=1, lr=1e-4)
        self.logger.log(f"Baseline accuracy: {self.baseline_accuracy:.2f}%")
        if self.surrogate: self.surrogate.update_baseline(self.baseline_accuracy)

        if self.surrogate and self.config.surrogate_warmup_samples > 0:
            self.pretrain_surrogate()

        self.compute_sensitivity()
        self.logger.log("\nPhase 2: Creating and training HLA")
        self.create_hla(); self.train_hla()

        current_acc_drop = 0.0; current_compression = 1.0
        num_cycles = 4

        for cycle in range(num_cycles):
            self.logger.log(f"\n{'='*80}")
            self.logger.log(f"CYCLE {cycle+1}/{num_cycles}")
            self.logger.log(f"{'='*80}")
            self.hla.update_environments(current_acc_drop, current_compression, cycle, num_cycles)
            if cycle > 0:
                self.logger.log("\nAdapting HLA policy based on previous results")
                self.hla.train_hla_agents(total_timesteps=512)
            is_deterministic       = (cycle >= num_cycles - 1)
            self.current_budgets   = self.get_budgets_from_hla(deterministic=is_deterministic)
            save_dir               = self.logger.get_run_dir()
            self.visualizer.plot_budget_allocation(
                self.current_budgets, self.sensitivity_scores,
                save_dir / f'budget_allocation_cycle_{cycle+1}.png')

            self.logger.log("Creating/Updating LLA agents with HLA budgets")
            for block_idx in range(self.num_blocks):
                budget = self.current_budgets.get(block_idx)
                if budget is None: continue
                if block_idx not in self.layer_agents:
                    self.layer_agents[block_idx] = self.create_lla_with_budget(block_idx, budget)
                else:
                    self.layer_agents[block_idx].update_budget(budget)

            self.logger.log("\nTraining LLA agents")
            for block_idx, lla in self.layer_agents.items():
                lla.train(total_timesteps=256)

            self.logger.log("\nExtracting configurations from LLA")
            configs = {}
            for block_idx, lla in self.layer_agents.items():
                config = lla.get_config()
                configs[block_idx] = config
                self.print_lla_voting_results(block_idx, lla, config)

            self.logger.log("\nApplying compression configurations and fine-tuning")
            self.apply_configs_to_model(configs)
            predicted_acc = None
            if self.surrogate and self.surrogate.get_buffer_size() >= 3:
                predicted_acc = self.surrogate.predict(configs)
            post_finetune_acc = self.finetune_with_early_stopping(
                self.model, max_epochs=self.config.finetune_epochs, patience=3, lr=5e-5)
            if predicted_acc is not None:
                self.logger.log_surrogate_prediction(cycle+1, predicted_acc, post_finetune_acc)
            if self.surrogate:
                self.surrogate.add_sample(configs, post_finetune_acc)
                if self.surrogate.get_buffer_size() % self.config.surrogate_update_freq == 0:
                    self.logger.log("Training surrogate model...")
                    self.surrogate.train(epochs=50, batch_size=16)
            current_acc_drop    = self.baseline_accuracy - post_finetune_acc
            avg_compression     = np.mean([c.compression_ratio() for c in configs.values()])
            current_compression = avg_compression
            compression_ratio_text = 1.0 / avg_compression if avg_compression > 0 else 1.0
            self.logger.log(f"\nCycle {cycle+1} Results:")
            self.logger.log(f"   Accuracy:    {post_finetune_acc:.2f}%")
            self.logger.log(f"   Acc Drop:    {current_acc_drop:.2f}%")
            self.logger.log(f"   Compression: {compression_ratio_text:.2f}x")
            self.logger.log_cycle_result(cycle+1, {
                'accuracy': post_finetune_acc, 'accuracy_drop': current_acc_drop,
                'compression_ratio': avg_compression,
                'configs': {k: v.to_dict() for k, v in configs.items()}})
            if post_finetune_acc > self.best_global_acc:
                self.best_global_acc    = post_finetune_acc
                self.best_global_config = copy.deepcopy(configs)
                self.logger.save_checkpoint(self.model, f'best_cycle_{cycle+1}')
            if post_finetune_acc > self.best_global_acc:
                self.best_global_acc    = post_finetune_acc
                self.best_global_config = copy.deepcopy(configs)
                self.logger.save_checkpoint(self.model, f'best_cycle_{cycle+1}')
            torch.cuda.empty_cache(); gc.collect()

        self.final_accuracy = self.best_global_acc
        self.logger.log("\n" + "="*80)
        self.logger.log("FINAL RESULTS")
        self.logger.log(f"Baseline: {self.baseline_accuracy:.2f}%")
        self.logger.log(f"Final:    {self.final_accuracy:.2f}%")
        self.logger.log(f"Drop:     {self.baseline_accuracy - self.final_accuracy:.2f}%")
        if self.best_global_config:
            compression = np.mean([c.compression_ratio() for c in self.best_global_config.values()])
            self.logger.log(f"Compression: {1.0/compression:.2f}x")
        self.generate_all_visualizations()
        return self.best_global_config

    def generate_all_visualizations(self):
        save_dir = self.logger.get_run_dir()
        cycle_history        = self.logger.metrics.get('cycle_history', [])
        surrogate_history    = self.logger.metrics.get('surrogate_history', [])
        surrogate_predictions= self.logger.metrics.get('surrogate_predictions', [])
        self.logger.log("\nGenerating Visualizations...")
        if self.best_global_config:
            self.visualizer.plot_compression_heatmap(self.best_global_config, save_dir/'compression_heatmap.png')
            self.visualizer.plot_compression_strategy_distribution(self.best_global_config, save_dir/'strategy_distribution.png')
        if cycle_history:
            self.visualizer.plot_pareto_front(cycle_history, save_dir/'pareto_front.png')
            self.visualizer.plot_cycle_progression(cycle_history, save_dir/'cycle_progression.png')
            self.visualizer.plot_solutions_comparison(cycle_history, save_dir/'solutions_comparison.png')
        if surrogate_history:
            self.visualizer.plot_surrogate_training(surrogate_history, save_dir/'surrogate_training.png')
        if surrogate_predictions:
            self.visualizer.plot_surrogate_predictions_vs_actual(surrogate_predictions, save_dir/'surrogate_predictions.png')
        self.logger.log("Visualization generation complete")


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def generate_comprehensive_report(configs, baseline_acc, final_acc, logger, surrogate=None):
    print("\n" + "="*80)
    print("COMPREHENSIVE FINAL REPORT")
    print("="*80)
    print(f"\nACCURACY METRICS")
    print(f"{'Baseline Accuracy:':<25} {baseline_acc:.2f}%")
    print(f"{'Final Accuracy:':<25} {final_acc:.2f}%")
    print(f"{'Accuracy Drop:':<25} {baseline_acc - final_acc:.2f}%")
    print(f"{'Relative Drop:':<25} {100*(baseline_acc - final_acc)/baseline_acc:.2f}%")
    both_strategies = 0; quant_only = 0; prune_only = 0; neither = 0
    total_budget_compliance = 0; budget_compliant_count = 0
    strategy_by_layer = []; compression_by_layer = []
    type_counts = {'INT': 0, 'FLOAT': 0}
    gran_counts  = {'uniform': 0, 'log': 0, 'per-channel': 0, 'learned': 0}
    for block_idx in sorted(configs.keys()):
        config = configs[block_idx]; budget = config.assigned_budget
        for name, kc in [('qkv', config.qkv_config), ('attn_proj', config.attn_proj_config),
                          ('mlp_fc1', config.mlp_fc1_config), ('mlp_fc2', config.mlp_fc2_config)]:
            if kc is None: continue
            hq = kc.weight_bits < 8; hp = kc.pruning_ratio < 1.0
            if hq and hp: both_strategies += 1
            elif hq: quant_only += 1
            elif hp: prune_only += 1
            else: neither += 1
            type_counts[kc.quant_type] = type_counts.get(kc.quant_type, 0) + 1
            gran_counts[kc.quant_mode] = gran_counts.get(kc.quant_mode,  0) + 1
        if budget:
            actual = config.compression_ratio(); target = budget.target_compression_ratio
            total_budget_compliance += 1.0 - abs(actual - target) / target
            if abs(actual - target) < 0.05: budget_compliant_count += 1
            strategy_by_layer.append(budget.preferred_strategy)
            compression_by_layer.append(1.0 / actual)
    avg_budget_compliance = total_budget_compliance / len(configs) if configs else 0
    total_kernels = both_strategies + quant_only + prune_only + neither
    print(f"\nCOMPRESSION STRATEGY DISTRIBUTION")
    for label, val in [("Both Quant + Prune", both_strategies), ("Quantization Only", quant_only),
                        ("Pruning Only", prune_only), ("Neither", neither)]:
        pct = 100*val/total_kernels if total_kernels else 0
        print(f"  {label:<22} {val}  ({pct:.1f}%)")
    print(f"  {'Total Kernels':<22} {total_kernels}")
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
        strategy_counts = Counter(strategy_by_layer)
        print(f"\nHLA STRATEGY SELECTION")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy.capitalize():<15} {count} layers ({100*count/len(strategy_by_layer):.1f}%)")
    compression = np.mean([c.compression_ratio() for c in configs.values()])
    print(f"\nCOMPRESSION METRICS")
    print(f"  {'Compression Ratio:':<25} {1.0/compression:.2f}x")
    print(f"  {'Model Size Reduction:':<25} {(1.0-compression)*100:.1f}%")
    print(f"  {'Budget Compliance:':<25} {avg_budget_compliance*100:.1f}%")
    print(f"  {'Budget Compliant Layers:':<25} {budget_compliant_count}/{len(configs)}")
    if compression_by_layer:
        print(f"  {'Min Compression:':<25} {min(compression_by_layer):.2f}x")
        print(f"  {'Max Compression:':<25} {max(compression_by_layer):.2f}x")
        print(f"  {'Std Compression:':<25} {np.std(compression_by_layer):.2f}")
    if surrogate:
        stats = surrogate.get_training_stats()
        print(f"\n SURROGATE MODEL PERFORMANCE")
        for label, val in [("Training Samples", stats['total_samples']),
                           ("Total Epochs", stats['total_epochs']),
                           ("Training Sessions", stats['training_sessions'])]:
            print(f"  {label:<25} {val}")
        print(f"  {'Total Training Time:':<25} {stats['total_time']:.2f}s")
        print(f"  {'Avg Time/Epoch:':<25} {stats['avg_time_per_epoch']:.3f}s")
        print(f"  {'Final Loss (MSE):':<25} {stats['best_loss']:.6f}")
        if logger.metrics.get('surrogate_predictions'):
            predictions = logger.metrics['surrogate_predictions']
            errors = [p['error'] for p in predictions]
            print(f"\nSURROGATE PREDICTION ACCURACY")
            print(f"  {'Mean Absolute Error:':<25} {np.mean(errors):.2f}%")
            print(f"  {'Std Deviation:':<25} {np.std(errors):.2f}%")
            print(f"  {'Min / Max Error:':<25} {min(errors):.2f}% / {max(errors):.2f}%")
            print(f"  {'Predictions Made:':<25} {len(predictions)}")
    print(f"\nOUTPUT DIRECTORY\n  {logger.get_run_dir()}\n" + "="*80)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    compression_goal = CompressionGoal(
        target_accuracy_drop=1.0, target_compression_ratio=0.25,
        min_layer_bits=3, max_layer_bits=8,
        min_layer_pruning=0.0, max_layer_pruning=0.6,
        alpha=50.0, beta=2.0, gamma=1.0)

    config = ExperimentConfig(
        model_name="deit_base_patch16_224",
        dataset="cifar100",
        num_classes=100,
        do_finetune=True,
        finetune_epochs=15,
        batch_size=64,
        num_lla_agents=3,
        num_hla_agents=3,
        lla_timesteps=256,
        hla_timesteps=256,
        use_surrogate=True,
        surrogate_warmup_samples=50,  
        hla_budget_update_freq=1,
        compression_goal=compression_goal,
        experiment_name='hrl_vit_configurable_base',
        quantization_type='mixed',
        strategy=None,
    )



    logger  = ExperimentLogger(config.experiment_name, config.output_dir)
    logger.log_config(asdict(config))
    trainer = ViTCompressionTrainer(config, logger)
    configs = trainer.run_hierarchical_compression()

    logger.log_metric('baseline_accuracy', trainer.baseline_accuracy)
    logger.log_metric('final_accuracy',    trainer.final_accuracy)
    if trainer.surrogate:
        stats = trainer.surrogate.get_training_stats()
        logger.log_metric('surrogate_samples', stats['total_samples'])
        logger.log_metric('surrogate_epochs',  stats['total_epochs'])
        logger.log_metric('surrogate_loss',    stats['best_loss'])
        logger.log_metric('surrogate_time',    stats['total_time'])
    logger.save_metrics()
    generate_comprehensive_report(
        configs, trainer.baseline_accuracy, trainer.final_accuracy,
        logger, trainer.surrogate)
    run_dir = logger.get_run_dir()
    print_comprehensive_summary(trainer, configs)
    plot_per_kernel_decisions(configs, save_path=run_dir / "agent_decisions.png")
    plot_acc_vs_reduction(trainer, configs, save_path=run_dir / "acc_vs_reduction.png")
    plot_sensitivity_analysis(trainer, configs)



if __name__ == "__main__":
    main()
