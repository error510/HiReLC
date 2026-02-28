"""Comprehensive visualization utilities"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path


class ComprehensiveVisualizer:
    """Visualization utilities for compression analysis."""
    
    @staticmethod
    def plot_per_kernel_decisions(configs: Dict, kernel_names: list,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot per-kernel compression decisions as heatmaps.
        
        Args:
            configs: Dict[block_idx -> LayerConfig]
            kernel_names: List of kernel names (e.g., ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2'])
            save_path: Optional path to save figure
        """
        num_blocks = len(configs)
        num_kernels = len(kernel_names)
        
        bits_mat = np.zeros((num_kernels, num_blocks))
        prune_mat = np.zeros((num_kernels, num_blocks))
        type_mat = np.zeros((num_kernels, num_blocks))
        gran_mat = np.zeros((num_kernels, num_blocks))
        
        for block_idx in range(num_blocks):
            if block_idx not in configs:
                continue
            
            config = configs[block_idx]
            if not hasattr(config, 'kernels'):
                continue
            
            for k_i, k_name in enumerate(kernel_names):
                if k_name in config.kernels:
                    k = config.kernels[k_name]
                    bits_mat[k_i, block_idx] = k.weight_bits
                    prune_mat[k_i, block_idx] = (1.0 - k.pruning_ratio) * 100
                    type_mat[k_i, block_idx] = 1 if k.quant_type == 'FLOAT' else 0
                    
                    gran_map = {'uniform': 0, 'log': 1, 'per-channel': 2, 'learned': 3}
                    gran_mat[k_i, block_idx] = gran_map.get(k.quant_mode, 0)
        
        x_labels = [f"B{i}" for i in range(num_blocks)]
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        # Bits heatmap
        sns.heatmap(bits_mat, annot=True, fmt='g', cmap="YlGnBu",
                   ax=axes[0], xticklabels=x_labels, yticklabels=kernel_names,
                   cbar_kws={'label': 'Bits'})
        axes[0].set_title("Quantization: Weight Bits per Kernel", fontsize=12, fontweight='bold')
        
        # Pruning heatmap
        sns.heatmap(prune_mat, annot=True, fmt='.0f', cmap="Reds",
                   ax=axes[1], xticklabels=x_labels, yticklabels=kernel_names,
                   cbar_kws={'label': '% Pruned'})
        axes[1].set_title("Sparsity: Percentage of Neurons Pruned", fontsize=12, fontweight='bold')
        
        # Type heatmap
        cmap_type = sns.color_palette(["#3498db", "#e67e22"], as_cmap=True)
        sns.heatmap(type_mat, annot=np.where(type_mat == 1, "FP", "INT"),
                   fmt="", cmap=cmap_type, ax=axes[2], cbar=False,
                   xticklabels=x_labels, yticklabels=kernel_names)
        axes[2].set_title("Data Type: INT vs FLOAT Decision", fontsize=12, fontweight='bold')
        
        # Granularity heatmap
        gran_str_map = {0: 'UNI', 1: 'LOG', 2: 'PCH', 3: 'LRN'}
        gran_annot = np.vectorize(lambda x: gran_str_map.get(int(x), '?'))(gran_mat)
        cmap_gran = sns.color_palette(["#27ae60", "#8e44ad", "#e74c3c", "#f39c12"],
                                     as_cmap=True)
        sns.heatmap(gran_mat, annot=gran_annot, fmt="", cmap=cmap_gran, ax=axes[3],
                   cbar=False, xticklabels=x_labels, yticklabels=kernel_names, vmin=0, vmax=3)
        axes[3].set_title(
            "Granularity: UNI=uniform / LOG=log / PCH=per-channel / LRN=learned",
            fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_accuracy_vs_size(baseline_acc: float, final_acc: float,
                             baseline_size_mb: float, compressed_size_mb: float,
                             save_path: Optional[str] = None) -> None:
        """
        Plot Pareto frontier: accuracy vs model size.
        
        Args:
            baseline_acc: Baseline model accuracy
            final_acc: Compressed model accuracy
            baseline_size_mb: Baseline model size in MB
            compressed_size_mb: Compressed model size in MB
            save_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter([baseline_size_mb], [baseline_acc], color='gray', s=100,
                  label='Baseline', marker='o', alpha=0.7)
        ax.scatter([compressed_size_mb], [final_acc], color='green', s=200,
                  label='HRL Result', marker='*', alpha=0.9)
        
        # Add annotation
        reduction = (1 - compressed_size_mb / baseline_size_mb) * 100
        acc_drop = baseline_acc - final_acc
        txt = f"Size: -{reduction:.1f}%\nAcc: {acc_drop:+.2f}%"
        ax.annotate(txt, (compressed_size_mb, final_acc),
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title("Pareto Frontier: Accuracy vs Model Size", fontsize=12, fontweight='bold')
        ax.set_xlabel("Model Size (MB)", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim(0, baseline_size_mb * 1.2)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_sensitivity_analysis(sensitivity_scores: Dict[int, float],
                                  compression_ratios: Dict[int, float],
                                  save_path: Optional[str] = None) -> None:
        """
        Plot sensitivity vs compression analysis.
        
        Args:
            sensitivity_scores: Dict[block_idx -> sensitivity_score]
            compression_ratios: Dict[block_idx -> compression_ratio]
            save_path: Optional save path
        """
        blocks = sorted(sensitivity_scores.keys())
        sens_scores = [sensitivity_scores[b] for b in blocks]
        comp_ratios = [compression_ratios.get(b, 1.0) for b in blocks]
        
        if len(blocks) == 0:
            print("No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(sens_scores, comp_ratios, c=blocks, cmap='viridis',
                           s=150, alpha=0.6, edgecolors='black')
        cbar = plt.colorbar(scatter, ax=ax, label='Block Index')
        
        # Add trendline if enough points
        if len(sens_scores) > 1:
            z = np.polyfit(sens_scores, comp_ratios, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(sens_scores), max(sens_scores), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend')
        
        ax.set_xlabel("Fisher Sensitivity Score (Higher = More Important)", fontsize=11)
        ax.set_ylabel("Compression Ratio (Lower = More Compressed)", fontsize=11)
        ax.set_title("Layer Sensitivity vs Compression Analysis", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_learning_curves(train_losses: list, val_accuracies: list,
                            save_path: Optional[str] = None) -> None:
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(val_accuracies)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Validation Accuracy")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_surrogate_predictions(predictions: list, actuals: list,
                                  save_path: Optional[str] = None) -> None:
        """Plot surrogate model predictions vs actual values."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(actuals, predictions, alpha=0.6, s=100, edgecolors='black')
        
        # Perfect prediction line
        min_val = min(min(predictions), min(actuals))
        max_val = max(max(predictions), max(actuals))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel("Actual Accuracy (%)", fontsize=11)
        ax.set_ylabel("Predicted Accuracy (%)", fontsize=11)
        ax.set_title("Surrogate Model Performance", fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
