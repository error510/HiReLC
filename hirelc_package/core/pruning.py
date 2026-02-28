"""Pruning module with multiple strategies"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import numpy as np


class AdvancedPruner:
    """
    Advanced pruning strategies including magnitude-based, movement,
    lottery ticket, and gradient-based pruning.
    """
    
    @staticmethod
    def prune_magnitude(tensor: torch.Tensor, pruning_ratio: float) -> torch.Tensor:
        """
        Magnitude-based pruning: remove smallest weights.
        
        Args:
            tensor: Weight tensor to prune
            pruning_ratio: Fraction of weights to zero out (0.0 - 1.0)
        
        Returns:
            Pruned tensor with smallest weights set to zero
        """
        if pruning_ratio <= 0:
            return tensor
        
        k = max(1, int(tensor.numel() * pruning_ratio))
        _, indices = torch.topk(torch.abs(tensor).flatten(), k, largest=False)
        
        pruned = tensor.clone()
        pruned.flatten()[indices] = 0
        return pruned
    
    @staticmethod
    def prune_movement(tensor: torch.Tensor, pruning_ratio: float,
                      initial_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Movement-based pruning: prune weights that move least during training.
        
        Args:
            tensor: Current weight tensor
            pruning_ratio: Fraction of weights to prune
            initial_tensor: Initial weights (before training)
        
        Returns:
            Pruned tensor
        """
        if pruning_ratio <= 0 or initial_tensor is None:
            return tensor
        
        movement = torch.abs(tensor - initial_tensor)
        k = max(1, int(tensor.numel() * pruning_ratio))
        _, indices = torch.topk(movement.flatten(), k, largest=False)
        
        pruned = tensor.clone()
        pruned.flatten()[indices] = 0
        return pruned
    
    @staticmethod
    def prune_gradient_based(tensor: torch.Tensor, gradients: torch.Tensor,
                           pruning_ratio: float) -> torch.Tensor:
        """
        Gradient-based pruning: prune weights with low gradient magnitude.
        
        Args:
            tensor: Weight tensor
            gradients: Gradient tensor (same shape as tensor)
            pruning_ratio: Fraction to prune
        
        Returns:
            Pruned tensor
        """
        if pruning_ratio <= 0 or gradients is None:
            return tensor
        
        # Score = weight magnitude * gradient magnitude
        scores = torch.abs(tensor) * torch.abs(gradients)
        k = max(1, int(tensor.numel() * pruning_ratio))
        _, indices = torch.topk(scores.flatten(), k, largest=False)
        
        pruned = tensor.clone()
        pruned.flatten()[indices] = 0
        return pruned
    
    @staticmethod
    def prune_fisher_pruning(tensor: torch.Tensor, fisher_scores: torch.Tensor,
                            pruning_ratio: float) -> torch.Tensor:
        """
        Fisher pruning: prune based on Fisher information scores.
        
        Args:
            tensor: Weight tensor
            fisher_scores: Fisher scores (sensitivity) for each parameter
            pruning_ratio: Fraction to prune
        
        Returns:
            Pruned tensor
        """
        if pruning_ratio <= 0:
            return tensor
        
        k = max(1, int(tensor.numel() * pruning_ratio))
        _, indices = torch.topk(fisher_scores.flatten(), k, largest=False)
        
        pruned = tensor.clone()
        pruned.flatten()[indices] = 0
        return pruned
    
    @staticmethod
    def prune_layer(layer: nn.Module, pruning_ratio: float,
                   method: str = 'magnitude') -> None:
        """
        Apply pruning to a layer in-place.
        
        Args:
            layer: PyTorch layer (Conv2d, Linear, etc.)
            pruning_ratio: Fraction to prune
            method: 'magnitude', 'movement', 'gradient'
        """
        if not hasattr(layer, 'weight') or layer.weight is None:
            return
        
        if method == 'magnitude':
            layer.weight.data = AdvancedPruner.prune_magnitude(
                layer.weight.data, pruning_ratio
            )
        elif method == 'movement':
            if hasattr(layer, 'weight_init'):
                layer.weight.data = AdvancedPruner.prune_movement(
                    layer.weight.data, pruning_ratio, layer.weight_init
                )
        # Add more methods as needed


class FineGrainedPruner:
    """
    Fine-grained pruning at structured (channel, filter) and
    unstructured (weight) levels.
    """
    
    @staticmethod
    def channel_pruning(tensor: torch.Tensor, pruning_ratio: float,
                       axis: int = 0) -> torch.Tensor:
        """
        Channel-wise pruning: remove entire channels.
        
        Args:
            tensor: Weight tensor (typically [out_channels, in_channels, H, W])
            pruning_ratio: Fraction of channels to remove
            axis: Axis along which to prune (0 = out_channels)
        
        Returns:
            Pruned tensor with some channels zeroed
        """
        if tensor.dim() < 2 or pruning_ratio <= 0:
            return tensor
        
        # Compute channel-wise importance (L2 norm)
        tensor_moved = torch.moveaxis(tensor, axis, 0)
        shape = tensor_moved.shape
        tensor_flat = tensor_moved.reshape(shape[0], -1)
        
        importance = torch.norm(tensor_flat, p=2, dim=1)
        
        k = max(1, int(len(importance) * pruning_ratio))
        _, indices = torch.topk(importance, k, largest=False)
        
        pruned = tensor.clone()
        pruned_moved = torch.moveaxis(pruned, axis, 0)
        pruned_moved[indices] = 0
        pruned = torch.moveaxis(pruned_moved, 0, axis)
        
        return pruned
    
    @staticmethod
    def filter_pruning(conv_layer: nn.Conv2d, pruning_ratio: float) -> None:
        """
        Filter pruning: remove entire filters (output channels) from Conv2d.
        Applied in-place.
        """
        weight = conv_layer.weight.data
        k = max(1, int(weight.shape[0] * pruning_ratio))
        
        importance = torch.norm(weight.view(weight.shape[0], -1), dim=1)
        _, indices = torch.topk(importance, k, largest=False)
        
        weight[indices] = 0
    
    @staticmethod
    def structured_pruning(layer: nn.Module, pruning_ratio: float) -> None:
        """
        Structured pruning (removes entire filters/channels).
        More hardware-friendly than unstructured pruning.
        """
        if isinstance(layer, nn.Conv2d):
            FineGrainedPruner.filter_pruning(layer, pruning_ratio)
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.data
            k = max(1, int(weight.shape[0] * pruning_ratio))
            importance = torch.norm(weight, dim=1) if weight.dim() > 1 else torch.abs(weight)
            _, indices = torch.topk(importance, k, largest=False)
            weight[indices] = 0


class AdaptivePruner:
    """
    Adaptive pruning that adjusts pruning ratios based on layer
    importance and compression targets.
    """
    
    @staticmethod
    def allocate_pruning_ratios(layer_sensitivities: Dict[int, float],
                               target_sparsity: float,
                               min_prune: float = 0.0,
                               max_prune: float = 0.8) -> Dict[int, float]:
        """
        Allocate pruning ratios to layers based on sensitivity.
        More sensitive layers get pruned less.
        
        Args:
            layer_sensitivities: Dict[layer_idx -> sensitivity_score]
            target_sparsity: Target overall sparsity (0 - 1)
            min_prune: Min pruning ratio per layer
            max_prune: Max pruning ratio per layer
        
        Returns:
            Dict[layer_idx -> pruning_ratio]
        """
        num_layers = len(layer_sensitivities)
        
        # Normalize sensitivities
        max_sens = max(layer_sensitivities.values()) if layer_sensitivities else 1.0
        norm_sens = {k: v / max_sens for k, v in layer_sensitivities.items()}
        
        # Allocate pruning ratios: less sensitive = more pruning
        allocation = {}
        for layer_idx, sensitivity in norm_sens.items():
            # Inverse relationship: lower sensitivity = higher pruning
            prune_ratio = target_sparsity * (1 - sensitivity)
            prune_ratio = max(min_prune, min(max_prune, prune_ratio))
            allocation[layer_idx] = prune_ratio
        
        return allocation
    
    @staticmethod
    def iterative_pruning(model: nn.Module,
                         layer_sensitivities: Dict[int, float],
                         target_sparsity: float,
                         num_iterations: int = 5) -> None:
        """
        Iteratively prune and fine-tune to gradually reach target sparsity.
        
        Args:
            model: Neural network model
            layer_sensitivities: Layer importance scores
            target_sparsity: Target sparsity ratio
            num_iterations: Number of prune-finetune cycles
        """
        current_sparsity = 0.0
        step = target_sparsity / num_iterations
        
        for iteration in range(num_iterations):
            current_sparsity = (iteration + 1) * step
            prune_ratios = AdaptivePruner.allocate_pruning_ratios(
                layer_sensitivities, current_sparsity
            )
            
            # Apply pruning based on allocated ratios
            # This would integrate with the model trainer to do pruning + finetuning
            # print(f"Iteration {iteration+1}: Current sparsity = {current_sparsity:.2%}")
