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
    def compute_importance_scores(weight: torch.Tensor,
                                  method: str = 'l2',
                                  gradient: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute neuron/channel importance scores.

        For Conv2d/Linear weights, importance is computed per output channel.
        """
        if weight.dim() < 2:
            w_flat = weight.view(-1, 1)
        else:
            w_flat = weight.view(weight.shape[0], -1)

        method = (method or 'l2').lower()
        if method == 'l1':
            return torch.norm(w_flat, p=1, dim=1)
        if method == 'fisher' and gradient is not None:
            g_flat = gradient.view(weight.shape[0], -1)
            fisher = (w_flat * g_flat) ** 2
            return fisher.sum(dim=1)
        return torch.norm(w_flat, p=2, dim=1)

    @staticmethod
    def create_neuron_mask(weight: torch.Tensor,
                           pruning_ratio: float,
                           importance_method: str = 'l2',
                           gradient: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Create a binary mask for neurons/channels to keep.

        Args:
            weight: Weight tensor
            pruning_ratio: Interpreted as keep ratio (0-1)
            importance_method: 'l2', 'l1', or 'fisher'
            gradient: Optional gradients for fisher importance

        Returns:
            Mask tensor with same shape as weight (values 0 or 1)
        """
        keep_ratio = float(pruning_ratio)
        keep_ratio = max(0.0, min(1.0, keep_ratio))

        if weight.dim() == 0:
            return torch.ones_like(weight)

        out_features = weight.shape[0]
        if keep_ratio <= 0.0:
            channel_mask = torch.zeros(out_features, device=weight.device, dtype=weight.dtype)
        else:
            n_keep = max(1, int(out_features * keep_ratio))
            importance = AdvancedPruner.compute_importance_scores(
                weight, method=importance_method, gradient=gradient
            )
            _, indices = torch.topk(importance, n_keep, largest=True)
            channel_mask = torch.zeros(out_features, device=weight.device, dtype=weight.dtype)
            channel_mask[indices] = 1.0

        # Expand to full weight shape
        if weight.dim() == 1:
            mask = channel_mask
        elif weight.dim() == 2:
            mask = channel_mask.view(-1, 1).expand_as(weight)
        elif weight.dim() == 4:
            mask = channel_mask.view(-1, 1, 1, 1).expand_as(weight)
        else:
            view_shape = [out_features] + [1] * (weight.dim() - 1)
            mask = channel_mask.view(*view_shape).expand_as(weight)

        return mask

    @staticmethod
    def apply_mask(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply a binary mask to a tensor."""
        if mask.shape == tensor.shape:
            return tensor * mask

        if tensor.dim() == 4:
            return tensor * mask.view(-1, 1, 1, 1)
        if tensor.dim() == 2:
            return tensor * mask.unsqueeze(1)
        if tensor.dim() == 1:
            return tensor * mask
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    
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
