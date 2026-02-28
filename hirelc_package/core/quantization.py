"""Core quantization module"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import numpy as np


@torch.jit.script
def _quantize_uniform_jit(tensor: torch.Tensor, bits: int, symmetric: bool) -> torch.Tensor:
    """JIT-compiled uniform quantization for speed."""
    if bits == 32:
        return tensor
    
    if symmetric:
        max_val = torch.max(torch.abs(tensor))
        scale = max_val / (2 ** (bits - 1) - 1)
    else:
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        scale = (max_val - min_val) / (2 ** bits - 1)
    
    if scale == 0:
        return tensor
    
    q_tensor = torch.round(tensor / scale) * scale
    return q_tensor


class AdvancedQuantizer:
    """
    Advanced quantization methods including uniform, logarithmic,
    per-channel, and learned quantization.
    """
    
    @staticmethod
    def quantize_uniform(tensor: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
        """Uniform quantization."""
        return _quantize_uniform_jit(tensor, bits, symmetric)
    
    @staticmethod
    def quantize_logarithmic(tensor: torch.Tensor, bits: int, base: float = 2.0) -> torch.Tensor:
        """Logarithmic quantization (better for skewed distributions)."""
        if bits == 32:
            return tensor
        
        sign = torch.sign(tensor)
        abs_tensor = torch.abs(tensor)
        
        # Avoid log(0)
        abs_tensor = torch.clamp(abs_tensor, min=1e-8)
        log_tensor = torch.log(abs_tensor + 1e-8) / np.log(base)
        
        max_val = torch.max(log_tensor)
        scale = max_val / (2 ** (bits - 1) - 1)
        
        if scale == 0:
            return tensor
        
        q_log = torch.round(log_tensor / scale) * scale
        q_tensor = sign * torch.pow(base, q_log)
        
        return q_tensor
    
    @staticmethod
    def quantize_per_channel(tensor: torch.Tensor, bits: int, axis: int = 0) -> torch.Tensor:
        """Per-channel quantization (each channel has its own scale)."""
        if bits == 32:
            return tensor
        
        if tensor.dim() < 2:
            return AdvancedQuantizer.quantize_uniform(tensor, bits)
        
        # Move axis to first position
        tensor_reshaped = torch.movedim(tensor, axis, 0)
        shape = tensor_reshaped.shape
        tensor_flat = tensor_reshaped.reshape(shape[0], -1)
        
        q_tensor = torch.zeros_like(tensor_flat)
        
        for i in range(shape[0]):
            channel = tensor_flat[i]
            max_val = torch.max(torch.abs(channel))
            scale = max_val / (2 ** (bits - 1) - 1) if max_val > 0 else 1.0
            q_tensor[i] = torch.round(channel / scale) * scale
        
        # Reshape back
        q_tensor = q_tensor.reshape(shape)
        q_tensor = torch.movedim(q_tensor, 0, axis)
        
        return q_tensor
    
    @staticmethod
    def quantize_learned(tensor: torch.Tensor, bits: int, 
                        scale: Optional[torch.Tensor] = None,
                        zero_point: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Learned quantization with trainable scale and zero-point.
        
        Returns:
            quantized_tensor: The quantized tensor
            scale: The scale factor
            zero_point: The zero point
        """
        if bits == 32:
            return tensor, torch.tensor(1.0), torch.tensor(0.0)
        
        if scale is None:
            max_val = torch.max(torch.abs(tensor))
            scale = max_val / (2 ** (bits - 1) - 1) if max_val > 0 else 1.0
        
        if zero_point is None:
            zero_point = torch.tensor(0.0, device=tensor.device)
        
        q_levels = 2 ** bits - 1
        q_tensor = torch.clamp(
            torch.round((tensor - zero_point) / scale),
            -q_levels // 2,
            q_levels // 2
        ) * scale + zero_point
        
        return q_tensor, scale, zero_point


class DynamicQuantizer:
    """
    Dynamic/adaptive quantization that adjusts bitwidth based on
    activation statistics to optimize accuracy-efficiency tradeoff.
    """
    
    @staticmethod
    def select_bitwidth(tensor: torch.Tensor, 
                       candidates: list = [2, 4, 8, 16, 32],
                       metric: str = 'mse') -> int:
        """
        Select optimal bitwidth for tensor based on reconstruction error.
        
        Args:
            tensor: Input tensor to quantize
            candidates: List of bitwidth candidates to evaluate
            metric: 'mse', 'mae', 'kl_divergence'
        
        Returns:
            Optimal bitwidth
        """
        if metric == 'mse':
            errors = []
            for bits in candidates:
                q_tensor = AdvancedQuantizer.quantize_uniform(tensor, bits)
                error = torch.mean((tensor - q_tensor) ** 2).item()
                errors.append(error)
        
        elif metric == 'mae':
            errors = []
            for bits in candidates:
                q_tensor = AdvancedQuantizer.quantize_uniform(tensor, bits)
                error = torch.mean(torch.abs(tensor - q_tensor)).item()
                errors.append(error)
        
        elif metric == 'kl_divergence':
            # KL divergence between histograms
            errors = []
            for bits in candidates:
                q_tensor = AdvancedQuantizer.quantize_uniform(tensor, bits)
                # Implement KL divergence between distributions
                errors.append(0.0)  # Placeholder
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Return bitwidth with minimum error
        optimal_bits = candidates[np.argmin(errors)]
        return optimal_bits


class MixedPrecisionQuantizer:
    """
    Mixed-precision quantization: assigns different bitwidths to
    different layers based on sensitivity analysis.
    """
    
    @staticmethod
    def allocate_bits(layer_sensitivities: Dict[int, float],
                     total_budget: int,
                     min_bits: int = 2,
                     max_bits: int = 8) -> Dict[int, int]:
        """
        Allocate bitwidths to layers based on sensitivity scores.
        
        Args:
            layer_sensitivities: Dict mapping layer_idx -> sensitivity score
            total_budget: Total bit budget across all layers
            min_bits: Minimum bitwidth per layer
            max_bits: Maximum bitwidth per layer
        
        Returns:
            Dict mapping layer_idx -> allocated_bits
        """
        num_layers = len(layer_sensitivities)
        # Start with average
        avg_bits = total_budget / num_layers
        
        # Allocate more bits to sensitive layers
        allocation = {}
        norm_sensitivities = {
            k: v / max(layer_sensitivities.values())
            for k, v in layer_sensitivities.items()
        }
        
        for layer_idx, sensitivity in norm_sensitivities.items():
            bits = int(avg_bits * (1 + sensitivity))
            bits = max(min_bits, min(max_bits, bits))
            allocation[layer_idx] = bits
        
        return allocation
