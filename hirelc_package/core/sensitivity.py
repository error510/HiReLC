"""Sensitivity estimation for layer importance analysis"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseSensitivityEstimator(ABC):
    """Abstract base class for all sensitivity estimators."""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.sensitivity_scores = {}
    
    @abstractmethod
    def estimate_importance(self, num_samples: int = 50) -> Dict[int, float]:
        """Estimate layer importance. Returns dict[layer_idx -> score]."""
        pass
    
    def get_scores(self) -> Dict[int, float]:
        """Get computed sensitivity scores."""
        return self.sensitivity_scores


class FisherSensitivityEstimator(BaseSensitivityEstimator):
    """
    Fisher information based sensitivity estimation.
    Measures how much removing a parameter affects the loss.
    """
    
    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 device: str = 'cuda', num_blocks: Optional[int] = None):
        super().__init__(model, dataloader, device)
        self.num_blocks = num_blocks
        self.fisher_dict = {}
    
    def compute_fisher_information(self, num_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information matrix diagonal for all parameters.
        
        Returns:
            Dict mapping parameter names to their Fisher scores
        """
        self.model.eval()
        fisher = {}
        
        # Initialize fisher matrix
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.dataloader, 
                                                            desc="Computing Fisher")):
            if batch_idx >= num_samples:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Compute gradients
            loss.backward(retain_graph=True)
            
            # Accumulate Fisher (gradient^2)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Average over samples
        for key in fisher:
            fisher[key] /= min(num_samples, batch_idx + 1)
        
        self.fisher_dict = fisher
        return fisher
    
    def get_layer_sensitivity_scores(self) -> Dict[int, float]:
        """
        Compute layer-wise sensitivity from parameter-wise Fisher scores.
        
        Returns:
            Dict[layer_idx -> sensitivity_score]
        """
        layer_scores = {}
        
        if self.num_blocks is None:
            # Auto-detect number of blocks from model structure
            self.num_blocks = self._detect_num_blocks()
        
        for layer_idx in range(self.num_blocks):
            score = 0.0
            count = 0
            
            for name, fisher_score in self.fisher_dict.items():
                # Check if parameter belongs to this layer
                if self._belongs_to_layer(name, layer_idx):
                    score += fisher_score.sum().item()
                    count += 1
            
            if count > 0:
                layer_scores[layer_idx] = score / count
        
        self.sensitivity_scores = layer_scores
        return layer_scores
    
    def _belongs_to_layer(self, param_name: str, layer_idx: int) -> bool:
        """Check if a parameter belongs to a specific layer."""
        if 'block' in param_name:
            return f'block.{layer_idx}' in param_name or f'blocks.{layer_idx}' in param_name
        return False
    
    def _detect_num_blocks(self) -> int:
        """Auto-detect number of blocks from model structure."""
        max_idx = -1
        for name in self.fisher_dict.keys():
            for keyword in ['block', 'blocks', 'layer', 'layers']:
                if keyword in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if keyword in part and i + 1 < len(parts):
                            try:
                                idx = int(parts[i + 1])
                                max_idx = max(max_idx, idx)
                            except (ValueError, IndexError):
                                pass
        return max(12, max_idx + 1)  # Default to 12 if can't auto-detect
    
    def estimate_importance(self, num_samples: int = 50) -> Dict[int, float]:
        """Estimate and return layer importance."""
        self.compute_fisher_information(num_samples)
        return self.get_layer_sensitivity_scores()


class HessianSensitivityEstimator(BaseSensitivityEstimator):
    """
    Hessian-based sensitivity estimation.
    Approximates second-order information about parameter importance.
    """
    
    def compute_hessian_diagonal(self, num_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal of Hessian matrix approximation.
        More computationally expensive than Fisher but potentially more accurate.
        """
        self.model.eval()
        hessian = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hessian[name] = torch.zeros_like(param.data)
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.dataloader,
                                                            desc="Computing Hessian")):
            if batch_idx >= num_samples:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # First pass: compute gradients
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            grads = torch.autograd.grad(loss, self.model.parameters(),
                                       create_graph=True, retain_graph=True)
            
            # Second pass: compute Hessian-vector product
            for i, grad in enumerate(grads):
                if grad is not None:
                    grad_norm = grad.pow(2).sum()
                    self.model.zero_grad()
                    grad_norm.backward(retain_graph=True)
                    
                    param = list(self.model.parameters())[i]
                    if param.grad is not None:
                        hessian[list(self.model.named_parameters())[i][0]] += param.grad.data.abs()
        
        for key in hessian:
            hessian[key] /= min(num_samples, batch_idx + 1)
        
        return hessian
    
    def estimate_importance(self, num_samples: int = 50) -> Dict[int, float]:
        """Estimate importance using Hessian approximation."""
        hessian = self.compute_hessian_diagonal(num_samples)
        
        layer_scores = {}
        for layer_idx in range(self._detect_num_blocks()):
            score = sum(h.sum().item() for name, h in hessian.items()
                       if f'block.{layer_idx}' in name or f'blocks.{layer_idx}' in name)
            if score > 0:
                layer_scores[layer_idx] = score
        
        self.sensitivity_scores = layer_scores
        return layer_scores
    
    def _detect_num_blocks(self) -> int:
        """Auto-detect number of blocks."""
        max_idx = -1
        for name, _ in self.model.named_parameters():
            for keyword in ['block', 'blocks']:
                if keyword in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if keyword in part and i + 1 < len(parts):
                            try:
                                idx = int(parts[i + 1])
                                max_idx = max(max_idx, idx)
                            except (ValueError, IndexError):
                                pass
        return max(12, max_idx + 1)


class GradientNormSensitivityEstimator(BaseSensitivityEstimator):
    """
    Simple gradient norm based sensitivity.
    Measures how much gradients flow through each layer.
    Faster than Fisher but less accurate.
    """
    
    def estimate_importance(self, num_samples: int = 50) -> Dict[int, float]:
        """Estimate importance via gradient norms."""
        self.model.eval()
        gradient_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                gradient_norms[name] = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.dataloader,
                                                            desc="Computing Gradients")):
            if batch_idx >= num_samples:
                break
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_norms[name] += param.grad.data.norm().item()
        
        # Average and aggregate by layer
        layer_scores = {}
        num_blocks = self._detect_num_blocks()
        
        for layer_idx in range(num_blocks):
            score = 0.0
            count = 0
            for name, norm in gradient_norms.items():
                if self._belongs_to_layer(name, layer_idx):
                    score += norm
                    count += 1
            if count > 0:
                layer_scores[layer_idx] = score / count
        
        self.sensitivity_scores = layer_scores
        return layer_scores
    
    def _belongs_to_layer(self, param_name: str, layer_idx: int) -> bool:
        """Check if parameter belongs to layer."""
        return (f'block.{layer_idx}' in param_name or 
                f'blocks.{layer_idx}' in param_name)
    
    def _detect_num_blocks(self) -> int:
        """Auto-detect number of blocks."""
        max_idx = -1
        for name, _ in self.model.named_parameters():
            for keyword in ['block', 'blocks']:
                if keyword in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if keyword in part and i + 1 < len(parts):
                            try:
                                idx = int(parts[i + 1])
                                max_idx = max(max_idx, idx)
                            except (ValueError, IndexError):
                                pass
        return max(12, max_idx + 1)


class CompositeSensitivityEstimator(BaseSensitivityEstimator):
    """Ensemble of multiple sensitivity estimators."""
    
    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 device: str = 'cuda', num_blocks: Optional[int] = None,
                 methods: List[str] = None):
        super().__init__(model, dataloader, device)
        self.num_blocks = num_blocks
        self.methods = methods or ['fisher', 'gradient']
        self.estimators = {}
        self._init_estimators()
    
    def _init_estimators(self):
        """Initialize sub-estimators."""
        if 'fisher' in self.methods:
            self.estimators['fisher'] = FisherSensitivityEstimator(
                self.model, self.dataloader, self.device, self.num_blocks
            )
        if 'gradient' in self.methods:
            self.estimators['gradient'] = GradientNormSensitivityEstimator(
                self.model, self.dataloader, self.device
            )
        if 'hessian' in self.methods:
            self.estimators['hessian'] = HessianSensitivityEstimator(
                self.model, self.dataloader, self.device
            )
    
    def estimate_importance(self, num_samples: int = 50) -> Dict[int, float]:
        """Ensemble importance estimation (average across methods)."""
        all_scores = {}
        
        for method_name, estimator in self.estimators.items():
            scores = estimator.estimate_importance(num_samples)
            all_scores[method_name] = scores
        
        # Average scores across methods
        combined_scores = {}
        if all_scores:
            layer_indices = list(list(all_scores.values())[0].keys())
            for layer_idx in layer_indices:
                scores_for_layer = [
                    all_scores[method].get(layer_idx, 0.0)
                    for method in all_scores
                ]
                combined_scores[layer_idx] = np.mean(scores_for_layer)
        
        self.sensitivity_scores = combined_scores
        return combined_scores


# CPU-specific alias for compatibility
CPUSensitivityEstimator = GradientNormSensitivityEstimator
