"""
Base trainer class for model compression.

Provides shared functionality and abstract interface for CNN and ViT trainers.
"""

import copy
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from abc import ABC, abstractmethod
from pathlib import Path

from hirelc_package.config import ExperimentConfig, LayerConfig
from hirelc_package.utils import ExperimentLogger, ReproducibilityManager


class BaseCompressionTrainer(ABC):
    """Base class for model compression trainers"""
    
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        """
        Initialize trainer
        
        Args:
            config: ExperimentConfig with all parameters
            logger: ExperimentLogger for tracking
        """
        self.config = config
        self.logger = logger
        
        self.model = None
        self.original_model = None
        self.train_loader = None
        self.test_loader = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.baseline_accuracy = 0.0
        self.final_accuracy = 0.0
        self.best_global_config = None
        self.best_global_acc = 0.0
    
    @abstractmethod
    def evaluate(self, model: nn.Module = None, max_batches: int = None) -> float:
        """Evaluate model accuracy"""
        pass
    
    @abstractmethod
    def compute_sensitivity(self):
        """Compute layer sensitivity scores"""
        pass
    
    def finetune_with_early_stopping(self, model: nn.Module, max_epochs: int = 10,
                                     patience: int = 3, lr: float = 5e-5) -> float:
        """
        Fine-tune model with early stopping
        
        Args:
            model: Model to fine-tune
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            lr: Learning rate
            
        Returns:
            Best validation accuracy achieved
        """
        self.logger.log(f"Fine-tuning (Max Epochs: {max_epochs})")
        
        # Create masks to preserve pruning
        masks = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name and param.dim() > 1:
                masks[name] = (param.data != 0).float().to(self.config.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        best_acc = 0.0
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        
        model.train()
        
        for epoch in range(max_epochs):
            running_loss = 0.0
            
            from tqdm import tqdm
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", 
                       leave=False)
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                # Apply masks to maintain pruning
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in masks:
                            param.data *= masks[name]
                
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss/(pbar.n+1))
            
            scheduler.step()
            
            val_acc = self.evaluate(model, max_batches=20)
            self.logger.log(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.log(f"Early stopping at epoch {epoch+1}")
                    break
        
        model.load_state_dict(best_model_state)
        return best_acc
    
    @abstractmethod
    def run_hierarchical_compression(self) -> Dict[int, LayerConfig]:
        """Run the complete hierarchical compression pipeline"""
        pass
