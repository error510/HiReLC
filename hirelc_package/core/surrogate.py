"""Surrogate model for accuracy prediction"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Any, Tuple
import time
import copy
from collections import defaultdict


class AccuracySurrogateModel(nn.Module):
    """
    Neural network surrogate model that predicts final accuracy
    based on compression configuration.
    
    Reduces the need for expensive full model evaluation during search.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layer_dims = [input_dim] + hidden_dims + [1]
        layers = []
        
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No activation on last layer
                layers.append(nn.ReLU())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        
        Returns:
            Predicted accuracy (batch_size, 1)
        """
        return self.net(x)


class SurrogateModelTrainer:
    """
    Trainer for the surrogate model.
    
    Collects compression configurations and their corresponding accuracies,
    then trains a surrogate to predict accuracy without full evaluations.
    """
    
    # Normalization constants for encoding
    _TYPE_NORM = {'INT': 0.0, 'FLOAT': 1.0}
    _GRAN_NORM = {
        'uniform': 0.00,
        'log': 0.33,
        'per-channel': 0.67,
        'learned': 1.00
    }
    _FEATURES_PER_KERNEL = 5  # [bits, pruning, type, gran, (reserved)]
    
    def __init__(self,
                 num_blocks: int,
                 num_kernels_per_block: int = 4,
                 hidden_dims: List[int] = None,
                 device: str = 'cuda',
                 baseline_accuracy: float = 85.0,
                 logger: Optional[Any] = None):
        """
        Initialize surrogate trainer.
        
        Args:
            num_blocks: Number of compression blocks
            num_kernels_per_block: Number of kernels per block (e.g., 4 for ViT)
            hidden_dims: Hidden layer dimensions
            device: Computation device ('cuda' or 'cpu')
            baseline_accuracy: Initial model accuracy
            logger: Optional logger instance
        """
        self.device = device
        self.num_blocks = num_blocks
        self.num_kernels_per_block = num_kernels_per_block
        self.baseline_accuracy = baseline_accuracy
        self.logger = logger
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        # Input dimension: each block has kernels with multiple features
        self.input_dim = num_blocks * num_kernels_per_block * self._FEATURES_PER_KERNEL
        
        self.model = AccuracySurrogateModel(self.input_dim, hidden_dims).to(device)
        self.optimizer = None
        self.criterion = None
        
        # Training data buffers
        self.config_buffer = []  # List of encoded configs
        self.accuracy_buffer = []  # List of accuracies
        
        # Training history
        self.best_loss = float('inf')
        self.best_state = None
        self.training_history = {
            'epochs': [],
            'losses': [],
            'samples': [],
            'train_times': []
        }
        
        # Cumulative tracking
        self.total_training_epochs = 0
        self.total_training_time = 0.0
    
    def encode_config(self, configs: Dict[int, Any]) -> torch.Tensor:
        """
        Encode compression configurations into a feature vector.
        
        Args:
            configs: Dict[block_idx -> LayerConfig or similar]
        
        Returns:
            Encoded tensor of shape [input_dim]
        """
        features = []

        def _default_features():
            return [8.0, 1.0, 0.0, 0.0, 0.0]

        for block_idx in range(self.num_blocks):
            if block_idx not in configs:
                for _ in range(self.num_kernels_per_block):
                    features.extend(_default_features())
                continue

            config = configs[block_idx]

            kernel_list: List[Any] = []
            if hasattr(config, 'kernel1_config') or hasattr(config, 'kernel2_config'):
                kernel_list = [
                    getattr(config, 'kernel1_config', None),
                    getattr(config, 'kernel2_config', None),
                    getattr(config, 'kernel3_config', None),
                    getattr(config, 'kernel4_config', None),
                ]
            elif hasattr(config, 'kernels') and config.kernels:
                if all(k in config.kernels for k in ['kernel_0', 'kernel_1', 'kernel_2', 'kernel_3']):
                    ordered = ['kernel_0', 'kernel_1', 'kernel_2', 'kernel_3']
                elif all(k in config.kernels for k in ['kernel1', 'kernel2', 'kernel3', 'kernel4']):
                    ordered = ['kernel1', 'kernel2', 'kernel3', 'kernel4']
                elif all(k in config.kernels for k in ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']):
                    ordered = ['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2']
                else:
                    ordered = sorted(config.kernels.keys())
                kernel_list = [config.kernels.get(k) for k in ordered]

            for i in range(self.num_kernels_per_block):
                kc = kernel_list[i] if i < len(kernel_list) else None
                if kc is None:
                    features.extend(_default_features())
                    continue
                bits_norm = float(kc.weight_bits) / 10.0
                prune_norm = float(kc.pruning_ratio)
                type_norm = self._TYPE_NORM.get(getattr(kc, 'quant_type', 'INT'), 0.0)
                gran_norm = self._GRAN_NORM.get(getattr(kc, 'quant_mode', 'uniform'), 0.0)
                combined_flag = 1.0 if (kc.pruning_ratio < 1.0 and kc.weight_bits < 8) else 0.5
                features.extend([bits_norm, prune_norm, type_norm, gran_norm, combined_flag])

        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def add_sample(self,
                   configs: Dict[int, Any],
                   post_finetune_accuracy: Optional[float] = None,
                   accuracy: Optional[float] = None) -> None:
        """
        Add a configuration and its accuracy to the training buffer.
        
        Args:
            configs: Compression configuration dictionary
            post_finetune_accuracy: Final accuracy after compression
        """
        acc_value = accuracy if accuracy is not None else post_finetune_accuracy
        if acc_value is None:
            raise ValueError("Accuracy value must be provided")

        encoded = self.encode_config(configs)
        self.config_buffer.append(encoded)
        self.accuracy_buffer.append(acc_value)
        
        if self.logger:
            self.logger.log_surrogate_prediction(
                len(self.config_buffer),
                predicted=acc_value,
                actual=acc_value
            )
    
    def train(self, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train the surrogate model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if len(self.config_buffer) < 3:
            if self.logger:
                self.logger.log("Insufficient samples for surrogate training (need >= 3)")
            return
        
        start_time = time.time()

        if self.optimizer is None:
            try:
                self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
                self.criterion = nn.MSELoss()
            except Exception as e:
                if self.logger:
                    self.logger.log(f"Surrogate optimizer init failed: {e}", level='WARNING')
                return
        
        # Prepare data
        X = torch.stack(self.config_buffer)
        y = torch.tensor(self.accuracy_buffer, dtype=torch.float32,
                        device=self.device).unsqueeze(1)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True
        )
        
        self.model.train()
        best_epoch_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = self.criterion(pred, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.training_history['epochs'].append(self.total_training_epochs + epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['samples'].append(len(self.config_buffer))
            
            if self.logger and epoch % max(1, epochs // 5) == 0:
                self.logger.log_surrogate_training(
                    self.total_training_epochs + epoch,
                    avg_loss,
                    len(self.config_buffer)
                )
            
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
        
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        
        self.best_loss = best_epoch_loss
        self.total_training_epochs += epochs
        
        elapsed = time.time() - start_time
        self.total_training_time += elapsed
        self.training_history['train_times'].append(elapsed)
        
        if self.logger:
            self.logger.log(f"Surrogate trained: loss={self.best_loss:.6f}, "
                          f"time={elapsed:.2f}s", level='SUCCESS')
    
    def predict(self, configs: Dict[int, Any]) -> float:
        """
        Predict accuracy for a configuration.
        
        Args:
            configs: Compression configuration dictionary
        
        Returns:
            Predicted accuracy (0-100)
        """
        self.model.eval()
        with torch.no_grad():
            encoded = self.encode_config(configs)
            if len(encoded.shape) == 1:
                encoded = encoded.unsqueeze(0)
            pred = self.model(encoded)
            pred_value = pred.squeeze().item()
        
        # Clamp to reasonable range
        return max(0.0, min(100.0, pred_value))
    
    def get_buffer_size(self) -> int:
        """Get number of samples in training buffer."""
        return len(self.config_buffer)
    
    def update_baseline(self, baseline_accuracy: float) -> None:
        """Update baseline accuracy."""
        self.baseline_accuracy = baseline_accuracy
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics and history."""
        return {
            'total_samples': len(self.config_buffer),
            'total_epochs': self.total_training_epochs,
            'total_time': self.total_training_time,
            'best_loss': self.best_loss,
            'history': self.training_history,
            'baseline_accuracy': self.baseline_accuracy,
        }
