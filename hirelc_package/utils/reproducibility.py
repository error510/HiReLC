"""Utility modules for logging, data management, and reproducibility"""

import os
import sys
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging


class ReproducibilityManager:
    """Ensures reproducible experiments across different runs."""
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seed for all libraries.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Additional CUDA reproducibility settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    @staticmethod
    def get_device() -> str:
        """Get available device ('cuda' or 'cpu')."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class ExperimentLogger:
    """Comprehensive experiment logging with file and console output."""
    
    def __init__(self, experiment_name: str, output_dir: str = './outputs'):
        """
        Initialize logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Output directory for logs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.run_dir / "experiment.log"
        self.metrics_file = self.run_dir / "metrics.json"
        
        self._init_log_file()
        self.metrics = {}
    
    def _init_log_file(self) -> None:
        """Initialize logging file."""
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def log(self, message: str, level: str = 'INFO', print_console: bool = True) -> None:
        """
        Log a message.
        
        Args:
            message: Message to log
            level: Log level ('INFO', 'WARNING', 'ERROR', 'SUCCESS', 'DEBUG')
            print_console: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
        
        # Print to console
        if print_console:
            if level == 'ERROR':
                print(f"\033[91m{log_message}\033[0m")  # Red
            elif level == 'WARNING':
                print(f"\033[93m{log_message}\033[0m")  # Yellow
            elif level == 'SUCCESS':
                print(f"\033[92m{log_message}\033[0m")  # Green
            else:
                print(log_message)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.log("\nExperiment Configuration:", level='INFO')
        for key, value in config.items():
            self.log(f"  {key}: {value}")
    
    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        """
        Log a metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if key not in self.metrics:
            self.metrics[key] = []
        
        self.metrics[key].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_cycle_result(self, cycle: int, data: Dict[str, Any]) -> None:
        """Log results from a compression cycle."""
        self.log(f"\n--- Cycle {cycle} Results ---", level='INFO')
        for key, value in data.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")
    
    def log_surrogate_training(self, epoch: int, loss: float, samples: int) -> None:
        """Log surrogate model training progress."""
        self.log(f"Surrogate [Epoch {epoch}] Loss: {loss:.6f}, Samples: {samples}")
    
    def log_surrogate_prediction(self, cycle: int, predicted: float, actual: float) -> None:
        """Log surrogate predictions vs actual."""
        error = abs(predicted - actual)
        self.log(f"Surrogate [Cycle {cycle}] Pred: {predicted:.2f}%, Actual: {actual:.2f}%, "
                f"Error: {error:.2f}%")
    
    def save_metrics(self) -> None:
        """Save collected metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_run_dir(self) -> Path:
        """Get the run directory."""
        return self.run_dir
    
    def save_checkpoint(self, model: torch.nn.Module, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        path = checkpoint_dir / f"{name}.pt"
        torch.save(model.state_dict(), path)
        self.log(f"Checkpoint saved: {path}")


class DataManager:
    """Data loading and management utilities."""
    
    @staticmethod
    def get_tinyimagenet(batch_size: int = 128,
                        num_workers: int = 4) -> tuple:
        """
        Load TinyImageNet dataset.
        
        Returns:
            (train_loader, test_loader)
        """
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
        ])
        
        # Note: TinyImageNet should be downloaded manually or via torchvision
        try:
            trainset = torchvision.datasets.ImageNet(
                root='./data', split='train', transform=transform_train
            )
            testset = torchvision.datasets.ImageNet(
                root='./data', split='val', transform=transform_test
            )
        except Exception:
            print("TinyImageNet not found. Using CIFAR-10 instead.")
            trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform_train
            )
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test
            )
        
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
    
    @staticmethod
    def get_cifar10(batch_size: int = 128,
                   num_workers: int = 4) -> tuple:
        """Load CIFAR-10 dataset."""
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
    
    @staticmethod
    def get_cifar100(batch_size: int = 128,
                    num_workers: int = 4) -> tuple:
        """Load CIFAR-100 dataset."""
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761)),
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        
        train_loader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
