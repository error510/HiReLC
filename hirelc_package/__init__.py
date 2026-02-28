"""
HiReLC: Hierarchical Reinforcement Learning for Model Compression

A modular framework for neural network compression combining:
- Hierarchical Reinforcement Learning (HRL)
- Quantization & Pruning
- Surrogate-guided optimization
- Multi-agent ensemble approaches
"""

__version__ = "0.1.0"
__author__ = "HiReLC Team"

from .config import CompressionGoal, LayerBudget, ExperimentConfig, KernelConfig, LayerConfig
from .core import (
    AdvancedQuantizer,
    AdvancedPruner,
    BaseSensitivityEstimator,
    CPUSensitivityEstimator,
    SurrogateModelTrainer
)
from .agents import (
    BudgetConstrainedCompressionEnv,
    OptimizedEnsembleLowLevelAgent,
    BudgetAllocationEnvironment,
    EnsembleHighLevelAgent
)
from .utils import (
    ExperimentLogger,
    DataManager,
    ReproducibilityManager,
    ComprehensiveVisualizer
)
from .trainers import BaseCompressionTrainer, CNNCompressionTrainer, ViTCompressionTrainer

__all__ = [
    "CompressionGoal",
    "LayerBudget",
    "ExperimentConfig",
    "KernelConfig",
    "LayerConfig",
    "AdvancedQuantizer",
    "AdvancedPruner",
    "BaseSensitivityEstimator",
    "CPUSensitivityEstimator",
    "SurrogateModelTrainer",
    "BudgetConstrainedCompressionEnv",
    "OptimizedEnsembleLowLevelAgent",
    "BudgetAllocationEnvironment",
    "EnsembleHighLevelAgent",
    "ExperimentLogger",
    "DataManager",
    "ReproducibilityManager",
    "ComprehensiveVisualizer",
    "BaseCompressionTrainer",
    "CNNCompressionTrainer",
    "ViTCompressionTrainer",
]
