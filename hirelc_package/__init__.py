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
    SNIPSensitivityEstimator,
    register_sensitivity_estimator,
    get_sensitivity_estimator,
    build_sensitivity_estimator,
    SurrogateModelTrainer
)
from .utils import (
    ExperimentLogger,
    DataManager,
    ReproducibilityManager,
    ComprehensiveVisualizer
)

# Optional imports (avoid hard dependency errors during basic usage/tests)
try:
    from .agents import (
        BudgetConstrainedCompressionEnv,
        OptimizedEnsembleLowLevelAgent,
        BudgetAllocationEnvironment,
        EnsembleHighLevelAgent
    )
except Exception:
    BudgetConstrainedCompressionEnv = None
    OptimizedEnsembleLowLevelAgent = None
    BudgetAllocationEnvironment = None
    EnsembleHighLevelAgent = None

try:
    from .trainers import BaseCompressionTrainer, CNNCompressionTrainer, ViTCompressionTrainer
except Exception:
    BaseCompressionTrainer = None
    CNNCompressionTrainer = None
    ViTCompressionTrainer = None

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
    "SNIPSensitivityEstimator",
    "register_sensitivity_estimator",
    "get_sensitivity_estimator",
    "build_sensitivity_estimator",
    "SurrogateModelTrainer",
    "ExperimentLogger",
    "DataManager",
    "ReproducibilityManager",
    "ComprehensiveVisualizer",
]

if BudgetConstrainedCompressionEnv is not None:
    __all__.extend([
        "BudgetConstrainedCompressionEnv",
        "OptimizedEnsembleLowLevelAgent",
        "BudgetAllocationEnvironment",
        "EnsembleHighLevelAgent",
    ])

if BaseCompressionTrainer is not None:
    __all__.extend([
        "BaseCompressionTrainer",
        "CNNCompressionTrainer",
        "ViTCompressionTrainer",
    ])
