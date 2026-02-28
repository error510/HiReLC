# HiReLC: Hierarchical Reinforcement Learning for Model Compression

A modular, production-ready Python framework for neural network compression based on hierarchical reinforcement learning. Automatically finds optimal combinations of quantization and pruning parameters while preserving model accuracy.

## Core Algorithm

HiReLC solves the neural network compression problem as a **two-level hierarchical Markov Decision Process (MDP)**:

### **Level 1: High-Level Agent (HLA) - Budget Allocation**
- **Decision**: Allocates a compression budget for each layer (target compression ratio per layer)
- **State**: Current compression ratios, accuracy drops, and layer sensitivity scores
- **Reward**: Encourages balanced compression across layers while respecting global accuracy/compression targets
- **Output**: LayerBudget objects that constrain what LLA agents can achieve per layer

### **Level 2: Low-Level Agents (LLA) - Parameter Selection**
- **Decision**: For each layer, selects specific compression parameters:
  - Quantization bitwidth (2-8 bits)
  - Pruning ratio (0-80%)
  - Quantization type (INT or FLOAT)
  - Granularity (uniform, logarithmic, per-channel, learned)
- **State**: Layer compression budget, sensitivity scores, previous performance
- **Reward**: Multi-objective: accuracy preservation + compression achievement + budget compliance
- **Output**: Concrete configuration (KernelConfig) for each kernel/layer

### **The Training Loop (Multi-Cycle)**
For each of N cycles:
1. **Sensitivity Analysis**: Compute how sensitive each layer is to compression (Fisher information)
2. **HLA Allocation**: Train or query HLA to get per-layer budgets
3. **LLA Configuration**: For each layer, use ensemble LLA agents to select parameters within budget
4. **Apply Compression**: Apply quantization masks and pruning to model weights
5. **Fine-tune**: Brief retraining to recover accuracy after compression
6. **Update Surrogate**: Add sample to surrogate model for next cycle's predictions
7. **Adapt HLA**: Feed back cycle results to improve HLA's budget allocation strategy

### **Surrogate Model Optimization**
- A lightweight neural network predicts post-compression accuracy without expensive retraining
- Trained on configurations from previous cycles
- Accelerates decision-making in later cycles
- Enables efficient exploration of the compression parameter space

## Features

✨ **Modular Architecture**: Clean separation of concerns with well-defined APIs
🤖 **Hierarchical RL**: High-level budget allocation + Low-level ensemble agents  
📊 **Multiple Compression Strategies**: Quantization, pruning, mixed-precision support
🎯 **Surrogate-Guided Optimization**: Neural accuracy predictor for faster cycles
📈 **Sensitivity Analysis**: Fisher information, Hessian, gradient-based methods
🔬 **Extensive Logging & Visualization**: Built-in monitoring and analysis tools
📋 **Fully Configurable**: Control quantization types, pruning strategies, RL algorithms

## Project Structure

```
hirelc_package/
├── config/                    # Configuration management
│   ├── __init__.py           # CompressionGoal, LayerBudget, ExperimentConfig
│   └── model_config.py        # KernelConfig, LayerConfig
│
├── core/                      # Core compression algorithms
│   ├── quantization.py        # Multiple quantization methods
│   ├── pruning.py             # Structured and unstructured pruning
│   ├── sensitivity.py         # Sensitivity estimation methods
│   └── surrogate.py           # Surrogate model for accuracy prediction
│
├── agents/                    # RL agents (to be implemented)
│   ├── base_agent.py          # Abstract base classes
│   ├── low_level_agent.py     # Per-layer compression agents
│   └── high_level_agent.py    # Budget allocation agents
│
├── utils/                     # Utilities
│   ├── reproducibility.py     # Seeding and reproducibility
│   ├── visualization.py       # Plotting and analysis
│   └── data_manager.py        # Dataset utilities
│
├── trainers/                  # High-level trainers (to be implemented)
│   ├── base_trainer.py        # Shared base trainer class
│   ├── cnn_trainer.py         # CNN model trainer
│   └── vit_trainer.py         # ViT model trainer
│
├── examples/                  # Example scripts
│   ├── example_cnn.py
│   └── example_vit.py
│
└── references/                # Original unmodified code for reference
    ├── HiReLC_CNN.py
    └── HiReLC_ViT.py
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd hirelc_package

# Install dependencies
pip install torch torchvision timm gymnasium stable-baselines3 numpy pandas matplotlib seaborn scikit-learn tqdm

# Install in development mode
pip install -e .
```

### Basic Usage

```python
from hirelc_package import (
    CompressionGoal,
    ExperimentConfig,
    CNNCompressionTrainer,
    ExperimentLogger,
    ReproducibilityManager
)

# Set reproducibility
ReproducibilityManager.set_seed(42)

# Define compression goal
compression_goal = CompressionGoal(
    target_accuracy_drop=1.0,
    target_compression_ratio=0.25,
    min_layer_bits=2,
    max_layer_bits=8,
    alpha=50.0,
    beta=2.0
)

# Create experiment config
config = ExperimentConfig(
    model_name="resnet18",
    dataset="tinyimagenet",
    num_classes=200,
    batch_size=128,
    num_lla_agents=3,
    num_hla_agents=3,
    lla_timesteps=2048,
    hla_timesteps=2048,
    use_surrogate=True,
    quantization_type='mixed',
    compression_goal=compression_goal,
    experiment_name='my_compression_exp'
)

# Initialize logger and trainer
logger = ExperimentLogger(config.experiment_name)
trainer = CNNCompressionTrainer(config, logger)

# Run hierarchical compression
compressed_configs = trainer.run_hierarchical_compression()

# Generate report
trainer.generate_all_visualizations()
```

## Module Documentation

### Configuration Module (`config/`)

#### CompressionGoal
Defines multi-objective optimization targets:
```python
goal = CompressionGoal(
    target_accuracy_drop=1.0,        # Max 1% accuracy drop
    target_compression_ratio=0.25,   # 4x compression
    min_layer_bits=2,
    max_layer_bits=8,
    alpha=50.0,                      # Accuracy weight
    beta=2.0,                        # Compression weight
    gamma=1.0                        # Compliance weight
)
```

#### ExperimentConfig  
Full experiment configuration with parameter control:
```python
config = ExperimentConfig(
    model_name="resnet18",
    dataset="tinyimagenet",
    sensitivity_method='fisher',  # or 'hessian', 'gradient'
    quantization_type='mixed',    # or 'int', 'float'
    default_strategy=None          # or 'uniform', 'log', 'per-channel'
)

# Save/load configuration
config.to_json('config.json')
config = ExperimentConfig.from_json('config.json')
```

### Core Module (`core/`)

#### Quantization
```python
from hirelc_package.core import AdvancedQuantizer

# Uniform quantization
q_tensor = AdvancedQuantizer.quantize_uniform(tensor, bits=8, symmetric=True)

# Logarithmic quantization (better for skewed distributions)
q_tensor = AdvancedQuantizer.quantize_logarithmic(tensor, bits=8)

# Per-channel quantization
q_tensor = AdvancedQuantizer.quantize_per_channel(tensor, bits=8, axis=0)

# Learned quantization with trainable scale/zero-point
q_tensor, scale, zp = AdvancedQuantizer.quantize_learned(tensor, bits=8)

# Dynamic bit selection
optimal_bits = DynamicQuantizer.select_bitwidth(tensor, candidates=[2, 4, 8, 16])
```

#### Pruning
```python
from hirelc_package.core import AdvancedPruner, FineGrainedPruner

# Magnitude-based pruning
pruned = AdvancedPruner.prune_magnitude(tensor, pruning_ratio=0.5)

# Movement-based pruning (requires initial weights)
pruned = AdvancedPruner.prune_movement(tensor, pruning_ratio=0.5, initial_tensor=initial)

# Gradient-based pruning
pruned = AdvancedPruner.prune_gradient_based(tensor, gradients, pruning_ratio=0.5)

# Structured channel pruning
pruned = FineGrainedPruner.channel_pruning(tensor, pruning_ratio=0.3, axis=0)

# Adaptive allocation based on sensitivity
prune_ratios = AdaptivePruner.allocate_pruning_ratios(
    layer_sensitivities={'layer_0': 0.8, 'layer_1': 0.5},
    target_sparsity=0.5
)
```

#### Sensitivity Estimation
```python
from hirelc_package.core import (
    FisherSensitivityEstimator,
    HessianSensitivityEstimator,
    GradientNormSensitivityEstimator,
    CompositeSensitivityEstimator
)

# Fisher information (recommended)
estimator = FisherSensitivityEstimator(model, dataloader, device='cuda')
scores = estimator.estimate_importance(num_samples=50)

# Gradient-based (fast)
estimator = GradientNormSensitivityEstimator(model, dataloader)
scores = estimator.estimate_importance(num_samples=50)

# Hessian-based (accurate but slow)
estimator = HessianSensitivityEstimator(model, dataloader)
scores = estimator.estimate_importance(num_samples=50)

# Ensemble (best accuracy)
estimator = CompositeSensitivityEstimator(
    model, dataloader, methods=['fisher', 'gradient', 'hessian']
)
scores = estimator.estimate_importance(num_samples=50)
```

#### Surrogate Model
```python
from hirelc_package.core import SurrogateModelTrainer

# Initialize surrogate
surrogate = SurrogateModelTrainer(
    num_blocks=12,
    num_kernels_per_block=4,
    hidden_dims=[64, 32],
    device='cuda'
)

# Add configurations and accuracies
surrogate.add_sample(config_dict={0: layer_config_0, 1: layer_config_1}, accuracy=92.5)

# Train
surrogate.train(epochs=50, batch_size=32)

# Predict
predicted_acc = surrogate.predict(new_config_dict)

# Get statistics
stats = surrogate.get_training_stats()
```

### Utilities Module (`utils/`)

#### Reproducibility
```python
from hirelc_package.utils import ReproducibilityManager

# Set seed for all libraries
ReproducibilityManager.set_seed(42)

# Get device
device = ReproducibilityManager.get_device()  # 'cuda' or 'cpu'
```

#### Logging
```python
from hirelc_package.utils import ExperimentLogger

logger = ExperimentLogger('my_experiment', output_dir='./outputs')

# Log messages
logger.log("Training started", level='INFO')
logger.log("Warning: high loss", level='WARNING')
logger.log("Compression complete", level='SUCCESS')

# Log metrics
logger.log_metric('accuracy', 92.5, step=1)

# Save checkpoints
logger.save_checkpoint(model, 'best_model')

# Get run directory
run_dir = logger.get_run_dir()
```

#### Data Management
```python
from hirelc_package.utils import DataManager

# Load datasets
train_loader, test_loader = DataManager.get_cifar10(batch_size=128)
train_loader, test_loader = DataManager.get_cifar100(batch_size=128)
train_loader, test_loader = DataManager.get_tinyimagenet(batch_size=128)
```

#### Visualization
```python
from hirelc_package.utils import ComprehensiveVisualizer

# Plot kernel decisions
ComprehensiveVisualizer.plot_per_kernel_decisions(
    configs=configs,
    kernel_names=['qkv', 'attn_proj', 'mlp_fc1', 'mlp_fc2'],
    save_path='./outputs/decisions.png'
)

# Plot accuracy vs size
ComprehensiveVisualizer.plot_accuracy_vs_size(
    baseline_acc=95.0, final_acc=94.2,
    baseline_size_mb=50.0, compressed_size_mb=12.5,
    save_path='./outputs/pareto.png'
)

# Plot sensitivity analysis
ComprehensiveVisualizer.plot_sensitivity_analysis(
    sensitivity_scores={0: 0.8, 1: 0.6, ...},
    compression_ratios={0: 0.3, 1: 0.25, ...},
    save_path='./outputs/sensitivity.png'
)
```

## Research Foundation

HiReLC implements the hierarchical reinforcement learning approach for neural network compression:

- **Hierarchical RL**: Decomposes the global compression problem into sub-problems (HLA decides budgets, LLA selects parameters within budget)
- **Quantization**: Supports INT8 and FP8 quantization with multiple granularities (per-tensor, per-channel, learned)
- **Pruning**: Magnitude-based, gradient-based, and Fisher-information-aware pruning strategies
- **Sensitivity Estimation**: Identifies which layers can tolerate compression better (important layers get larger budgets)
- **Ensemble Agents**: Multiple RL agents vote on decisions, reducing single-agent bias
- **Surrogate Optimization**: Predicts post-compression accuracy to skip expensive retraining

### Mathematical Formulation

The problem is formulated as minimizing:
$$\text{Loss} = \alpha \cdot \text{AccuracyDrop} + \beta \cdot \text{CompressionRatio} + \gamma \cdot \text{BudgetDeviation} + \delta \cdot \text{SensitivityPenalty}$$

Where:
- **Accuracy Drop**: Difference between baseline and compressed model accuracy
- **Compression Ratio**: Target compression vs actual achieved (lower is better, capped at 1.0)
- **Budget Deviation**: How well each layer respects its allocated budget
- **Sensitivity Penalty**: Penalizes aggressive compression of sensitive layers

The HLA learns to allocate budgets that minimize this loss, and the LLA learns to select parameters within those budgets.

### CompressionGoal Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_accuracy_drop` | float | 1.0 | Max acceptable accuracy drop (%) |
| `target_compression_ratio` | float | 0.25 | Target compression ratio |
| `min_layer_bits` | int | 2 | Minimum bitwidth |
| `max_layer_bits` | int | 8 | Maximum bitwidth |
| `min_layer_pruning` | float | 0.0 | Minimum pruning ratio |
| `max_layer_pruning` | float | 0.8 | Maximum pruning ratio |
| `alpha, beta, gamma, delta` | float | Various | Loss function weights |

### ExperimentConfig Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "resnet18" | Model architecture to compress |
| `dataset` | str | "tinyimagenet"  | Dataset for evaluation |
| `sensitivity_method` | str | 'fisher' | {'fisher', 'hessian', 'gradient', 'snip'} |
| `quantization_type` | str | 'mixed' | {'mixed', 'int', 'float'} |
| `default_strategy` | str | None | {'uniform', 'log', 'per-channel', 'learned'} |
| `use_surrogate` | bool | True | Use surrogate model for speedup |
| `num_lla_agents` | int | 3 | Number of low-level agents |
| `num_hla_agents` | int | 3 | Number of high-level agents |
| `lla_timesteps` | int | 2048 | LLA training timesteps |
| `hla_timesteps` | int | 2048 | HLA training timesteps |

## Extending the Framework

### Adding Custom Sensitivity Methods
```python
from hirelc_package.core import BaseSensitivityEstimator

class CustomSensitivityEstimator(BaseSensitivityEstimator):
    def estimate_importance(self, num_samples=50):
        # Implement your method
        return layer_scores_dict

# Use in config
config.sensitivity_method = 'custom'
```

### Adding Custom RL Agents
Implement the base agent interface and register in the agents module.

### Adding Custom Pruning Strategies
```python
from hirelc_package.core import AdvancedPruner

# Extend AdvancedPruner with new methods
pruned = AdvancedPruner.custom_pruning(tensor, params)
```

## Performance Tips & Best Practices

### Speed Optimization
1. **Use Surrogate Model**: Enable `use_surrogate=True` to skip expensive retraining in early cycles
   - After surrogate warmup, predictions are ~100-200ms instead of minutes for full finetuning
   - Enables more exploration of the compression space per unit time
   
2. **Reduce Sensitivity Sampling**: Use 'gradient' method instead of 'hessian' 
   - Gradient: ~30 samples, 2-3 minutes
   - Hessian: ~50 samples, 10-15 minutes
   
3. **Lower RL Timesteps in Early Cycles**: Start with 256-512 timesteps, increase to 2048 in final cycle
   
4. **Increase Batch Size**: Use 128-256 if workable for your GPU memory

### Compression Quality
1. **Multi-Cycle Approach**: Use 4+ cycles to allow HLA to adapt its budget allocation
2. **Ensemble LLA Agents**: Keep 3 agents (often outperforms single agents due to voting)
3. **Mixed-Precision Quantization**: Use quantization_type='mixed' to let agents choose INT vs FLOAT
4. **Budget Adaptation**: Enable dynamic budget updates where failed layers get retried with less aggressive targets

## Troubleshooting

**Issue**: Out of memory error
- Solution: Reduce `batch_size`, disable `use_surrogate`, or use gradient checkpointing

**Issue**: Low compression rate achieved
- Solution: Increase `target_compression_ratio`, adjust `max_layer_pruning`, or use more aggressive settings

**Issue**: Accuracy drops too much
- Solution: Decrease `target_compression_ratio`, increase `max_layer_bits`, or increase finetuning epochs

## Testing

Comprehensive test suite included to validate all functionality:

```bash
# Run all tests
python run_all_tests.py

# Run individual test modules
python test_config.py      # Test configuration classes
python test_core.py        # Test quantization, pruning, sensitivity
python test_utils.py       # Test logger, reproducibility, data loading
python test_integration.py # Test full pipeline
```

See [TEST_README.md](TEST_README.md) for detailed test documentation.

## Citation

If you use HiReLC in your research, please cite:
```
@framework{hirelc2024,
  title={Hierarchical Reinforcement Learning for Neuron Networks Compression (HiReLC): Pruning and Quantization},
  author={},
  year={2026}
}
```

## License

MIT License - feel free to use for research and commercial purposes

## Contributing

Contributions welcome! Areas for contribution:
- Additional sensitivity estimation methods
- New pruning strategies
- RL algorithm implementations
- Optimization for specific hardware (edge devices, TPUs)
- Documentation and examples

## References

- Original Implementation: `Paper Experements/HiReLC_CNN.py` and `HiReLC_ViT.py`
- RL Algorithms: stable-baselines3 (PPO, A2C)
- Quantization Research: [Add relevant papers]
- Pruning Research: [Add relevant papers]

---

For questions, issues, or feature requests, please open an issue on GitHub.
