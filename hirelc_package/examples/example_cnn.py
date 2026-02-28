"""
Example: Compressing a CNN Model (ResNet18)

This example demonstrates how to use HiReLC to compress a CNN model
on the TinyImageNet dataset.
"""

import sys
import torch
from pathlib import Path

# Add parent to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from hirelc_package import (
    CompressionGoal,
    ExperimentConfig,
    ReproducibilityManager,
    ExperimentLogger,
)

# ============================================================================
# EXAMPLE 1: Basic Setup with Default Configuration
# ============================================================================
def example_basic():
    """Run compression with default configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic CNN Compression")
    print("="*80)
    
    # Set reproducibility
    ReproducibilityManager.set_seed(42)
    device = ReproducibilityManager.get_device()
    print(f"Device: {device}")
    
    # Define compression goal
    compression_goal = CompressionGoal(
        target_accuracy_drop=1.0,
        target_compression_ratio=0.25,  # 4x compression
        min_layer_bits=2,
        max_layer_bits=8,
        min_layer_pruning=0.0,
        max_layer_pruning=0.6,
        alpha=50.0,  # Accuracy weight
        beta=2.0,    # Compression weight
        gamma=1.0,   # Compliance weight
    )
    
    print("\nCompression Goal:")
    print(f"  Target Accuracy Drop: {compression_goal.target_accuracy_drop}%")
    print(f"  Target Compression: {1/compression_goal.target_compression_ratio:.1f}x")
    print(f"  Bitwidth Range: [{compression_goal.min_layer_bits}, {compression_goal.max_layer_bits}]")
    print(f"  Pruning Range: [{compression_goal.min_layer_pruning:.0%}, {compression_goal.max_layer_pruning:.0%}]")
    
    # Create experiment configuration
    config = ExperimentConfig(
        model_name="resnet18",
        dataset="tinyimagenet",
        num_classes=200,
        batch_size=128,
        finetune_epochs=10,
        num_lla_agents=3,
        num_hla_agents=3,
        lla_timesteps=512,        # Reduce for quick test
        hla_timesteps=512,        # Reduce for quick test
        use_surrogate=True,
        quantization_type='mixed',
        sensitivity_method='fisher',
        compression_goal=compression_goal,
        experiment_name='example_cnn_basic',
        device=device,
    )
    
    print("\nExperiment Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Sensitivity Method: {config.sensitivity_method}")
    print(f"  Quantization Type: {config.quantization_type}")
    
    # Save configuration
    config_path = Path('./outputs/cnn_config.json')
    config.to_json(str(config_path))
    print(f"\nConfiguration saved to: {config_path}")


# ============================================================================
# EXAMPLE 2: Custom Configuration with Different Sensitivity Methods
# ============================================================================
def example_sensitivity_methods():
    """Compare different sensitivity estimation methods."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Different Sensitivity Estimation Methods")
    print("="*80)
    
    ReproducibilityManager.set_seed(42)
    
    sensitivity_methods = ['fisher', 'gradient', 'hessian']
    
    for method in sensitivity_methods:
        config = ExperimentConfig(
            model_name="resnet18",
            dataset="cifar10",
            num_classes=10,
            batch_size=128,
            sensitivity_method=method,
            lla_timesteps=256,
            hla_timesteps=256,
            experiment_name=f'example_cnn_{method}',
        )
        
        print(f"\nConfiguration for method '{method}':")
        print(f"  Sensitivity Method: {config.sensitivity_method}")
        
        # Can now use this config to run compression with specific sensitivity method


# ============================================================================
# EXAMPLE 3: Aggressive Compression Settings
# ============================================================================
def example_aggressive_compression():
    """Setup for aggressive compression (higher compression, lower accuracy)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Aggressive Compression Settings")
    print("="*80)
    
    # More aggressive goal
    compression_goal = CompressionGoal(
        target_accuracy_drop=3.0,           # Accept 3% accuracy drop
        target_compression_ratio=0.1,       # 10x compression (aggressive!)
        min_layer_bits=2,
        max_layer_bits=4,                   # Force lower bitwidths
        min_layer_pruning=0.3,              # Higher minimum pruning
        max_layer_pruning=0.8,              # Allow more pruning
        alpha=20.0,                         # Lower weight on accuracy
        beta=10.0,                          # Higher weight on compression
    )
    
    config = ExperimentConfig(
        model_name="mobilenetv2",
        dataset="cifar10",
        num_classes=10,
        compression_goal=compression_goal,
        quantization_type='int',            # Force integer quantization
        default_strategy='log',             # Use log quantization
        experiment_name='example_cnn_aggressive',
    )
    
    print("\nAggressive Compression Settings:")
    print(f"  Accuracy Drop Tolerance: {compression_goal.target_accuracy_drop}%")
    print(f"  Target Compression: {1/compression_goal.target_compression_ratio:.1f}x")
    print(f"  Bitwidth: [{compression_goal.min_layer_bits}, "
          f"{compression_goal.max_layer_bits}]")
    print(f"  Quantization: {config.quantization_type}")
    print(f"  Strategy: {config.default_strategy}")


# ============================================================================
# EXAMPLE 4: Fine-grained Control with Multiple Agents
# ============================================================================
def example_multi_agent_config():
    """Configuration with multiple RL agents for better exploration."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Agent Configuration")
    print("="*80)
    
    config = ExperimentConfig(
        model_name="resnet50",
        dataset="cifar100",
        num_classes=100,
        # Use more agents for better search
        num_lla_agents=5,           # More low-level agents
        num_hla_agents=5,           # More high-level agents
        lla_timesteps=2048,         # More training
        hla_timesteps=2048,
        use_surrogate=True,
        compression_goal=CompressionGoal(
            target_compression_ratio=0.3,
            target_accuracy_drop=1.5,
        ),
        experiment_name='example_cnn_multiagent',
    )
    
    print("\nMulti-Agent Configuration:")
    print(f"  Low-Level Agents: {config.num_lla_agents}")
    print(f"  High-Level Agents: {config.num_hla_agents}")
    print(f"  LLA Timesteps: {config.lla_timesteps}")
    print(f"  HLA Timesteps: {config.hla_timesteps}")
    print(f"  Surrogate Model: {'Enabled' if config.use_surrogate else 'Disabled'}")


# ============================================================================
# EXAMPLE 5: Minimal Compression (Conservative)
# ============================================================================
def example_minimal_compression():
    """Setup for minimal compression (preserve accuracy at all costs)."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Minimal Compression (Conservative)")
    print("="*80)
    
    compression_goal = CompressionGoal(
        target_accuracy_drop=0.1,           # Accept only 0.1% drop
        target_compression_ratio=0.5,       # 2x compression max
        min_layer_bits=4,
        max_layer_bits=8,
        min_layer_pruning=0.0,
        max_layer_pruning=0.2,              # Limited pruning
        alpha=100.0,                        # Very high accuracy weight
        beta=0.5,                           # Low compression weight
    )
    
    config = ExperimentConfig(
        model_name="vgg16",
        dataset="cifar10",
        num_classes=10,
        compression_goal=compression_goal,
        quantization_type='float',         # Use float for better accuracy
        experiment_name='example_cnn_conservative',
    )
    
    print("\nMinimal/Conservative Compression:")
    print(f"  Accuracy Drop Tolerance: {compression_goal.target_accuracy_drop}%")
    print(f"  Max Compression: {1/compression_goal.target_compression_ratio:.1f}x")
    print(f"  Data Type: {config.quantization_type}")
    print(f"  Alpha (accuracy):    {compression_goal.alpha}")
    print(f"  Beta (compression):  {compression_goal.beta}")


# ============================================================================
# EXAMPLE 6: Save and Load Configuration
# ============================================================================
def example_config_serialization():
    """Demonstrate saving and loading configurations."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Configuration Serialization")
    print("="*80)
    
    # Create config
    config = ExperimentConfig(
        model_name="resnet18",
        dataset="cifar10",
        num_classes=10,
        batch_size=256,
        num_lla_agents=4,
        compression_goal=CompressionGoal(
            target_compression_ratio=0.25,
            target_accuracy_drop=1.0,
        ),
        experiment_name='serialization_test',
    )
    
    # Save to JSON
    config_file = Path('./outputs/config_example.json')
    config.to_json(str(config_file))
    print(f"\nConfiguration saved to: {config_file}")
    
    # Load from JSON
    loaded_config = ExperimentConfig.from_json(str(config_file))
    print(f"Configuration loaded: {loaded_config.model_name} on {loaded_config.dataset}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("HiReLC CNN Compression Examples")
    print("="*80)
    
    # Run all examples
    example_basic()
    example_sensitivity_methods()
    example_aggressive_compression()
    example_multi_agent_config()
    example_minimal_compression()
    example_config_serialization()
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
