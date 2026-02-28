"""
Example: Compressing a Vision Transformer (ViT)

This example demonstrates how to use HiReLC to compress Vision Transformer
models on CIFAR10/CIFAR100.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hirelc_package import (
    CompressionGoal,
    ExperimentConfig,
    ReproducibilityManager,
    ExperimentLogger,
    ExperimentConfig,
)


# ============================================================================
# EXAMPLE 1: ViT Compression on CIFAR-10
# ============================================================================
def example_vit_cifar10():
    """Compress ViT-Tiny on CIFAR-10."""
    print("\n" + "="*80)
    print("EXAMPLE 1: ViT Compression on CIFAR-10")
    print("="*80)
    
    ReproducibilityManager.set_seed(42)
    device = ReproducibilityManager.get_device()
    
    compression_goal = CompressionGoal(
        target_accuracy_drop=1.5,
        target_compression_ratio=0.2,       # 5x compression
        min_layer_bits=2,
        max_layer_bits=8,
        min_layer_pruning=0.0,
        max_layer_pruning=0.5,
    )
    
    config = ExperimentConfig(
        model_name="vit_tiny_patch16_224",  # Tiny ViT
        dataset="cifar10",
        num_classes=10,
        batch_size=128,
        num_lla_agents=3,
        num_hla_agents=3,
        lla_timesteps=1024,
        hla_timesteps=1024,
        use_surrogate=True,
        quantization_type='mixed',
        compression_goal=compression_goal,
        experiment_name='example_vit_cifar10',
        device=device,
    )
    
    print("\nViT Compression Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Target Compression: {1/compression_goal.target_compression_ratio:.1f}x")
    print(f"  Device: {device}")


# ============================================================================
# EXAMPLE 2: Multi-Scale ViT Comparison
# ============================================================================
def example_vit_variants():
    """Compare different ViT model sizes."""
    print("\n" + "="*80)
    print("EXAMPLE 2: ViT Variants Compression")
    print("="*80)
    
    vit_models = [
        ("vit_tiny_patch16_224", "Tiny"),
        ("vit_small_patch16_224", "Small"),
        ("vit_base_patch16_224", "Base"),
    ]
    
    compression_goal = CompressionGoal(
        target_compression_ratio=0.25,
        target_accuracy_drop=1.0,
    )
    
    for model_name, variant in vit_models:
        config = ExperimentConfig(
            model_name=model_name,
            dataset="cifar10",
            num_classes=10,
            compression_goal=compression_goal,
            experiment_name=f'example_vit_{variant.lower()}',
        )
        
        print(f"\n  {variant} ViT Configuration:")
        print(f"    Model: {config.model_name}")


# ============================================================================
# EXAMPLE 3: ViT on CIFAR-100 (More Challenging)
# ============================================================================
def example_vit_cifar100():
    """Compress ViT on more challenging CIFAR-100."""
    print("\n" + "="*80)
    print("EXAMPLE 3: ViT Compression on CIFAR-100")
    print("="*80)
    
    compression_goal = CompressionGoal(
        target_accuracy_drop=2.0,           # Allow more drop for harder task
        target_compression_ratio=0.3,       # 3.3x compression
        min_layer_bits=3,
        max_layer_bits=8,
        min_layer_pruning=0.0,
        max_layer_pruning=0.4,
        alpha=40.0,                         # Slightly lower accuracy weight
        beta=3.0,                           # Slightly higher compression weight
    )
    
    config = ExperimentConfig(
        model_name="vit_small_patch16_224",
        dataset="cifar100",
        num_classes=100,
        batch_size=64,                      # Smaller batch for CIFAR-100
        finetune_epochs=15,
        num_lla_agents=4,
        num_hla_agents=4,
        lla_timesteps=2048,
        hla_timesteps=2048,
        use_surrogate=True,
        compression_goal=compression_goal,
        experiment_name='example_vit_cifar100',
    )
    
    print("\nCIFAR-100 (100 Classes) Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Classes: {config.num_classes}")
    print(f"  Target Compression: {1/compression_goal.target_compression_ratio:.1f}x")


# ============================================================================
# EXAMPLE 4: ViT with Different Quantization Strategies
# ============================================================================
def example_vit_quantization_strategies():
    """Compare different quantization strategies for ViT."""
    print("\n" + "="*80)
    print("EXAMPLE 4: ViT Quantization Strategies")
    print("="*80)
    
    strategies = [
        ('uniform', 'Uniform'),
        ('log', 'Logarithmic'),
        ('per-channel', 'Per-Channel'),
        ('learned', 'Learned'),
    ]
    
    for strategy, name in strategies:
        config = ExperimentConfig(
            model_name="vit_tiny_patch16_224",
            dataset="cifar10",
            quantization_type='mixed',
            default_strategy=strategy,
            experiment_name=f'example_vit_strategy_{strategy}',
        )
        
        print(f"\n  {name} Quantization:")
        print(f"    Strategy: {config.default_strategy}")


# ============================================================================
# EXAMPLE 5: ViT with Integer-Only Quantization (for edge devices)
# ============================================================================
def example_vit_integer_only():
    """Setup ViT compression with integer-only quantization."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Integer-Only ViT (Edge Device Ready)")
    print("="*80)
    
    config = ExperimentConfig(
        model_name="vit_tiny_patch16_224",
        dataset="cifar10",
        num_classes=10,
        quantization_type='int',           # Force integer quantization
        default_strategy='uniform',         # Use uniform quantization
        compression_goal=CompressionGoal(
            target_compression_ratio=0.2,
            min_layer_bits=8,               # Minimum 8 bits for integer
            max_layer_bits=8,               # Fixed to 8 bits (INT8)
        ),
        experiment_name='example_vit_int8_edge',
    )
    
    print("\nInteger-Only (INT8) Configuration:")
    print(f"  Quantization Type: {config.quantization_type}")
    print(f"  Strategy: {config.default_strategy}")
    print("  Note: INT8 models run efficiently on")
    print("        - Mobile devices (NNAPI, CoreML)")
    print("        - IoT devices (TensorFlow Lite)")
    print("        - FPGA/Hardware accelerators")


# ============================================================================
# EXAMPLE 6: Fast ViT Compression (Quick Testing)
# ============================================================================
def example_vit_fast_compression():
    """Minimal settings for fast testing (not production quality)."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Fast ViT Compression (for testing)")
    print("="*80)
    
    config = ExperimentConfig(
        model_name="vit_tiny_patch16_224",
        dataset="cifar10",
        num_classes=10,
        batch_size=32,                      # Small batch
        finetune_epochs=2,                  # Quick finetuning
        lla_timesteps=128,                  # Minimal RL training
        hla_timesteps=128,
        use_surrogate=False,                # Skip surrogate for speed
        sensitivity_method='gradient',      # Fast method
        experiment_name='example_vit_fast_test',
    )
    
    print("\nFast Testing Configuration:")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Finetune Epochs: {config.finetune_epochs}")
    print(f"  LLA Timesteps: {config.lla_timesteps}")
    print(f"  HLA Timesteps: {config.hla_timesteps}")
    print(f"  Surrogate: Disabled (faster)")
    print(f"  Sensitivity: {config.sensitivity_method} (fast)")


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Run all ViT examples."""
    print("\n" + "="*80)
    print("HiReLC Vision Transformer Compression Examples")
    print("="*80)
    
    example_vit_cifar10()
    example_vit_variants()
    example_vit_cifar100()
    example_vit_quantization_strategies()
    example_vit_integer_only()
    example_vit_fast_compression()
    
    print("\n" + "="*80)
    print("All ViT examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
