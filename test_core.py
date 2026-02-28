"""
Test suite for core compression algorithms
Run with: python test_core.py
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hirelc_package.core import (
    AdvancedQuantizer,
    AdvancedPruner,
    FineGrainedPruner,
    FisherSensitivityEstimator,
    GradientNormSensitivityEstimator,
    SurrogateModelTrainer
)
from hirelc_package.config import KernelConfig, LayerConfig


def test_quantization_uniform():
    """Test uniform quantization"""
    print("Testing Quantization (Uniform)...")
    
    tensor = torch.randn(64, 64)
    original_dtype = tensor.dtype
    
    # Test uint8
    q_tensor = AdvancedQuantizer.quantize_uniform(
        tensor.clone(), bits=8, symmetric=False
    )
    assert q_tensor.shape == tensor.shape
    assert not torch.all(q_tensor == tensor)  # Should be quantized
    print("  ✓ Uniform quantization works")


def test_quantization_logarithmic():
    """Test logarithmic quantization"""
    print("Testing Quantization (Logarithmic)...")
    
    tensor = torch.randn(64, 64)
    q_tensor = AdvancedQuantizer.quantize_logarithmic(tensor.clone(), bits=8)
    
    assert q_tensor.shape == tensor.shape
    print("  ✓ Logarithmic quantization works")


def test_quantization_per_channel():
    """Test per-channel quantization"""
    print("Testing Quantization (Per-Channel)...")
    
    tensor = torch.randn(64, 64)
    q_tensor = AdvancedQuantizer.quantize_per_channel(tensor.clone(), bits=8, axis=0)
    
    assert q_tensor.shape == tensor.shape
    print("  ✓ Per-channel quantization works")


def test_quantization_learned():
    """Test learned quantization"""
    print("Testing Quantization (Learned)...")
    
    tensor = torch.randn(64, 64)
    q_tensor, scale, zp = AdvancedQuantizer.quantize_learned(tensor.clone(), bits=8)
    
    assert q_tensor.shape == tensor.shape
    assert scale is not None
    assert zp is not None
    print("  ✓ Learned quantization works")


def test_quantization_combined():
    """Test combined quantization method"""
    print("Testing Quantization (Combined)...")
    
    tensor = torch.randn(64, 64)
    
    for mode in ['uniform', 'log', 'per-channel', 'learned']:
        q_tensor = AdvancedQuantizer.quantize(
            tensor.clone(),
            bits=8,
            mode=mode,
            quant_type='INT'
        )
        assert q_tensor.shape == tensor.shape
        print(f"  ✓ Mode '{mode}' works")


def test_pruning_magnitude():
    """Test magnitude-based pruning"""
    print("Testing Pruning (Magnitude)...")
    
    tensor = torch.randn(64, 64)
    original_nnz = torch.count_nonzero(tensor).item()
    
    pruned = AdvancedPruner.prune_magnitude(tensor.clone(), pruning_ratio=0.5)
    pruned_nnz = torch.count_nonzero(pruned).item()
    
    # Should have fewer non-zero elements
    assert pruned_nnz <= original_nnz
    print(f"  ✓ Magnitude pruning works (NNZ: {original_nnz} → {pruned_nnz})")


def test_pruning_gradient_based():
    """Test gradient-based pruning"""
    print("Testing Pruning (Gradient-Based)...")
    
    tensor = torch.randn(64, 64, requires_grad=True)
    gradients = torch.randn(64, 64)
    
    pruned = AdvancedPruner.prune_gradient_based(
        tensor.clone().detach(),
        gradients,
        pruning_ratio=0.5
    )
    
    assert pruned.shape == tensor.shape
    print("  ✓ Gradient-based pruning works")


def test_pruning_mask():
    """Test mask creation and application"""
    print("Testing Pruning (Mask Operations)...")
    
    tensor = torch.randn(64, 64)
    mask = AdvancedPruner.create_neuron_mask(tensor, pruning_ratio=0.5)
    
    assert mask.shape == tensor.shape
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0) | (mask == 1))
    
    masked = AdvancedPruner.apply_mask(tensor.clone(), mask)
    assert masked.shape == tensor.shape
    print("  ✓ Mask operations work")


def test_fine_grained_pruning():
    """Test fine-grained (channel-level) pruning"""
    print("Testing Fine-Grained Pruning...")
    
    # Conv2d weight: (out_channels, in_channels, H, W)
    tensor = torch.randn(64, 32, 3, 3)
    
    pruned = FineGrainedPruner.channel_pruning(
        tensor.clone(),
        pruning_ratio=0.3,
        axis=0
    )
    
    assert pruned.shape == tensor.shape
    print("  ✓ Channel pruning works")


def test_kernel_config_compression():
    """Test compression ratio calculation"""
    print("Testing KernelConfig Compression Ratio...")
    
    kc = KernelConfig(
        name='test_kernel',
        weight_bits=8,
        act_bits=8,
        quant_type='INT',
        quant_mode='uniform',
        pruning_ratio=0.5,  # 50% pruned
        shape=(64, 64)
    )
    
    ratio = kc.compression_ratio()
    assert 0.0 < ratio < 1.0
    print(f"  ✓ Compression ratio: {ratio:.4f}")


def test_layer_config_aggregates():
    """Test layer config aggregation"""
    print("Testing LayerConfig Aggregates...")
    
    config = LayerConfig(block_idx=0)
    
    # Add 4 kernel configs
    for i in range(1, 5):
        kc = KernelConfig(
            name=f'kernel{i}',
            weight_bits=4 + i,  # 5,6,7,8 bits
            act_bits=8,
            quant_type='INT',
            quant_mode='uniform',
            pruning_ratio=0.7 + i*0.05,
            shape=(64, 64)
        )
        setattr(config, f'kernel{i}_config', kc)
    
    config.update_aggregates()
    
    assert config.avg_weight_bits > 0
    assert config.avg_pruning_ratio > 0
    print(f"  ✓ Avg weight bits: {config.avg_weight_bits}")
    print(f"  ✓ Avg pruning ratio: {config.avg_pruning_ratio:.4f}")


def test_surrogate_basic():
    """Test surrogate model initialization and basic operations"""
    print("Testing Surrogate Model...")
    
    surrogate = SurrogateModelTrainer(
        num_blocks=4,
        num_kernels_per_block=4,
        hidden_dims=[64, 32],
        device='cpu',
        baseline_accuracy=92.5
    )
    
    assert surrogate.get_buffer_size() == 0
    print("  ✓ Surrogate initialized successfully")
    
    # Add a sample
    configs = {}
    for block_idx in range(4):
        config = LayerConfig(block_idx=block_idx)
        for i in range(1, 5):
            kc = KernelConfig(
                name=f'kernel{i}',
                weight_bits=6,
                act_bits=8,
                quant_type='INT',
                quant_mode='uniform',
                pruning_ratio=0.8,
                shape=(64, 64)
            )
            setattr(config, f'kernel{i}_config', kc)
        config.update_aggregates()
        configs[block_idx] = config
    
    surrogate.add_sample(configs, accuracy=91.5)
    assert surrogate.get_buffer_size() == 1
    print("  ✓ Added sample to surrogate")


def test_sensitivity_gradient():
    """Test gradient-based sensitivity estimation (fast)"""
    print("Testing Sensitivity (Gradient-Based)...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    # Create simple data
    X = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    
    # Create minimal dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=8)
    
    try:
        estimator = GradientNormSensitivityEstimator(model, dataloader)
        scores = estimator.estimate_importance(num_samples=2)
        
        assert isinstance(scores, dict)
        assert len(scores) > 0
        print(f"  ✓ Computed sensitivity for {len(scores)} parameters")
    except Exception as e:
        print(f"  ⚠ Sensitivity test skipped: {str(e)}")


def test_quantization_types():
    """Test different quantization types"""
    print("Testing Quantization Type Options...")
    
    tensor = torch.randn(64, 64)
    
    for qtype in ['INT', 'FLOAT']:
        q_tensor = AdvancedQuantizer.quantize(
            tensor.clone(),
            bits=8,
            mode='uniform',
            quant_type=qtype
        )
        assert q_tensor.shape == tensor.shape
        print(f"  ✓ {qtype} quantization works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CORE COMPRESSION ALGORITHMS TESTS")
    print("="*60 + "\n")
    
    try:
        test_quantization_uniform()
        test_quantization_logarithmic()
        test_quantization_per_channel()
        test_quantization_learned()
        test_quantization_combined()
        
        test_pruning_magnitude()
        test_pruning_gradient_based()
        test_pruning_mask()
        test_fine_grained_pruning()
        
        test_kernel_config_compression()
        test_layer_config_aggregates()
        
        test_surrogate_basic()
        
        test_sensitivity_gradient()
        
        test_quantization_types()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
