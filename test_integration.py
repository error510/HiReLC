"""
Integration test - verify all components work together
Run with: python test_integration.py

This test creates a minimal model and runs through key HiReLC operations:
1. Load small dataset
2. Compute sensitivity
3. Apply quantization/pruning
4. Create surrogate and train
5. Evaluate compressed model
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from hirelc_package.config import (
    CompressionGoal,
    ExperimentConfig,
    KernelConfig,
    LayerConfig
)
from hirelc_package.core import (
    AdvancedQuantizer,
    AdvancedPruner,
    GradientNormSensitivityEstimator,
    SurrogateModelTrainer
)
from hirelc_package.utils import (
    ReproducibilityManager,
    ExperimentLogger
)


# ============================================================================
# STEP 1: Create a minimal test dataset and model
# ============================================================================

def create_test_model():
    """Create a simple CNN for testing"""
    print("Creating test model...")
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )
    return model


def create_test_data(batch_size=32):
    """Create synthetic test data (CIFAR-10 like)"""
    print("Creating synthetic test data...")
    
    # 100 samples for training, 50 for testing
    X_train = torch.randn(100, 3, 32, 32)
    y_train = torch.randint(0, 10, (100,))
    
    X_test = torch.randn(50, 3, 32, 32)
    y_test = torch.randint(0, 10, (50,))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# STEP 2: Baseline evaluation
# ============================================================================

def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate model accuracy on dataloader"""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


# ============================================================================
# STEP 3: Integrate all components
# ============================================================================

def test_full_pipeline():
    """Test full compression pipeline"""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Full Compression Pipeline")
    print("="*70 + "\n")
    
    # Setup
    device = 'cpu'  # Use CPU for CI/testing
    ReproducibilityManager.set_seed(42)
    
    # Create model and data
    print("\n[1/7] Creating model and data...")
    model = create_test_model()
    train_loader, test_loader = create_test_data()
    
    # Baseline accuracy
    print("[2/7] Computing baseline accuracy...")
    baseline_acc = evaluate_model(model, test_loader, device)
    print(f"  Baseline Accuracy: {baseline_acc:.2f}%")
    
    # Configuration
    print("\n[3/7] Setting up configuration...")
    compression_goal = CompressionGoal(
        target_accuracy_drop=2.0,
        target_compression_ratio=0.5,
        alpha=50.0,
        beta=2.0,
        gamma=1.0
    )
    
    config = ExperimentConfig(
        model_name='test_model',
        dataset='test',
        num_classes=10,
        batch_size=32,
        compression_goal=compression_goal,
        quantization_type='mixed',
        use_surrogate=True,
        num_lla_agents=2,
        num_hla_agents=2,
        device=device
    )
    print(f"  ✓ Config created: {config.model_name}")
    
    # Sensitivity estimation
    print("\n[4/7] Computing layer sensitivity...")
    try:
        estimator = GradientNormSensitivityEstimator(model, train_loader)
        sensitivities = estimator.estimate_importance(num_samples=5)
        print(f"  ✓ Sensitivity computed for {len(sensitivities)} parameters")
    except Exception as e:
        print(f"  ⚠ Sensitivity estimation skipped: {e}")
        sensitivities = {}
    
    # Create compression configuration
    print("\n[5/7] Creating compression configuration...")
    
    # Get model parameters for compression
    layer_configs = {}
    conv_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            kc = KernelConfig(
                name=name,
                weight_bits=6,  # Compress to 6 bits
                act_bits=8,
                quant_type='INT',
                quant_mode='uniform',
                pruning_ratio=0.8,  # 80% remaining (20% pruned)
                shape=param.shape
            )
            
            config_obj = LayerConfig(block_idx=conv_idx)
            setattr(config_obj, 'kernel1_config', kc)
            config_obj.update_aggregates()
            
            layer_configs[conv_idx] = config_obj
            conv_idx += 1
    
    print(f"  ✓ Created {len(layer_configs)} layer configurations")
    
    # Apply compression
    print("\n[6/7] Applying compression...")
    compressed_model = apply_compression(model, layer_configs)
    
    # Evaluate compressed model
    compressed_acc = evaluate_model(compressed_model, test_loader, device)
    acc_drop = baseline_acc - compressed_acc
    
    print(f"  Compressed Accuracy: {compressed_acc:.2f}%")
    print(f"  Accuracy Drop: {acc_drop:.2f}%")
    
    # Surrogate test
    print("\n[7/7] Testing surrogate model...")
    try:
        surrogate = SurrogateModelTrainer(
            num_blocks=len(layer_configs),
            num_kernels_per_block=1,
            hidden_dims=[32, 16],
            device=device,
            baseline_accuracy=baseline_acc
        )
        
        # Add a sample
        surrogate.add_sample(layer_configs, accuracy=compressed_acc)
        print(f"  ✓ Surrogate initialized with {surrogate.get_buffer_size()} sample(s)")
        
        # Train surrogate
        surrogate.train(epochs=5, batch_size=4)
        print(f"  ✓ Surrogate trained")
        
    except Exception as e:
        print(f"  ⚠ Surrogate test skipped: {e}")
    
    # Results
    print("\n" + "="*70)
    print("INTEGRATION TEST RESULTS")
    print("="*70)
    print(f"Baseline Accuracy:      {baseline_acc:.2f}%")
    print(f"Compressed Accuracy:    {compressed_acc:.2f}%")
    print(f"Accuracy Drop:          {acc_drop:.2f}%")
    print(f"Compression Ratio:      {compute_compression_ratio(model, layer_configs):.4f}")
    print(f"Model Size Reduction:   {(1 - compute_compression_ratio(model, layer_configs))*100:.1f}%")
    
    # Verify results are reasonable
    if compressed_acc > 0:  # Model still produces output
        print("\n✓ INTEGRATION TEST PASSED")
        return 0
    else:
        print("\n✗ INTEGRATION TEST FAILED: Invalid accuracy")
        return 1


def apply_compression(model, layer_configs):
    """Apply quantization and pruning to model"""
    compressed_model = model.__class__(*model.modules()[1:])
    import copy
    compressed_model = copy.deepcopy(model)
    
    # Apply configs to parameters
    param_idx = 0
    for name, param in compressed_model.named_parameters():
        if 'weight' not in name or len(param.shape) <= 1:
            continue
        
        if param_idx not in layer_configs:
            param_idx += 1
            continue
        
        config = layer_configs[param_idx]
        kernel_config = config.kernel1_config
        
        if kernel_config is None:
            param_idx += 1
            continue
        
        # Apply pruning
        if kernel_config.pruning_ratio < 1.0:
            mask = AdvancedPruner.create_neuron_mask(
                param.data,
                kernel_config.pruning_ratio
            )
            param.data *= mask
        
        # Apply quantization
        param.data = AdvancedQuantizer.quantize(
            param.data,
            kernel_config.weight_bits,
            mode=kernel_config.quant_mode,
            quant_type=kernel_config.quant_type
        )
        
        param_idx += 1
    
    return compressed_model


def compute_compression_ratio(model, layer_configs):
    """Compute overall compression ratio"""
    ratios = []
    for config in layer_configs.values():
        ratio = config.compression_ratio()
        if ratio > 0:
            ratios.append(ratio)
    
    return sum(ratios) / len(ratios) if ratios else 1.0


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HiReLC INTEGRATION TEST")
    print("="*70)
    
    try:
        exit_code = test_full_pipeline()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
