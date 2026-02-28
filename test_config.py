"""
Test suite for configuration module
Run with: python test_config.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hirelc_package.config import (
    CompressionGoal,
    LayerBudget,
    ExperimentConfig,
    KernelConfig,
    LayerConfig
)


def test_compression_goal():
    """Test CompressionGoal configuration"""
    print("Testing CompressionGoal...")
    
    goal = CompressionGoal(
        target_accuracy_drop=1.0,
        target_compression_ratio=0.25,
        min_layer_bits=2,
        max_layer_bits=8,
        alpha=50.0,
        beta=2.0
    )
    
    assert goal.target_accuracy_drop == 1.0
    assert goal.target_compression_ratio == 0.25
    assert goal.alpha == 50.0
    
    # Test serialization
    goal_dict = goal.to_dict()
    assert isinstance(goal_dict, dict)
    assert 'target_accuracy_drop' in goal_dict
    
    print("✓ CompressionGoal: PASSED")


def test_layer_budget():
    """Test LayerBudget configuration"""
    print("Testing LayerBudget...")
    
    budget = LayerBudget(
        block_idx=0,
        target_compression_ratio=0.25,
        max_accuracy_drop=1.0,
        priority=0.8,
        sensitivity=0.7,
        global_min_bits=2,
        global_max_bits=8
    )
    
    assert budget.block_idx == 0
    assert budget.target_compression_ratio == 0.25
    assert budget.priority == 0.8
    assert budget.sensitivity == 0.7
    
    # Test serialization
    budget_dict = budget.to_dict()
    assert isinstance(budget_dict, dict)
    
    print("✓ LayerBudget: PASSED")


def test_kernel_config():
    """Test KernelConfig for individual kernels"""
    print("Testing KernelConfig...")
    
    kc = KernelConfig(
        name='kernel1',
        weight_bits=8,
        act_bits=8,
        quant_type='INT',
        quant_mode='uniform',
        pruning_ratio=0.9,
        shape=(64, 64)
    )
    
    assert kc.name == 'kernel1'
    assert kc.weight_bits == 8
    assert kc.quant_type == 'INT'
    
    # Test compression ratio
    compression = kc.compression_ratio()
    assert 0.0 < compression <= 1.0
    
    # Test serialization
    kc_dict = kc.to_dict()
    assert kc_dict['name'] == 'kernel1'
    
    print("✓ KernelConfig: PASSED")


def test_layer_config():
    """Test LayerConfig for entire layers"""
    print("Testing LayerConfig...")
    
    config = LayerConfig(block_idx=0)
    
    # Add kernel configs
    kc1 = KernelConfig(
        name='kernel1',
        weight_bits=8,
        act_bits=8,
        quant_type='INT',
        quant_mode='uniform',
        pruning_ratio=0.9,
        shape=(64, 64)
    )
    config.kernel1_config = kc1
    config.update_aggregates()
    
    assert config.block_idx == 0
    assert config.kernel1_config is not None
    assert config.avg_weight_bits == 8
    
    # Test compression ratio
    compression = config.compression_ratio()
    assert compression > 0.0
    
    print("✓ LayerConfig: PASSED")


def test_experiment_config():
    """Test full experiment configuration"""
    print("Testing ExperimentConfig...")
    
    compression_goal = CompressionGoal(
        target_accuracy_drop=1.0,
        target_compression_ratio=0.25,
        alpha=50.0,
        beta=2.0
    )
    
    config = ExperimentConfig(
        model_name="resnet18",
        dataset="cifar10",
        num_classes=10,
        compression_goal=compression_goal,
        quantization_type='mixed',
        batch_size=64,
        num_lla_agents=3,
        num_hla_agents=2
    )
    
    assert config.model_name == "resnet18"
    assert config.dataset == "cifar10"
    assert config.num_lla_agents == 3
    assert config.quantization_type == 'mixed'
    
    # Test serialization to JSON
    config_json = config.to_json()
    assert isinstance(config_json, str)
    
    # Test loading from JSON
    config_loaded = ExperimentConfig.from_json(config_json)
    assert config_loaded.model_name == "resnet18"
    assert config_loaded.num_classes == 10
    
    print("✓ ExperimentConfig: PASSED")


def test_config_constraints():
    """Test configuration validation and constraints"""
    print("Testing Config Constraints...")
    
    # Valid config
    goal = CompressionGoal(
        target_accuracy_drop=1.0,
        target_compression_ratio=0.25,
        min_layer_bits=2,
        max_layer_bits=8,
        min_layer_pruning=0.0,
        max_layer_pruning=0.8
    )
    
    assert goal.min_layer_bits < goal.max_layer_bits
    assert goal.min_layer_pruning < goal.max_layer_pruning
    
    # Test alpha, beta are positive
    assert goal.alpha > 0
    assert goal.beta > 0
    
    print("✓ Config Constraints: PASSED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CONFIGURATION MODULE TESTS")
    print("="*60 + "\n")
    
    try:
        test_compression_goal()
        test_layer_budget()
        test_kernel_config()
        test_layer_config()
        test_experiment_config()
        test_config_constraints()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
