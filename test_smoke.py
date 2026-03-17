#!/usr/bin/env python
"""
Smoke test - Verify all imports and basic functionality work
Run with: python test_smoke.py

This is the fastest test to run before running full test suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported"""
    print("\nTesting imports...")
    
    try:
        from hirelc_package.config import (
            CompressionGoal,
            LayerBudget,
            ExperimentConfig,
            KernelConfig,
            LayerConfig
        )
        print("  OK Config module imported")
        
        from hirelc_package.core import (
            AdvancedQuantizer,
            AdvancedPruner,
            SurrogateModelTrainer
        )
        print("  OK Core module imported")
        
        from hirelc_package.utils import (
            ReproducibilityManager,
            ExperimentLogger,
            DataManager
        )
        print("  OK Utils module imported")
        
        return True
    except ImportError as e:
        print(f"  FAIL Import failed: {e}")
        return False


def test_basic_creation():
    """Test basic object creation"""
    print("\nTesting basic object creation...")
    
    try:
        from hirelc_package.config import CompressionGoal, ExperimentConfig
        
        # Create goal
        goal = CompressionGoal(
            target_accuracy_drop=1.0,
            target_compression_ratio=0.25
        )
        print("  OK CompressionGoal created")
        
        # Create config
        config = ExperimentConfig(
            model_name="resnet18",
            dataset="cifar10",
            compression_goal=goal
        )
        print("  OK ExperimentConfig created")
        
        return True
    except Exception as e:
        print(f"  FAIL Object creation failed: {e}")
        return False


def test_torch_available():
    """Test PyTorch is available"""
    print("\nTesting PyTorch availability...")
    
    try:
        import torch
        print(f"  OK PyTorch {torch.__version__} available")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  OK Device: {device}")
        
        return True
    except ImportError:
        print("  FAIL PyTorch not available")
        return False


if __name__ == "__main__":
    print("="*60)
    print("HiReLC SMOKE TEST")
    print("="*60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_torch_available():
        all_passed = False
    
    if not test_basic_creation():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("SMOKE TEST PASSED OK")
        print("="*60 + "\n")
        print("Ready to run full test suite:")
        print("  python run_all_tests.py")
        sys.exit(0)
    else:
        print("SMOKE TEST FAILED FAIL")
        print("="*60 + "\n")
        print("Fix issues above before running full test suite")
        sys.exit(1)

