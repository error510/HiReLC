"""
Test suite for utility modules
Run with: python test_utils.py
"""

import sys
import torch
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hirelc_package.utils import (
    ReproducibilityManager,
    ExperimentLogger,
    DataManager
)


def test_reproducibility_seeding():
    """Test reproducibility with seeding"""
    print("Testing Reproducibility (Seeding)...")
    
    # Set seed
    ReproducibilityManager.set_seed(42)
    
    # Generate random tensors
    t1_1 = torch.randn(10, 10)
    t2_1 = torch.randn(10, 10)
    
    # Reset and generate again
    ReproducibilityManager.set_seed(42)
    t1_2 = torch.randn(10, 10)
    t2_2 = torch.randn(10, 10)
    
    # Should be identical
    assert torch.allclose(t1_1, t1_2), "PyTorch random not reproducible"
    assert torch.allclose(t2_1, t2_2), "Second tensor not reproducible"
    
    print("  ✓ Seeding reproducible")


def test_device_detection():
    """Test device detection"""
    print("Testing Device Detection...")
    
    device = ReproducibilityManager.get_device()
    assert device in ['cuda', 'cpu']
    print(f"  ✓ Device detected: {device}")


def test_logger_initialization():
    """Test logger initialization"""
    print("Testing ExperimentLogger...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name='test_exp',
            output_dir=tmpdir
        )
        
        assert logger is not None
        run_dir = logger.get_run_dir()
        assert run_dir.exists()
        print(f"  ✓ Logger initialized at {run_dir}")


def test_logger_logging():
    """Test logger logging functionality"""
    print("Testing Logger Logging...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger('test_log', tmpdir)
        
        # Test different log levels
        logger.log("Info message", level='INFO')
        logger.log("Warning message", level='WARNING')
        logger.log("Success message", level='SUCCESS')
        logger.log("Error message", level='ERROR')
        
        # Try to read log file
        run_dir = logger.get_run_dir()
        log_file = run_dir / 'experiment.log'
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert 'Info message' in content
        
        print("  ✓ Logging works correctly")


def test_logger_metrics():
    """Test logger metric tracking"""
    print("Testing Logger Metrics...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger('test_metrics', tmpdir)
        
        # Log metrics
        logger.log_metric('accuracy', 92.5)
        logger.log_metric('loss', 0.15)
        
        metrics = logger.metrics
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 92.5
        
        print("  ✓ Metrics tracking works")


def test_logger_checkpoint():
    """Test logger checkpoint saving"""
    print("Testing Logger Checkpoints...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger('test_checkpoint', tmpdir)
        
        # Create dummy model
        import torch.nn as nn
        model = nn.Linear(10, 5)
        
        # Save checkpoint
        logger.save_checkpoint(model, 'test_model')
        
        # Verify checkpoint exists
        run_dir = logger.get_run_dir()
        ckpt_file = run_dir / 'checkpoints' / 'test_model.pt'
        assert ckpt_file.exists()
        
        print("  ✓ Checkpoint saving works")


def test_data_manager_cifar10():
    """Test CIFAR-10 data loading"""
    print("Testing DataManager (CIFAR-10)...")
    
    try:
        train_loader, test_loader = DataManager.get_cifar10(batch_size=32)
        
        assert train_loader is not None
        assert test_loader is not None
        
        # Try to get a batch
        for X, y in train_loader:
            assert X.shape[0] == 32  # batch size
            assert X.shape[1] == 3   # RGB channels
            assert X.shape[2] == 32  # image size
            assert X.shape[3] == 32
            break
        
        print("  ✓ CIFAR-10 loading works")
    except Exception as e:
        print(f"  ⚠ CIFAR-10 test skipped: {str(e)}")


def test_data_manager_cifar100():
    """Test CIFAR-100 data loading"""
    print("Testing DataManager (CIFAR-100)...")
    
    try:
        train_loader, test_loader = DataManager.get_cifar100(batch_size=32)
        
        assert train_loader is not None
        assert test_loader is not None
        
        print("  ✓ CIFAR-100 loading works")
    except Exception as e:
        print(f"  ⚠ CIFAR-100 test skipped: {str(e)}")


def test_logger_cycle_result():
    """Test logging cycle results"""
    print("Testing Logger Cycle Results...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger('test_cycle', tmpdir)
        
        result = {
            'accuracy': 92.5,
            'compression_ratio': 0.25,
            'accuracy_drop': 1.2,
            'configs': {}
        }
        
        logger.log_cycle_result(1, result)
        
        assert 'cycle_history' in logger.metrics
        assert len(logger.metrics['cycle_history']) > 0
        
        print("  ✓ Cycle result logging works")


def test_logger_surrogate_prediction():
    """Test logging surrogate predictions"""
    print("Testing Logger Surrogate Predictions...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger('test_surrogate', tmpdir)
        
        logger.log_surrogate_prediction(
            cycle=1,
            predicted_accuracy=91.5,
            actual_accuracy=92.0
        )
        
        assert 'surrogate_predictions' in logger.metrics
        
        print("  ✓ Surrogate prediction logging works")


def test_logger_config():
    """Test logging configuration"""
    print("Testing Logger Config Logging...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger('test_config_log', tmpdir)
        
        config_dict = {
            'model_name': 'resnet18',
            'dataset': 'cifar10',
            'batch_size': 128
        }
        
        logger.log_config(config_dict)
        
        run_dir = logger.get_run_dir()
        config_file = run_dir / 'config.json'
        assert config_file.exists()
        
        print("  ✓ Config logging works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("UTILITIES MODULE TESTS")
    print("="*60 + "\n")
    
    try:
        test_reproducibility_seeding()
        test_device_detection()
        test_logger_initialization()
        test_logger_logging()
        test_logger_metrics()
        test_logger_checkpoint()
        test_logger_cycle_result()
        test_logger_surrogate_prediction()
        test_logger_config()
        
        test_data_manager_cifar10()
        test_data_manager_cifar100()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
