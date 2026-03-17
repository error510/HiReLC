# HiReLC Test Suite

Comprehensive test suite for validating HiReLC package functionality.

## Running Tests

### Quick Start - Run All Tests

```bash
python run_all_tests.py
```

This will:
1. Run all test modules in sequence
2. Display results for each test
3. Generate a summary report

### Run Individual Test Modules

```bash
# Test configuration classes
python test_config.py

# Test core compression algorithms  
python test_core.py

# Test utility functions
python test_utils.py

# Test integration of all components
python test_integration.py
```

## Test Coverage

### Configuration Tests (`test_config.py`)
- ✓ CompressionGoal creation and serialization
- ✓ LayerBudget constraints and aggregation
- ✓ KernelConfig compression ratio calculation
- ✓ LayerConfig with multiple kernels
- ✓ ExperimentConfig JSON serialization
- ✓ Configuration validation and constraints

**What it validates**: All configuration classes work correctly and can be serialized/deserialized

### Core Algorithm Tests (`test_core.py`)

#### Quantization Methods
- ✓ Uniform quantization (symmetric and asymmetric)
- ✓ Logarithmic quantization (for skewed distributions)
- ✓ Per-channel quantization (different scales per output channel)
- ✓ Learned quantization (trainable parameters)
- ✓ Combined quantization with mode selection
- ✓ INT and FLOAT quantization types

#### Pruning Methods
- ✓ Magnitude-based pruning
- ✓ Gradient-based pruning
- ✓ Mask creation and application
- ✓ Channel-level (fine-grained) pruning
- ✓ Sparsity preservation

#### Sensitivity Estimation
- ✓ Gradient-norm based sensitivity (fast method)

#### Surrogate Model
- ✓ Surrogate model initialization
- ✓ Adding training samples
- ✓ Buffer management

**What it validates**: All compression algorithms produce correct outputs and sparsity patterns

### Utility Tests (`test_utils.py`)

#### Reproducibility
- ✓ Seeding reproducibility (PyTorch, NumPy, random)
- ✓ Device detection (CUDA vs CPU)

#### Logging
- ✓ Logger initialization
- ✓ Multi-level logging (INFO, WARNING, SUCCESS, ERROR)
- ✓ Metric tracking and storage
- ✓ Checkpoint saving and restoration
- ✓ Configuration logging
- ✓ Cycle result logging
- ✓ Surrogate prediction logging

#### Data Loading
- ✓ CIFAR-10 dataset loading
- ✓ CIFAR-100 dataset loading
- ✓ DataLoader creation and batching

**What it validates**: Infrastructure components work correctly

### Integration Test (`test_integration.py`)

End-to-end pipeline test that:
1. Creates a small CNN model (~50K parameters)
2. Constructs synthetic 10-class image data (100 training, 50 test samples)
3. Computes baseline accuracy
4. Applies quantization (to 6 bits) and pruning (20% sparsity)
5. Evaluates compressed model
6. Tests surrogate model training
7. Computes compression ratio and accuracy drop

**What it validates**: All components work together in a realistic workflow

## Test Execution Flow

```
run_all_tests.py
├── test_config.py
│   ├── CompressionGoal
│   ├── LayerBudget
│   ├── KernelConfig
│   ├── LayerConfig
│   ├── ExperimentConfig
│   └── Config Constraints
├── test_core.py
│   ├── Quantization (5 methods)
│   ├── Pruning (4 methods)
│   ├── Sensitivity Estimation
│   ├── Surrogate Model
│   └── Quantization Types
├── test_utils.py
│   ├── Reproducibility
│   ├── Device Detection
│   ├── Logger (7 tests)
│   └── Data Manager (2 datasets)
└── test_integration.py
    └── Full Pipeline (7 steps)
```

## Understanding Test Output

### Successful Test Output
```
============================================================
CONFIGURATION MODULE TESTS
============================================================

Testing CompressionGoal...
✓ CompressionGoal: PASSED
Testing LayerBudget...
✓ LayerBudget: PASSED
...

============================================================
ALL TESTS PASSED ✓
============================================================
```

### Failed Test Output
```
Testing Quantization (Per-Channel)...
❌ TEST FAILED: <error message>

Traceback:
  File "test_core.py", line 50, in test_quantization_per_channel
    assert q_tensor.shape == tensor.shape
AssertionError
```

## Conditional Test Skipping

Some tests may be skipped with warnings if:
- Dataset downloads fail (CIFAR-10/100)
- GPU/CUDA not available
- Optional dependencies missing

Example:
```
Testing DataManager (CIFAR-10)...
⚠ CIFAR-10 test skipped: Connection timeout
```

These are non-critical and don't cause test failure.

## Test Performance

Typical execution times:

| Test Module | Time | Notes |
|-------------|------|-------|
| test_config.py | <1s | Fast - no GPU needed |
| test_core.py | 3-5s | Quantization/pruning on CPU |
| test_utils.py | 5-10s | Includes data loading, disk I/O |
| test_integration.py | 10-15s | Full pipeline, model training |
| **Total** | **~20-30s** | All tests together |

Times may vary depending on:
- CPU/GPU speed
- Available memory
- Network connectivity (for datasets)
- Disk I/O speed

## Debugging Failed Tests

### 1. Run Individual Test Module
```bash
python test_core.py  # Verbose output for specific failures
```

### 2. Add Debug Prints
Edit the test file and add:
```python
import pdb; pdb.set_trace()  # Stops at breakpoint
```

### 3. Check Dependencies
```bash
pip list | grep torch
pip list | grep timm
# Verify all required packages installed
```

### 4. Check GPU (if applicable)
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Adding New Tests

### Template for New Test
```python
def test_new_feature():
    """Test description"""
    print("Testing New Feature...")
    
    # Setup
    obj = SomeClass()
    
    # Test
    result = obj.some_method()
    
    # Verify
    assert result is not None
    assert result.property == expected_value
    
    print("  ✓ New feature works")
```

### Naming Convention
- Test files: `test_<module>.py`
- Test functions: `test_<feature_name>()`
- Assertions: Use `assert` statements

### Adding to Test Suite
1. Create test function following template
2. Add to appropriate test file or create new file
3. Update `run_all_tests.py` to include new file
4. Run `python run_all_tests.py` to verify

## Continuous Integration

To integrate with CI/CD:

```bash
# GitHub Actions example
- name: Run Tests
  run: python run_all_tests.py
  
- name: Check Exit Code
  if: failure()
  run: exit 1
```

## Requirements

Basic requirements for running tests:
- Python 3.8+
- PyTorch 1.13+
- timm (for model loading)
- NumPy, Pandas, Matplotlib

See `../requirements.txt` for full list.

## Troubleshooting

### "ModuleNotFoundError: No module named 'hirelc_package'"
```bash
# Run tests from parent directory (d:\hirelc\)
cd ..
python run_all_tests.py
```

### "CUDA out of memory"
Tests automatically fallback to CPU if CUDA unavailable.

### "Connection timeout downloading CIFAR"
Optional - these tests skip gracefully. Core functionality tests still pass.

## Test Maintenance

- **Review Frequency**: After algorithm changes
- **Update**: When adding new features
- **Extend**: When fixing bugs (add regression test)

## Questions?

See [../README.md](../README.md) for algorithm explanation and usage guide.
