#!/usr/bin/env python
"""
Master test runner for HiReLC package
Run with: python run_all_tests.py

This script runs all test modules and provides a comprehensive summary.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_test_module(test_file):
    """Run a single test module and return result"""
    print(f"\n{'='*70}")
    print(f"Running: {test_file}")
    print('='*70)
    
    result = subprocess.run([sys.executable, str(test_file)], capture_output=True, text=True)
    
    return {
        'file': test_file.name,
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'passed': result.returncode == 0
    }


def main():
    """Run all tests and generate report"""
    test_dir = Path(__file__).parent
    
    # Find all test files (smoke test first, then others)
    test_files = sorted([
        test_dir / 'test_smoke.py',
        test_dir / 'test_config.py',
        test_dir / 'test_core.py',
        test_dir / 'test_utils.py',
        test_dir / 'test_integration.py'
    ])
    
    print("\n" + "="*70)
    print("HiReLC PACKAGE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Directory: {test_dir}")
    print(f"Number of Test Files: {len(test_files)}")
    
    # Run all tests
    results = []
    for test_file in test_files:
        if test_file.exists():
            result = run_test_module(test_file)
            results.append(result)
            
            # Print output
            if result['stdout']:
                print(result['stdout'])
            if result['stderr']:
                print("STDERR:", result['stderr'])
        else:
            print(f"⚠ Test file not found: {test_file}")
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"\nTotal Tests Run: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    
    print("\nDetailed Results:")
    for result in results:
        status = "✓ PASSED" if result['passed'] else "✗ FAILED"
        print(f"  {result['file']:<25} {status}")
    
    print(f"\nTest End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    if passed_count == total_count:
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "="*70)
        print(f"SOME TESTS FAILED ({total_count - passed_count}/{total_count})")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
