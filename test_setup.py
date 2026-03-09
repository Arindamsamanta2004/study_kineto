#!/usr/bin/env python3
"""
Quick Setup Test Script

Verifies that all dependencies are installed and the environment is configured correctly.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that required packages can be imported."""
    print("Testing Python imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch import failed: {e}")
        return False

    try:
        import yaml
        print(f"  ✓ PyYAML")
    except ImportError as e:
        print(f"  ✗ PyYAML import failed: {e}")
        return False

    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device: {torch.cuda.get_device_name()}")
            print(f"    Capability: {torch.cuda.get_device_capability()}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"  ⚠ CUDA not available (will run on CPU)")
            return True  # Not a failure, just a warning

    except Exception as e:
        print(f"  ✗ CUDA check failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "configs/config_a.yaml",
        "configs/config_b.yaml",
        "benchmarks/vector_add_benchmark.py",
        "profiling/profile_kineto.py",
        "profiling/profile_nsight.py",
        "analysis/compare_profilers.py",
        "run_all_profilers.sh",
        "requirements.txt",
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} not found")
            all_exist = False

    return all_exist


def test_benchmark_import():
    """Test that benchmark module can be imported."""
    print("\nTesting benchmark module...")

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmarks.vector_add_benchmark import VectorAddBenchmark, create_benchmark_from_config
        print(f"  ✓ Benchmark module imports successfully")
        return True
    except Exception as e:
        print(f"  ✗ Benchmark import failed: {e}")
        return False


def test_config_loading():
    """Test that config files can be loaded."""
    print("\nTesting config loading...")

    try:
        import yaml

        configs = ["configs/config_a.yaml", "configs/config_b.yaml"]
        for config_path in configs:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"  ✓ {config_path}: {config['name']}")

        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_benchmark_creation():
    """Test that benchmark can be created from config."""
    print("\nTesting benchmark creation...")

    try:
        import yaml
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmarks.vector_add_benchmark import create_benchmark_from_config

        with open("configs/config_a.yaml") as f:
            config = yaml.safe_load(f)

        benchmark = create_benchmark_from_config(config)
        print(f"  ✓ Benchmark created successfully")
        print(f"    Shape: {benchmark.shape}")
        print(f"    Total elements: {benchmark.total_elements:,}")
        print(f"    Device: {benchmark.device}")

        return True
    except Exception as e:
        print(f"  ✗ Benchmark creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  GPU Profiling Benchmark - Setup Test")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("File Structure", test_file_structure),
        ("Benchmark Import", test_benchmark_import),
        ("Config Loading", test_config_loading),
        ("Benchmark Creation", test_benchmark_creation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} - {name}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("  1. Run a quick benchmark: python3 benchmarks/vector_add_benchmark.py --config configs/config_a.yaml --iterations 10")
        print("  2. Run profiling: ./run_all_profilers.sh")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
