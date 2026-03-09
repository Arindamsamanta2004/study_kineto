"""
Vector Addition Benchmark

Simple vector addition kernel for profiling GPU performance.
Compares Kineto and NVIDIA NSight profiling capabilities.
"""

import torch
import time
import argparse
import yaml
import sys
from pathlib import Path
from typing import Tuple


class VectorAddBenchmark:
    """
    Vector addition benchmark for GPU profiling.

    Performs element-wise addition: C = A + B
    Where A, B, C are tensors shaped according to config.
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device

        # Total number of elements
        self.total_elements = batch_size * seq_len * hidden_dim

        # Shape for tensors (batch, seq_len, hidden)
        self.shape = (batch_size, seq_len, hidden_dim)

        print(f"\nBenchmark Configuration:")
        print(f"  Shape: {self.shape}")
        print(f"  Total elements: {self.total_elements:,}")
        print(f"  Dtype: {dtype}")
        print(f"  Device: {device}")
        print(f"  Memory per tensor: {self.total_elements * self.dtype_size() / 1e9:.4f} GB")

    def dtype_size(self) -> int:
        """Return size in bytes for the dtype."""
        if self.dtype == torch.float32:
            return 4
        elif self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            return 2
        else:
            return 4

    def create_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create random input tensors."""
        a = torch.randn(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )
        b = torch.randn(
            self.shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True
        )
        return a, b

    def vector_add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Perform vector addition: C = A + B"""
        return a + b

    def benchmark_step(self, use_backward: bool = True) -> float:
        """
        Single benchmark iteration with forward (and optionally backward) pass.

        Args:
            use_backward: If True, also compute gradients

        Returns:
            Computed sum of output (for verification)
        """
        # Create tensors
        a, b = self.create_tensors()

        # Forward pass: vector addition
        c = self.vector_add(a, b)

        # Compute a simple scalar for backward pass
        if use_backward:
            loss = c.sum()
            loss.backward()

            return loss.item()
        else:
            return c.sum().item()

    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics."""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        return {}

    def get_bandwidth_estimate(self, time_ms: float) -> float:
        """
        Estimate memory bandwidth usage.

        For C = A + B:
        - Read A: N elements
        - Read B: N elements
        - Write C: N elements
        Total: 3N elements

        Args:
            time_ms: Execution time in milliseconds

        Returns:
            Estimated bandwidth in GB/s
        """
        bytes_per_element = self.dtype_size()
        total_bytes = 3 * self.total_elements * bytes_per_element
        time_s = time_ms / 1000.0
        bandwidth_gbs = (total_bytes / 1e9) / time_s
        return bandwidth_gbs

    def get_flops(self) -> int:
        """Return number of FLOPs for one vector addition."""
        # Each element: 1 addition operation
        return self.total_elements


def create_benchmark_from_config(config: dict) -> VectorAddBenchmark:
    """
    Create benchmark from configuration dictionary.

    Args:
        config: Configuration dict with 'model' key

    Returns:
        VectorAddBenchmark instance
    """
    model_config = config['model']

    # Map dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }

    dtype = dtype_map.get(model_config.get('dtype', 'float32'), torch.float32)

    return VectorAddBenchmark(
        batch_size=model_config['batch_size'],
        seq_len=model_config['seq_len'],
        hidden_dim=model_config['hidden_dim'],
        dtype=dtype,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )


def run_warmup(benchmark: VectorAddBenchmark, iterations: int = 10):
    """Run warmup iterations."""
    print(f"\nWarming up ({iterations} iterations)...")
    for i in range(iterations):
        benchmark.benchmark_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if (i + 1) % 5 == 0:
            print(f"  Warmup {i+1}/{iterations}")
    print("Warmup complete.")


def run_baseline_benchmark(benchmark: VectorAddBenchmark, iterations: int = 100) -> dict:
    """
    Run baseline benchmark without profiling.

    Returns:
        Dictionary with timing statistics
    """
    print(f"\nRunning baseline benchmark ({iterations} iterations)...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []

    for i in range(iterations):
        start = time.perf_counter()
        benchmark.benchmark_step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{iterations}")

    # Calculate statistics
    import statistics
    stats = {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stddev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "total_iterations": iterations,
    }

    # Add bandwidth estimate
    stats["bandwidth_gbs"] = benchmark.get_bandwidth_estimate(stats["mean_ms"])
    stats["flops"] = benchmark.get_flops()
    stats["gflops"] = (benchmark.get_flops() / 1e9) / (stats["mean_ms"] / 1000)

    # Add memory stats
    stats["memory"] = benchmark.get_memory_stats()

    print(f"\nBaseline Results:")
    print(f"  Mean time: {stats['mean_ms']:.4f} ms")
    print(f"  Median time: {stats['median_ms']:.4f} ms")
    print(f"  Min time: {stats['min_ms']:.4f} ms")
    print(f"  Max time: {stats['max_ms']:.4f} ms")
    print(f"  Std dev: {stats['stddev_ms']:.4f} ms")
    print(f"  Bandwidth: {stats['bandwidth_gbs']:.2f} GB/s")
    print(f"  GFLOPS: {stats['gflops']:.2f}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector addition benchmark")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"{'='*60}")
    print(f"Vector Addition Benchmark - {config['name']}")
    print(f"{'='*60}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
    else:
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")

    # Create benchmark
    benchmark = create_benchmark_from_config(config)

    # Warmup
    warmup_iters = config['profiling'].get('warmup_iterations', 10)
    run_warmup(benchmark, warmup_iters)

    # Run benchmark
    stats = run_baseline_benchmark(benchmark, args.iterations)

    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}\n")
