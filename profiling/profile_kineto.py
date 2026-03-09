"""
Kineto Profiler Script

Profiles attention benchmark using PyTorch's Kineto profiler.
Captures GPU kernel traces, memory usage, and stack traces.
"""

import argparse
import yaml
import torch
import time
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.attention_benchmark import create_benchmark_from_config


def profile_with_kineto(benchmark, config, output_dir):
    """
    Profile benchmark using PyTorch Kineto profiler.

    Args:
        benchmark: AttentionBenchmark instance
        config: Configuration dictionary
        output_dir: Directory to save results
    """
    profiling_config = config['profiling']
    warmup_iters = profiling_config['warmup_iterations']
    profile_iters = profiling_config['profile_iterations']

    print(f"\n{'='*70}")
    print(f"KINETO PROFILING - {config['name']}")
    print(f"{'='*70}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Profile iterations: {profile_iters}")
    print(f"Output directory: {output_dir}")

    # Warmup
    print("\nWarming up...")
    for i in range(warmup_iters):
        benchmark.benchmark_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if (i + 1) % 5 == 0:
            print(f"  Warmup {i+1}/{warmup_iters}")

    # Clear memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print("\nStarting Kineto profiling...")

    # Configure profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Profile
    with torch.profiler.profile(
        activities=activities,
        record_shapes=profiling_config.get('record_shapes', True),
        profile_memory=profiling_config.get('profile_memory', True),
        with_stack=profiling_config.get('with_stack', True),
        with_flops=True,
        with_modules=True,  # Enable module tracking for attention layers
    ) as prof:
        for i in range(profile_iters):
            with torch.profiler.record_function("attention_iteration"):
                benchmark.benchmark_step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if (i + 1) % 10 == 0:
                print(f"  Iteration {i+1}/{profile_iters}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\nProfiling complete. Saving results...")

    # Save Chrome trace (visualize at chrome://tracing)
    trace_file = output_dir / f"kineto_trace_{config['name']}.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"  ✓ Chrome trace: {trace_file}")

    # Save stack traces
    stacks_file = output_dir / f"kineto_stacks_{config['name']}.txt"
    with open(stacks_file, 'w') as f:
        f.write(prof.key_averages(group_by_stack_n=10).table(
            sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
            row_limit=50
        ))
    print(f"  ✓ Stack traces: {stacks_file}")

    # Print summary tables
    print("\n" + "="*70)
    print("KINETO RESULTS - Top operations by CUDA time:")
    print("="*70)
    if torch.cuda.is_available():
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20
        ))
    else:
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20
        ))

    print("\n" + "="*70)
    print("KINETO RESULTS - Top operations by self time:")
    print("="*70)
    if torch.cuda.is_available():
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=20
        ))
    else:
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=20
        ))

    if profiling_config.get('profile_memory', True) and torch.cuda.is_available():
        print("\n" + "="*70)
        print("KINETO RESULTS - Top operations by memory:")
        print("="*70)
        print(prof.key_averages().table(
            sort_by="self_cuda_memory_usage",
            row_limit=20
        ))

    # Collect structured statistics
    stats = {
        "config_name": config['name'],
        "profiler": "kineto",
        "model_config": config['model'],
        "profiling_config": profiling_config,
        "memory_stats": benchmark.get_memory_stats(),
        "estimated_flops": benchmark.get_flops_estimate(),
    }

    # Parse kernel statistics
    key_avg = prof.key_averages()
    kernel_stats = []

    for evt in key_avg:
        kernel_info = {
            "name": evt.key,
            "count": evt.count,
            "cpu_time_total_us": evt.cpu_time_total,
            "self_cpu_time_total_us": evt.self_cpu_time_total,
        }

        if torch.cuda.is_available():
            kernel_info.update({
                "cuda_time_total_us": evt.cuda_time_total,
                "self_cuda_time_total_us": evt.self_cuda_time_total,
                "cuda_memory_usage": evt.cuda_memory_usage,
                "self_cuda_memory_usage": evt.self_cuda_memory_usage,
            })

        kernel_info.update({
            "cpu_memory_usage": evt.cpu_memory_usage,
            "self_cpu_memory_usage": evt.self_cpu_memory_usage,
        })

        kernel_stats.append(kernel_info)

    # Sort by CUDA time
    sort_key = "cuda_time_total_us" if torch.cuda.is_available() else "cpu_time_total_us"
    stats["kernel_stats"] = sorted(
        kernel_stats,
        key=lambda x: x[sort_key],
        reverse=True
    )[:50]  # Top 50

    # Save structured stats
    stats_file = output_dir / f"kineto_stats_{config['name']}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  ✓ Structured stats: {stats_file}")

    # Measure baseline (without profiling overhead)
    print("\n" + "="*70)
    print("Measuring baseline performance (no profiling overhead)...")
    print("="*70)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(profile_iters):
        benchmark.benchmark_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    baseline_time_ms = (end_time - start_time) / profile_iters * 1000

    # Calculate performance metrics
    flops = benchmark.get_flops_estimate()
    tflops = (flops / (baseline_time_ms / 1000)) / 1e12

    print(f"Baseline time (no profiling): {baseline_time_ms:.4f} ms/iter")
    print(f"Estimated TFLOP/s: {tflops:.2f}")

    stats["baseline_time_ms_per_iter"] = baseline_time_ms
    stats["estimated_tflops"] = tflops

    # Update stats file
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*70}")
    print("KINETO PROFILING COMPLETED!")
    print(f"{'='*70}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Profile with Kineto")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir) / f"kineto_{config['name']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")

    # Create benchmark
    print("\nCreating benchmark...")
    benchmark = create_benchmark_from_config(config)

    # Run profiling
    stats = profile_with_kineto(benchmark, config, output_dir)

    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
