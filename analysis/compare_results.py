"""
Compare Kineto and NSight Compute profiling results.

Generates a comprehensive comparison report analyzing the differences
between PyTorch Kineto and NVIDIA NSight Compute profiling outputs.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from analysis.utils import (
    load_kineto_stats,
    load_nsight_stats,
    load_config,
    get_model_info,
    extract_top_kernels,
    aggregate_kernel_families,
    create_comparison_table,
    format_bytes,
    format_time,
)


def generate_report_header(configs: List[str]) -> str:
    """Generate report header."""
    header = []
    header.append("=" * 80)
    header.append("KINETO VS NVIDIA NSIGHT COMPUTE - PROFILING COMPARISON REPORT")
    header.append("=" * 80)
    header.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.append(f"Configurations analyzed: {len(configs)}")
    for config in configs:
        header.append(f"  - {config}")
    header.append("\n" + "=" * 80)
    header.append("")
    return "\n".join(header)


def generate_config_comparison(config_name: str, kineto_stats: Dict, nsight_stats: Dict, config: Dict) -> str:
    """Generate comparison for a single configuration."""
    sections = []

    sections.append(f"\n{'=' * 80}")
    sections.append(f"CONFIGURATION: {config_name.upper()}")
    sections.append(f"{'=' * 80}\n")

    # Model configuration
    sections.append("Model Configuration:")
    sections.append("-" * 80)
    model_info = get_model_info(config)
    sections.append(f"  Batch size:        {model_info['batch_size']}")
    sections.append(f"  Sequence length:   {model_info['seq_len']}")
    sections.append(f"  Hidden dimension:  {model_info['hidden_dim']}")
    sections.append(f"  Number of heads:   {model_info['num_heads']}")
    sections.append(f"  Input tensor size: {model_info['input_size_mb']:.2f} MB")
    sections.append(f"  Attention matrix:  {model_info['attn_matrix_size_mb']:.2f} MB")
    sections.append(f"  Total working set: {model_info['total_working_set_mb']:.2f} MB")
    sections.append("")

    # Kineto Results
    sections.append("Kineto Profiling Results:")
    sections.append("-" * 80)

    if kineto_stats:
        baseline_time = kineto_stats.get('baseline_time_ms_per_iter', 0)
        tflops = kineto_stats.get('estimated_tflops', 0)
        mem_stats = kineto_stats.get('memory_stats', {})

        sections.append(f"  Execution time (per iter):  {baseline_time:.4f} ms")
        sections.append(f"  Estimated performance:      {tflops:.2f} TFLOP/s")
        sections.append(f"  Memory allocated:           {mem_stats.get('allocated_gb', 0):.4f} GB")
        sections.append(f"  Memory reserved:            {mem_stats.get('reserved_gb', 0):.4f} GB")
        sections.append(f"  Peak memory:                {mem_stats.get('max_allocated_gb', 0):.4f} GB")

        # Top kernels
        sections.append("\n  Top 10 GPU Kernels (by CUDA time):")
        top_kernels = extract_top_kernels(kineto_stats, top_k=10)

        if top_kernels:
            kernel_table_rows = []
            for i, kernel in enumerate(top_kernels, 1):
                name = kernel.get('name', '')[:50]  # Truncate long names
                count = kernel.get('count', 0)
                total_time = kernel.get('cuda_time_total_us', 0)
                avg_time = total_time / count if count > 0 else 0

                kernel_table_rows.append([
                    str(i),
                    name,
                    str(count),
                    f"{total_time/1000:.2f} ms",
                    f"{avg_time:.2f} μs"
                ])

            table = create_comparison_table(
                kernel_table_rows,
                ["#", "Kernel Name", "Calls", "Total Time", "Avg Time"]
            )
            sections.append("\n" + "\n".join("  " + line for line in table.split("\n")))

        # Kernel families
        sections.append("\n  Kernel Time by Family:")
        if top_kernels:
            families = aggregate_kernel_families(kineto_stats.get('kernel_stats', []))

            family_rows = []
            for family, data in sorted(families.items(), key=lambda x: x[1]['total_time_us'], reverse=True):
                time_ms = data['total_time_us'] / 1000
                count = data['count']
                family_rows.append([
                    family,
                    str(count),
                    f"{time_ms:.2f} ms",
                    f"{(time_ms / baseline_time * 100):.1f}%"
                ])

            if family_rows:
                fam_table = create_comparison_table(
                    family_rows,
                    ["Family", "Calls", "Total Time", "% of Total"]
                )
                sections.append("\n" + "\n".join("  " + line for line in fam_table.split("\n")))

    else:
        sections.append("  ❌ No Kineto profiling data available")

    sections.append("")

    # NSight Compute Results
    sections.append("NVIDIA NSight Compute Results:")
    sections.append("-" * 80)

    if nsight_stats and nsight_stats.get('success'):
        sections.append(f"  ✓ Profiling completed successfully")
        sections.append(f"  Output file: {nsight_stats.get('ncu_output', 'N/A')}")
        sections.append("")
        sections.append("  To view detailed metrics:")
        sections.append(f"    ncu-ui {nsight_stats.get('ncu_output', '')}")
        sections.append("")
        sections.append("  Key hardware metrics available in NSight Compute:")
        sections.append("    - SM utilization and occupancy")
        sections.append("    - Memory bandwidth utilization (% of peak)")
        sections.append("    - L1/L2 cache hit rates")
        sections.append("    - Warp stall reasons")
        sections.append("    - Instruction mix (FP32/FP64/INT ops)")
        sections.append("    - Memory transaction efficiency")
        sections.append("    - Roofline analysis")
    else:
        if nsight_stats:
            sections.append(f"  ❌ NSight profiling failed or incomplete")
            sections.append(f"  Return code: {nsight_stats.get('return_code', 'N/A')}")
        else:
            sections.append("  ❌ No NSight Compute profiling data available")

    sections.append("")

    return "\n".join(sections)


def generate_architectural_comparison() -> str:
    """Generate high-level architectural comparison section."""
    sections = []

    sections.append("\n" + "=" * 80)
    sections.append("ARCHITECTURAL COMPARISON: KINETO VS NSIGHT COMPUTE")
    sections.append("=" * 80)
    sections.append("")

    sections.append("See ARCHITECTURE_COMPARISON.md for detailed analysis.")
    sections.append("")

    # Quick comparison table
    comparison_data = [
        ["Design Philosophy", "Application-level ML profiling", "Hardware-level GPU profiling"],
        ["Integration", "Native PyTorch integration", "External CLI tool"],
        ["Overhead", "~5-10% runtime", "10-200x slowdown"],
        ["Granularity", "PyTorch operators + kernels", "Individual kernel analysis"],
        ["Hardware Metrics", "Limited (time, memory)", "100+ metrics (SM, cache, etc.)"],
        ["Context Awareness", "Full Python → CUDA stack", "Kernel-level only"],
        ["Visualization", "TensorBoard, Chrome Trace", "NSight Compute GUI"],
        ["Use Case", "Initial profiling, bottlenecks", "Kernel optimization"],
        ["Multi-pass", "Single pass", "Multi-pass for metrics"],
        ["Production Ready", "Yes (low overhead)", "No (high overhead)"],
    ]

    table = create_comparison_table(
        comparison_data,
        ["Aspect", "Kineto", "NSight Compute"]
    )

    sections.append(table)
    sections.append("")

    return "\n".join(sections)


def generate_recommendations(config_results: Dict[str, Dict]) -> str:
    """Generate recommendations based on results."""
    sections = []

    sections.append("\n" + "=" * 80)
    sections.append("RECOMMENDATIONS")
    sections.append("=" * 80)
    sections.append("")

    sections.append("Based on the profiling results:")
    sections.append("")

    sections.append("1. WHEN TO USE KINETO:")
    sections.append("   ✓ Initial profiling of your training/inference pipeline")
    sections.append("   ✓ Identifying slow PyTorch operators")
    sections.append("   ✓ Finding CPU-GPU synchronization issues")
    sections.append("   ✓ Debugging memory allocation patterns")
    sections.append("   ✓ Production monitoring (low overhead)")
    sections.append("   ✓ Multi-GPU distributed training analysis")
    sections.append("")

    sections.append("2. WHEN TO USE NSIGHT COMPUTE:")
    sections.append("   ✓ Deep-dive into specific slow kernels")
    sections.append("   ✓ Optimizing custom CUDA kernels")
    sections.append("   ✓ Analyzing hardware utilization (SM, memory bandwidth)")
    sections.append("   ✓ Diagnosing cache efficiency issues")
    sections.append("   ✓ Understanding warp-level behavior")
    sections.append("   ✓ Comparing against theoretical roofline model")
    sections.append("")

    sections.append("3. RECOMMENDED WORKFLOW:")
    sections.append("   Step 1: Profile with Kineto to identify bottlenecks")
    sections.append("   Step 2: Focus on top 5-10 slowest operators/kernels")
    sections.append("   Step 3: Use NSight Compute for deep analysis of those kernels")
    sections.append("   Step 4: Optimize based on hardware metrics")
    sections.append("   Step 5: Validate with Kineto to confirm overall speedup")
    sections.append("")

    # Specific recommendations based on results
    for config_name, results in config_results.items():
        kineto_stats = results.get('kineto_stats')
        if kineto_stats:
            tflops = kineto_stats.get('estimated_tflops', 0)

            sections.append(f"4. SPECIFIC RECOMMENDATIONS FOR {config_name.upper()}:")

            # Analyze performance
            if tflops > 0:
                sections.append(f"   Current performance: {tflops:.2f} TFLOP/s")

                # H200 theoretical peak for FP32: ~67 TFLOPS (non-tensor core)
                # With Tensor Cores (TF32): ~989 TFLOPS
                theoretical_peak_fp32 = 67.0
                utilization = (tflops / theoretical_peak_fp32) * 100

                sections.append(f"   Estimated FP32 utilization: {utilization:.1f}% of peak")

                if utilization < 50:
                    sections.append("   ⚠ Low compute utilization - investigate with NSight:")
                    sections.append("     - Check memory bandwidth utilization")
                    sections.append("     - Look for memory-bound kernels")
                    sections.append("     - Consider using Tensor Cores (FP16/BF16)")
                elif utilization < 80:
                    sections.append("   ✓ Moderate utilization - room for improvement:")
                    sections.append("     - Check kernel occupancy in NSight")
                    sections.append("     - Look for warp stalls")
                else:
                    sections.append("   ✓ Good compute utilization")

            sections.append("")

    return "\n".join(sections)


def generate_summary(config_results: Dict[str, Dict]) -> str:
    """Generate executive summary."""
    sections = []

    sections.append("\n" + "=" * 80)
    sections.append("EXECUTIVE SUMMARY")
    sections.append("=" * 80)
    sections.append("")

    # Count successful profiling runs
    kineto_success = sum(1 for r in config_results.values() if r.get('kineto_stats'))
    nsight_success = sum(1 for r in config_results.values()
                        if r.get('nsight_stats') and r['nsight_stats'].get('success'))

    sections.append(f"Configurations profiled: {len(config_results)}")
    sections.append(f"Successful Kineto runs: {kineto_success}/{len(config_results)}")
    sections.append(f"Successful NSight runs: {nsight_success}/{len(config_results)}")
    sections.append("")

    # Performance summary
    sections.append("Performance Summary:")
    sections.append("")

    perf_rows = []
    for config_name, results in sorted(config_results.items()):
        kineto_stats = results.get('kineto_stats')
        if kineto_stats:
            time_ms = kineto_stats.get('baseline_time_ms_per_iter', 0)
            tflops = kineto_stats.get('estimated_tflops', 0)
            mem_gb = kineto_stats.get('memory_stats', {}).get('allocated_gb', 0)

            perf_rows.append([
                config_name,
                f"{time_ms:.2f} ms",
                f"{tflops:.2f}",
                f"{mem_gb:.2f} GB"
            ])

    if perf_rows:
        perf_table = create_comparison_table(
            perf_rows,
            ["Config", "Time/Iter", "TFLOP/s", "Memory"]
        )
        sections.append(perf_table)

    sections.append("")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Kineto and NSight Compute profiling results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory containing profiling outputs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison_report.md",
        help="Output report file"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs='+',
        default=["config_a", "config_b"],
        help="Configuration names to analyze"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("Kineto vs NSight Compute - Results Analysis")
    print(f"{'=' * 80}\n")

    # Load all results
    config_results = {}

    for config_name in args.configs:
        print(f"Loading results for {config_name}...")

        kineto_dir = results_dir / f"kineto_{config_name}"
        nsight_dir = results_dir / f"nsight_{config_name}"

        kineto_stats = load_kineto_stats(kineto_dir) if kineto_dir.exists() else None
        nsight_stats = load_nsight_stats(nsight_dir) if nsight_dir.exists() else None

        # Load config
        config_path = Path(f"configs/{config_name}.yaml")
        config = load_config(config_path) if config_path.exists() else None

        config_results[config_name] = {
            'kineto_stats': kineto_stats,
            'nsight_stats': nsight_stats,
            'config': config,
        }

        if kineto_stats:
            print(f"  ✓ Kineto results loaded")
        else:
            print(f"  ✗ Kineto results not found")

        if nsight_stats and nsight_stats.get('success'):
            print(f"  ✓ NSight results loaded")
        else:
            print(f"  ✗ NSight results not found or failed")

    print()

    # Generate report
    print("Generating comparison report...")

    report_sections = []

    # Header
    report_sections.append(generate_report_header(args.configs))

    # Executive summary
    report_sections.append(generate_summary(config_results))

    # Architectural comparison
    report_sections.append(generate_architectural_comparison())

    # Per-config analysis
    for config_name, results in sorted(config_results.items()):
        if results['config']:
            report_sections.append(
                generate_config_comparison(
                    config_name,
                    results['kineto_stats'],
                    results['nsight_stats'],
                    results['config']
                )
            )

    # Recommendations
    report_sections.append(generate_recommendations(config_results))

    # Combine all sections
    full_report = "\n".join(report_sections)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(full_report)

    print(f"\n{'=' * 80}")
    print(f"Report generated successfully!")
    print(f"{'=' * 80}")
    print(f"\nOutput: {output_path}")
    print(f"\nTo view:")
    print(f"  cat {output_path}")
    print(f"  # or")
    print(f"  less {output_path}")
    print()


if __name__ == "__main__":
    main()
