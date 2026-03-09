"""
Profiler Comparison Analysis

Compares results from Kineto and NVIDIA NSight Compute profiling.
Generates comparative analysis and visualizations.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List
import sys


def load_kineto_stats(config_name: str, results_dir: Path) -> Dict:
    """Load Kineto profiling statistics."""
    kineto_dir = results_dir / f"kineto_{config_name}"
    stats_file = kineto_dir / f"kineto_stats_{config_name}.json"

    if not stats_file.exists():
        print(f"WARNING: Kineto stats not found: {stats_file}")
        return None

    with open(stats_file) as f:
        return json.load(f)


def load_nsight_stats(config_name: str, results_dir: Path) -> Dict:
    """Load NSight Compute profiling statistics."""
    nsight_dir = results_dir / f"nsight_{config_name}"
    stats_file = nsight_dir / f"nsight_stats_{config_name}.json"

    if not stats_file.exists():
        print(f"WARNING: NSight stats not found: {stats_file}")
        return None

    with open(stats_file) as f:
        return json.load(f)


def analyze_kineto_results(stats: Dict) -> Dict:
    """Extract key metrics from Kineto results."""
    if not stats:
        return {}

    analysis = {
        "profiler": "Kineto",
        "baseline_time_ms": stats.get("baseline_time_ms_per_iter", None),
        "bandwidth_gbs": stats.get("bandwidth_gbs", None),
        "memory_allocated_gb": stats.get("memory_stats", {}).get("allocated_gb", None),
        "memory_max_gb": stats.get("memory_stats", {}).get("max_allocated_gb", None),
        "total_elements": stats.get("total_elements", None),
    }

    # Extract top kernels
    kernel_stats = stats.get("kernel_stats", [])
    if kernel_stats:
        top_kernel = kernel_stats[0]
        analysis["top_kernel"] = {
            "name": top_kernel.get("name", ""),
            "cuda_time_us": top_kernel.get("cuda_time_total_us", 0),
            "count": top_kernel.get("count", 0),
        }

        # Calculate overhead
        if "cuda_time_total_us" in top_kernel and analysis["baseline_time_ms"]:
            profiled_time_ms = top_kernel["cuda_time_total_us"] / 1000.0
            baseline_ms = analysis["baseline_time_ms"]
            overhead_pct = ((profiled_time_ms - baseline_ms) / baseline_ms) * 100
            analysis["profiling_overhead_pct"] = overhead_pct

    return analysis


def analyze_nsight_results(stats: Dict) -> Dict:
    """Extract key metrics from NSight results."""
    if not stats:
        return {}

    analysis = {
        "profiler": "NSight Compute",
        "success": stats.get("success", False),
        "ncu_report": stats.get("ncu_output", ""),
    }

    return analysis


def compare_profilers(config_name: str, results_dir: Path) -> Dict:
    """Compare Kineto and NSight results for a configuration."""
    print(f"\n{'='*70}")
    print(f"COMPARATIVE ANALYSIS: {config_name}")
    print(f"{'='*70}\n")

    kineto_stats = load_kineto_stats(config_name, results_dir)
    nsight_stats = load_nsight_stats(config_name, results_dir)

    kineto_analysis = analyze_kineto_results(kineto_stats)
    nsight_analysis = analyze_nsight_results(nsight_stats)

    comparison = {
        "config_name": config_name,
        "kineto": kineto_analysis,
        "nsight": nsight_analysis,
    }

    # Print comparison
    print("KINETO PROFILING:")
    print("-" * 70)
    if kineto_analysis:
        for key, value in kineto_analysis.items():
            if key != "top_kernel":
                print(f"  {key:30s}: {value}")
        if "top_kernel" in kineto_analysis:
            print(f"\n  Top kernel:")
            for k, v in kineto_analysis["top_kernel"].items():
                print(f"    {k:28s}: {v}")
    else:
        print("  No Kineto results found")

    print(f"\n{'='*70}\n")

    print("NSIGHT COMPUTE PROFILING:")
    print("-" * 70)
    if nsight_analysis:
        for key, value in nsight_analysis.items():
            print(f"  {key:30s}: {value}")
    else:
        print("  No NSight results found")

    print(f"\n{'='*70}")

    return comparison


def generate_markdown_report(comparisons: List[Dict], output_file: Path):
    """Generate a markdown report comparing profilers."""

    report = []
    report.append("# Kineto vs NVIDIA NSight Compute - Profiling Comparison\n")
    report.append("## Executive Summary\n")
    report.append("This report compares PyTorch's Kineto profiler with NVIDIA NSight Compute (NCU)")
    report.append("for profiling GPU kernels on vector addition benchmarks.\n")

    for comp in comparisons:
        config_name = comp["config_name"]
        report.append(f"\n## Configuration: {config_name}\n")

        kineto = comp.get("kineto", {})
        nsight = comp.get("nsight", {})

        # Kineto section
        report.append("### Kineto Profiler\n")
        if kineto:
            report.append("**Key Metrics:**\n")
            report.append(f"- Baseline time: {kineto.get('baseline_time_ms', 'N/A'):.4f} ms/iter\n")
            report.append(f"- Memory bandwidth: {kineto.get('bandwidth_gbs', 'N/A'):.2f} GB/s\n")
            report.append(f"- Memory allocated: {kineto.get('memory_allocated_gb', 'N/A'):.4f} GB\n")
            if "profiling_overhead_pct" in kineto:
                report.append(f"- Profiling overhead: {kineto['profiling_overhead_pct']:.2f}%\n")

            if "top_kernel" in kineto:
                report.append(f"\n**Top Kernel:**\n")
                report.append(f"- Name: `{kineto['top_kernel'].get('name', '')}`\n")
                report.append(f"- CUDA time: {kineto['top_kernel'].get('cuda_time_us', 0):.2f} μs\n")
                report.append(f"- Call count: {kineto['top_kernel'].get('count', 0)}\n")
        else:
            report.append("*No results available*\n")

        # NSight section
        report.append("\n### NVIDIA NSight Compute\n")
        if nsight:
            report.append(f"- Status: {'✓ Success' if nsight.get('success') else '✗ Failed'}\n")
            report.append(f"- Report: `{nsight.get('ncu_report', 'N/A')}`\n")
        else:
            report.append("*No results available*\n")

    # Add comparison table
    report.append("\n## Profiler Comparison Matrix\n")
    report.append("| Feature | Kineto | NVIDIA NSight Compute |\n")
    report.append("|---------|--------|-----------------------|\n")
    report.append("| **Ease of Use** | High - Python API | Medium - Command line |\n")
    report.append("| **Integration** | Native PyTorch | External tool |\n")
    report.append("| **Overhead** | Low (~5-10%) | Variable |\n")
    report.append("| **Python Stack Traces** | Yes | No |\n")
    report.append("| **Hardware Counters** | Limited | Extensive |\n")
    report.append("| **Memory Profiling** | Yes | Yes |\n")
    report.append("| **Kernel Analysis** | Basic | Deep |\n")
    report.append("| **Visualization** | Chrome Trace | NCU-UI |\n")
    report.append("| **Export Formats** | JSON, Chrome | CSV, JSON |\n")

    report.append("\n## Pros and Cons\n")

    report.append("\n### Kineto Profiler\n")
    report.append("**Pros:**\n")
    report.append("- Native PyTorch integration - no external dependencies\n")
    report.append("- Easy to use Python API\n")
    report.append("- Low profiling overhead\n")
    report.append("- Python-level stack traces\n")
    report.append("- Good for debugging PyTorch models\n")
    report.append("- Chrome trace visualization (chrome://tracing)\n")

    report.append("\n**Cons:**\n")
    report.append("- Limited hardware counter access\n")
    report.append("- Less detailed kernel-level analysis\n")
    report.append("- Focused on PyTorch operations\n")
    report.append("- No source-level correlation\n")

    report.append("\n### NVIDIA NSight Compute\n")
    report.append("**Pros:**\n")
    report.append("- Comprehensive hardware performance counters\n")
    report.append("- Deep kernel-level analysis\n")
    report.append("- Source code correlation (with debug info)\n")
    report.append("- Roofline analysis\n")
    report.append("- Memory access pattern analysis\n")
    report.append("- Works with any CUDA application\n")

    report.append("\n**Cons:**\n")
    report.append("- External tool - requires separate installation\n")
    report.append("- Higher profiling overhead\n")
    report.append("- No Python-level context\n")
    report.append("- More complex to set up and use\n")
    report.append("- Requires understanding of GPU architecture\n")

    report.append("\n## Recommendations\n")
    report.append("\n**Use Kineto when:**\n")
    report.append("- Profiling PyTorch models in development\n")
    report.append("- Need to identify Python-level bottlenecks\n")
    report.append("- Want quick, lightweight profiling\n")
    report.append("- Need to understand operator-level performance\n")

    report.append("\n**Use NVIDIA NSight Compute when:**\n")
    report.append("- Need detailed kernel-level analysis\n")
    report.append("- Optimizing specific CUDA kernels\n")
    report.append("- Need hardware counter data\n")
    report.append("- Investigating memory access patterns\n")
    report.append("- Performance tuning for production\n")

    report.append("\n## Conclusion\n")
    report.append("Both profilers serve different purposes:\n\n")
    report.append("- **Kineto** is ideal for PyTorch model development and Python-level profiling\n")
    report.append("- **NSight Compute** excels at deep kernel analysis and hardware optimization\n\n")
    report.append("For comprehensive profiling, use both: Kineto for high-level insights,")
    report.append(" NSight for kernel-level optimization.\n")

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n✓ Markdown report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare Kineto and NSight profiling results")
    parser.add_argument(
        "--configs",
        type=str,
        nargs='+',
        required=True,
        help="Configuration names to compare (e.g., config_a config_b)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing profiling results"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiler_comparison_report.md",
        help="Output markdown report file"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # Compare each configuration
    comparisons = []
    for config_name in args.configs:
        comparison = compare_profilers(config_name, results_dir)
        comparisons.append(comparison)

        # Save individual comparison
        output_file = results_dir / f"comparison_{config_name}.json"
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✓ Saved comparison: {output_file}")

    # Generate markdown report
    output_path = Path(args.output)
    generate_markdown_report(comparisons, output_path)

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
