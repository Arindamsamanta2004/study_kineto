"""
Utility functions for analyzing profiling results.

Provides common functions for parsing Kineto and NSight Compute outputs.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


def load_kineto_stats(result_dir: Path) -> Optional[Dict]:
    """
    Load Kineto profiling statistics from results directory.

    Args:
        result_dir: Path to kineto results directory

    Returns:
        Dictionary containing Kineto statistics or None if not found
    """
    stats_files = list(result_dir.glob("kineto_stats_*.json"))

    if not stats_files:
        print(f"Warning: No Kineto stats found in {result_dir}")
        return None

    stats_file = stats_files[0]

    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f"Error loading Kineto stats from {stats_file}: {e}")
        return None


def load_nsight_stats(result_dir: Path) -> Optional[Dict]:
    """
    Load NSight Compute statistics from results directory.

    Args:
        result_dir: Path to nsight results directory

    Returns:
        Dictionary containing NSight statistics or None if not found
    """
    stats_files = list(result_dir.glob("nsight_stats_*.json"))

    if not stats_files:
        print(f"Warning: No NSight stats found in {result_dir}")
        return None

    stats_file = stats_files[0]

    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        print(f"Error loading NSight stats from {stats_file}: {e}")
        return None


def parse_nsight_stdout(result_dir: Path) -> Optional[Dict]:
    """
    Parse NSight Compute stdout for kernel metrics.

    Args:
        result_dir: Path to nsight results directory

    Returns:
        Dictionary containing parsed metrics or None
    """
    stdout_files = list(result_dir.glob("nsight_stdout_*.txt"))

    if not stdout_files:
        return None

    stdout_file = stdout_files[0]

    try:
        with open(stdout_file, 'r') as f:
            content = f.read()

        # Parse key metrics from stdout
        metrics = {}

        # Look for common metric patterns
        # Example: "gpu__time_duration.sum                                     25.340 msecond"
        metric_pattern = r'(\S+)\s+([\d.]+)\s+(\S+)'

        for match in re.finditer(metric_pattern, content):
            metric_name = match.group(1)
            value = float(match.group(2))
            unit = match.group(3)

            metrics[metric_name] = {
                'value': value,
                'unit': unit
            }

        return metrics if metrics else None

    except Exception as e:
        print(f"Error parsing NSight stdout from {stdout_file}: {e}")
        return None


def extract_top_kernels(kineto_stats: Dict, top_k: int = 10) -> List[Dict]:
    """
    Extract top K GPU kernels from Kineto stats.

    Args:
        kineto_stats: Kineto statistics dictionary
        top_k: Number of top kernels to extract

    Returns:
        List of top kernel dictionaries
    """
    if not kineto_stats or 'kernel_stats' not in kineto_stats:
        return []

    kernels = kineto_stats['kernel_stats']

    # Sort by CUDA time (already sorted in the stats file, but ensure)
    sorted_kernels = sorted(
        kernels,
        key=lambda x: x.get('cuda_time_total_us', x.get('cpu_time_total_us', 0)),
        reverse=True
    )

    return sorted_kernels[:top_k]


def format_bytes(bytes_val: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def format_time(microseconds: float) -> str:
    """Format microseconds into human-readable string."""
    if microseconds < 1000:
        return f"{microseconds:.2f} μs"
    elif microseconds < 1000000:
        return f"{microseconds/1000:.2f} ms"
    else:
        return f"{microseconds/1000000:.2f} s"


def calculate_speedup(baseline: float, optimized: float) -> float:
    """Calculate speedup ratio."""
    if optimized == 0:
        return float('inf')
    return baseline / optimized


def calculate_overhead(profiled_time: float, baseline_time: float) -> float:
    """
    Calculate profiling overhead percentage.

    Args:
        profiled_time: Time with profiling enabled
        baseline_time: Time without profiling

    Returns:
        Overhead percentage
    """
    if baseline_time == 0:
        return 0.0
    return ((profiled_time - baseline_time) / baseline_time) * 100


def load_config(config_path: Path) -> Optional[Dict]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return None


def get_model_info(config: Dict) -> Dict:
    """Extract model information from configuration."""
    model_config = config.get('model', {})

    batch_size = model_config.get('batch_size', 0)
    seq_len = model_config.get('seq_len', 0)
    hidden_dim = model_config.get('hidden_dim', 0)

    # Calculate tensor sizes
    input_size_mb = (batch_size * seq_len * hidden_dim * 4) / (1024 ** 2)  # FP32

    # Calculate QKV sizes
    qkv_size_mb = input_size_mb * 3  # Q, K, V projections

    # Attention matrix size
    num_heads = model_config.get('num_heads', 1)
    attn_matrix_size_mb = (batch_size * num_heads * seq_len * seq_len * 4) / (1024 ** 2)

    return {
        'batch_size': batch_size,
        'seq_len': seq_len,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'input_size_mb': input_size_mb,
        'qkv_size_mb': qkv_size_mb,
        'attn_matrix_size_mb': attn_matrix_size_mb,
        'total_working_set_mb': input_size_mb + qkv_size_mb + attn_matrix_size_mb,
    }


def compare_metrics(kineto_val: float, nsight_val: float, metric_name: str) -> Dict:
    """
    Compare a metric between Kineto and NSight.

    Args:
        kineto_val: Value from Kineto
        nsight_val: Value from NSight
        metric_name: Name of the metric

    Returns:
        Comparison dictionary with differences
    """
    diff_abs = nsight_val - kineto_val
    diff_pct = ((nsight_val - kineto_val) / kineto_val * 100) if kineto_val != 0 else 0

    return {
        'metric': metric_name,
        'kineto': kineto_val,
        'nsight': nsight_val,
        'diff_absolute': diff_abs,
        'diff_percent': diff_pct,
    }


def aggregate_kernel_families(kernels: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate kernels by family (e.g., all GEMM kernels together).

    Args:
        kernels: List of kernel dictionaries

    Returns:
        Dictionary mapping kernel family to aggregated stats
    """
    families = {}

    for kernel in kernels:
        name = kernel.get('name', '')

        # Determine kernel family
        if 'gemm' in name.lower() or 'sgemm' in name.lower():
            family = 'GEMM'
        elif 'softmax' in name.lower():
            family = 'Softmax'
        elif 'elementwise' in name.lower() or 'copy' in name.lower():
            family = 'Elementwise'
        elif 'reduction' in name.lower() or 'reduce' in name.lower():
            family = 'Reduction'
        elif 'transpose' in name.lower():
            family = 'Transpose'
        else:
            family = 'Other'

        if family not in families:
            families[family] = {
                'count': 0,
                'total_time_us': 0,
                'kernels': []
            }

        families[family]['count'] += kernel.get('count', 1)
        families[family]['total_time_us'] += kernel.get('cuda_time_total_us', 0)
        families[family]['kernels'].append(kernel)

    return families


def create_comparison_table(rows: List[List[str]], headers: List[str]) -> str:
    """
    Create a formatted ASCII table.

    Args:
        rows: List of row data
        headers: Column headers

    Returns:
        Formatted table string
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]

    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create header
    header_line = "| " + " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    ) + " |"

    separator = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    # Create rows
    table_rows = []
    for row in rows:
        table_row = "| " + " | ".join(
            str(cell).ljust(w) for cell, w in zip(row, col_widths)
        ) + " |"
        table_rows.append(table_row)

    return "\n".join([header_line, separator] + table_rows)


def extract_metric_summary(kineto_stats: Dict, nsight_stats: Dict, config: Dict) -> Dict:
    """
    Extract summary metrics from both profilers for comparison.

    Args:
        kineto_stats: Kineto statistics
        nsight_stats: NSight statistics
        config: Configuration dictionary

    Returns:
        Summary dictionary with key metrics
    """
    summary = {
        'config_name': config.get('name', 'unknown'),
        'model': get_model_info(config),
    }

    # Kineto metrics
    if kineto_stats:
        summary['kineto'] = {
            'baseline_time_ms': kineto_stats.get('baseline_time_ms_per_iter', 0),
            'estimated_tflops': kineto_stats.get('estimated_tflops', 0),
            'memory_allocated_gb': kineto_stats.get('memory_stats', {}).get('allocated_gb', 0),
            'memory_reserved_gb': kineto_stats.get('memory_stats', {}).get('reserved_gb', 0),
            'top_kernels': extract_top_kernels(kineto_stats, top_k=5),
        }

    # NSight metrics (would need parsing of .ncu-rep file for detailed metrics)
    if nsight_stats:
        summary['nsight'] = {
            'success': nsight_stats.get('success', False),
            'ncu_output': nsight_stats.get('ncu_output', ''),
        }

    return summary
