"""
NVIDIA NSight Compute (NCU) Profiler Script

Profiles vector addition benchmark using NVIDIA NSight Compute.
Captures hardware performance counters, metrics, and kernel analysis.
"""

import argparse
import yaml
import torch
import json
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.vector_add_benchmark import create_benchmark_from_config


def profile_with_nsight(benchmark, config, config_path, output_dir, ncu_path="ncu"):
    """
    Profile benchmark using NVIDIA NSight Compute.

    This function creates a wrapper script that runs the benchmark
    and then profiles it with ncu.

    Args:
        benchmark: VectorAddBenchmark instance
        config: Configuration dictionary
        config_path: Path to the config file
        output_dir: Directory to save results
        ncu_path: Path to ncu executable
    """
    profiling_config = config['profiling']
    warmup_iters = profiling_config['warmup_iterations']
    profile_iters = profiling_config['profile_iterations']

    print(f"\n{'='*70}")
    print(f"NVIDIA NSIGHT COMPUTE PROFILING - {config['name']}")
    print(f"{'='*70}")
    print(f"Profile iterations: {profile_iters}")
    print(f"Output directory: {output_dir}")

    # Create a standalone script to profile
    profile_script = output_dir / f"_ncu_profile_{config['name']}.py"

    # Build metrics string
    nsight_config = config.get('nsight', {})
    metrics = nsight_config.get('metrics', [])
    metrics_str = ",".join(metrics) if metrics else ""

    # Get absolute path to config
    config_abs_path = Path(config_path).absolute()

    # Write profile script
    script_content = f"""#!/usr/bin/env python3
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.vector_add_benchmark import create_benchmark_from_config
import yaml

# Load config
with open('{config_abs_path}') as f:
    config = yaml.safe_load(f)

# Create benchmark
benchmark = create_benchmark_from_config(config)

# Warmup
print("Warming up...")
for i in range({warmup_iters}):
    benchmark.benchmark_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Running profiled iterations...")
# Run iterations for profiling
for i in range({profile_iters}):
    benchmark.benchmark_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Profiling complete!")
"""

    with open(profile_script, 'w') as f:
        f.write(script_content)

    profile_script.chmod(0o755)

    print(f"\n  ✓ Created profile script: {profile_script}")

    # Build ncu command
    ncu_output = output_dir / f"nsight_{config['name']}"

    ncu_cmd = [
        ncu_path,
        "--set", "full",  # Full metric set
        "--target-processes", "all",
        "--export", str(ncu_output),
        "--force-overwrite",
    ]

    # Add specific metrics if provided
    if metrics:
        ncu_cmd.extend(["--metrics", metrics_str])

    # Add print summary
    ncu_cmd.extend([
        "--print-summary", "per-kernel",
    ])

    # Add the python command
    ncu_cmd.extend([
        sys.executable,
        str(profile_script)
    ])

    print(f"\n  ✓ NCU command:")
    print(f"    {' '.join(ncu_cmd)}")

    # Create a shell script for easier manual running
    shell_script = output_dir / f"run_nsight_{config['name']}.sh"
    with open(shell_script, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"# NSight Compute profiling for {config['name']}\n\n")
        f.write(" \\\n  ".join(ncu_cmd))
        f.write("\n")

    shell_script.chmod(0o755)

    print(f"\n  ✓ Created shell script: {shell_script}")

    print(f"\n{'='*70}")
    print("RUNNING NSIGHT COMPUTE...")
    print(f"{'='*70}\n")

    try:
        # Run ncu
        result = subprocess.run(
            ncu_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Save stdout and stderr
        stdout_file = output_dir / f"nsight_stdout_{config['name']}.txt"
        stderr_file = output_dir / f"nsight_stderr_{config['name']}.txt"

        with open(stdout_file, 'w') as f:
            f.write(result.stdout)
        with open(stderr_file, 'w') as f:
            f.write(result.stderr)

        print(result.stdout)

        if result.returncode != 0:
            print(f"\nERROR: NCU exited with code {result.returncode}")
            print(f"STDERR:\n{result.stderr}")
            print(f"\nSaved error output to: {stderr_file}")
        else:
            print(f"\n  ✓ NCU output: {ncu_output}.ncu-rep")
            print(f"  ✓ Stdout: {stdout_file}")
            print(f"  ✓ Stderr: {stderr_file}")

        # Create summary stats
        stats = {
            "config_name": config['name'],
            "profiler": "nsight_compute",
            "model_config": config['model'],
            "profiling_config": profiling_config,
            "nsight_config": nsight_config,
            "ncu_output": str(ncu_output) + ".ncu-rep",
            "return_code": result.returncode,
            "success": result.returncode == 0,
        }

        stats_file = output_dir / f"nsight_stats_{config['name']}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"  ✓ Stats file: {stats_file}")

        print(f"\n{'='*70}")
        print("NSIGHT COMPUTE PROFILING COMPLETED!")
        print(f"{'='*70}")
        print(f"\nTo view results:")
        print(f"  ncu-ui {ncu_output}.ncu-rep")
        print(f"\nOr re-run manually:")
        print(f"  {shell_script}")
        print(f"{'='*70}\n")

        return stats

    except subprocess.TimeoutExpired:
        print("\nERROR: NCU profiling timed out (10 minutes)")
        return None
    except FileNotFoundError:
        print(f"\nERROR: NCU executable not found: {ncu_path}")
        print("Make sure NVIDIA NSight Compute is installed and in PATH")
        print("Or specify the path with --ncu-path")
        return None
    except Exception as e:
        print(f"\nERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Profile with NVIDIA NSight Compute")
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
    parser.add_argument(
        "--ncu-path",
        type=str,
        default="ncu",
        help="Path to ncu executable"
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir) / f"nsight_{config['name']}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. NSight Compute requires CUDA.")
        sys.exit(1)

    print(f"\nCUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")

    # Create benchmark (just for info, actual profiling runs in subprocess)
    print("\nCreating benchmark...")
    benchmark = create_benchmark_from_config(config)

    # Run profiling
    stats = profile_with_nsight(benchmark, config, config_path, output_dir, args.ncu_path)

    if stats and stats.get("success"):
        print(f"\nAll results saved to: {output_dir}")
    else:
        print("\nProfiling failed or incomplete.")
        sys.exit(1)


if __name__ == "__main__":
    main()
