# Usage Guide: Kineto vs NVIDIA NSight Compute Profiling

This guide provides step-by-step instructions for running the profiling benchmarks and analyzing results.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Understanding the Results](#understanding-the-results)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Required

- **NVIDIA GPU**: H200, H100, A100, or any CUDA-capable GPU
- **CUDA**: Version 12.0 or later
- **Python**: 3.8 or later
- **PyTorch**: 2.0 or later with CUDA support

### Optional (for NSight Compute profiling)

- **NVIDIA NSight Compute**: [Download here](https://developer.nvidia.com/nsight-compute)
  - Adds the `ncu` command to your PATH
  - Required for hardware-level profiling

### Installation

```bash
# Clone or navigate to the Kineto directory
cd /path/to/Kineto

# Install Python dependencies
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify NSight Compute (optional)
ncu --version
```

---

## Quick Start

### Run Everything at Once

```bash
cd profiling
./run_all.sh
```

This will:
1. Profile Config A (Batch=32, Seq=4K, Hidden=256) with both Kineto and NSight
2. Profile Config B (Batch=64, Seq=8K, Hidden=512) with both Kineto and NSight
3. Save all results to `results/` directory

### Generate Comparison Report

```bash
cd ..
python analysis/compare_results.py
```

View the report:
```bash
cat results/comparison_report.md
```

---

## Step-by-Step Guide

### Step 1: Run Kineto Profiling

Profile a single configuration with Kineto:

```bash
# Config A (small)
python profiling/profile_kineto.py --config configs/config_a.yaml

# Config B (medium)
python profiling/profile_kineto.py --config configs/config_b.yaml
```

**Expected output:**
- `results/kineto_config_a/trace.json` - Chrome trace for visualization
- `results/kineto_config_a/operator_summary.txt` - PyTorch operator breakdown
- `results/kineto_config_a/kernel_details.txt` - GPU kernel statistics
- `results/kineto_config_a/memory_summary.txt` - Memory allocation details
- `results/kineto_config_a/summary_stats.txt` - Overall profiling summary

**Time:** ~2-5 minutes per config

### Step 2: Visualize Kineto Results

#### Option A: Chrome Tracing (Timeline View)

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" button
4. Select `results/kineto_config_a/trace.json`

You'll see:
- Timeline of CPU and GPU activity
- PyTorch operators with durations
- GPU kernel launches
- Memory transfers
- Python stack traces (if enabled)

#### Option B: View Text Summaries

```bash
# Operator summary
cat results/kineto_config_a/operator_summary.txt

# Kernel details
cat results/kineto_config_a/kernel_details.txt

# Memory usage
cat results/kineto_config_a/memory_summary.txt
```

### Step 3: Run NSight Compute Profiling

Profile with hardware-level metrics:

```bash
# Config A
python profiling/profile_nsight.py --config configs/config_a.yaml

# Config B
python profiling/profile_nsight.py --config configs/config_b.yaml
```

**Expected output:**
- `results/nsight_config_a/nsight_config_a.ncu-rep` - NSight report file
- `results/nsight_config_a/nsight_stdout_config_a.txt` - Text output
- `results/nsight_config_a/run_nsight_config_a.sh` - Re-runnable script

**Time:** ~20-60 minutes per config (much slower than Kineto)

**Note:** NSight Compute has high overhead (10-200x slowdown) because it replays kernels multiple times to collect different metric sets.

### Step 4: View NSight Compute Results

#### Option A: NSight Compute GUI (Recommended)

```bash
# Launch GUI with report
ncu-ui results/nsight_config_a/nsight_config_a.ncu-rep
```

In the GUI you can:
- View roofline analysis
- Inspect hardware counters (SM utilization, memory bandwidth, etc.)
- See warp-level metrics and stall reasons
- Analyze cache hit rates
- Compare against theoretical peak performance
- View source-level attribution (CUDA/PTX/SASS)

#### Option B: Command-line Summary

```bash
# Print summary
ncu --import results/nsight_config_a/nsight_config_a.ncu-rep --print-summary per-kernel
```

### Step 5: Generate Comparison Report

```bash
python analysis/compare_results.py
```

This creates `results/comparison_report.md` with:
- Executive summary
- Architectural comparison (Kineto vs NSight)
- Per-config performance analysis
- Top kernels from each profiler
- Recommendations for optimization

---

## Understanding the Results

### Kineto Results Interpretation

**Key files to examine:**

1. **`operator_summary.txt`**: Shows PyTorch operators sorted by time
   - Look for operators taking >10% of total time
   - These are your optimization targets

2. **`kernel_details.txt`**: GPU kernel breakdown
   - Identifies which CUDA kernels are slow
   - Shows kernel launch counts

3. **`memory_summary.txt`**: Memory allocation patterns
   - Helps debug OOM (Out of Memory) issues
   - Shows peak memory usage

4. **`trace.json`**: Visual timeline (view in chrome://tracing)
   - See CPU-GPU interaction
   - Identify synchronization points
   - Detect data loading bottlenecks

**What to look for:**
- ✅ High GPU utilization (minimal gaps in timeline)
- ✅ Overlapped CPU/GPU work
- ❌ Long CPU preprocessing (data loading bottleneck)
- ❌ Frequent CPU-GPU synchronization
- ❌ Large memory allocations during training

### NSight Compute Results Interpretation

**Key metrics to examine:**

1. **GPU Time Duration**: Actual kernel execution time
   - Compare with Kineto times for validation

2. **SM Utilization** (`sm__cycles_elapsed.avg.pct_of_peak_sustained_active`)
   - Target: >80% for compute-bound kernels
   - Low (<50%): Memory-bound or low occupancy

3. **Memory Bandwidth** (`dram__throughput.avg.pct_of_peak_sustained_elapsed`)
   - Target: >80% for memory-bound kernels
   - H200 peak: ~4.8 TB/s

4. **Occupancy**: Warps active vs. theoretical maximum
   - Low occupancy → Check register/shared memory usage

5. **Cache Hit Rates**: L1/L2 cache efficiency
   - Low hit rates → Poor data locality

6. **Warp Stalls**: Reasons for warp not issuing instructions
   - Memory dependency stalls → Memory-bound
   - Execution dependency → Compute-bound
   - Sync stalls → Too much synchronization

**Optimization workflow:**
1. Identify bottleneck from roofline model
2. If memory-bound: Improve cache locality, reduce memory traffic
3. If compute-bound: Increase parallelism, use Tensor Cores

---

## Troubleshooting

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size in config
# Edit configs/config_a.yaml:
#   batch_size: 16  # was 32

# Or reduce sequence length
#   seq_len: 2048  # was 4096
```

### NSight Compute Not Found

**Problem:** `ERROR: NCU executable not found`

**Solutions:**
```bash
# Install NSight Compute
# https://developer.nvidia.com/nsight-compute

# Add to PATH (Ubuntu/Linux)
export PATH=$PATH:/opt/nvidia/nsight-compute/2024.1/ncu

# Or specify path explicitly
python profiling/profile_nsight.py --config configs/config_a.yaml --ncu-path /path/to/ncu
```

### NSight Profiling Takes Too Long

**Problem:** NSight profiling is extremely slow

**Solutions:**
```bash
# Reduce profile iterations in config
# Edit configs/config_a.yaml:
#   profile_iterations: 10  # was 100

# Or use lighter metric set (edit nsight metrics in config)
# Remove some metrics from nsight.metrics list

# Or profile only specific kernels
ncu --kernel-name "gemm" python profiling/profile_nsight.py --config configs/config_a.yaml
```

### Permission Denied on Scripts

**Problem:** `bash: ./run_all.sh: Permission denied`

**Solution:**
```bash
chmod +x profiling/run_all.sh
chmod +x analysis/compare_results.py
```

### ImportError or Module Not Found

**Problem:** `ImportError: No module named 'yaml'`

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install specific package
pip install pyyaml
```

---

## Advanced Usage

### Profile Only Specific Operations

Modify the profiling scripts to filter specific operators or kernels:

```python
# In profiling/profile_kineto.py
with torch.profiler.profile(...) as prof:
    with torch.profiler.record_function("attention_only"):
        # Profile only attention, not full benchmark_step
        output, attn_weights = model.attention(x)
```

### Custom Configurations

Create your own config file:

```bash
cp configs/config_a.yaml configs/my_config.yaml

# Edit my_config.yaml with your settings
# Then profile:
python profiling/profile_kineto.py --config configs/my_config.yaml
```

### Profile Specific NSight Metrics

Edit the `nsight.metrics` section in your config:

```yaml
nsight:
  metrics:
    - "gpu__time_duration.sum"
    - "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    - "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    # Add more metrics from: ncu --query-metrics
```

List available metrics:
```bash
ncu --query-metrics
```

### Batch Profiling Multiple Configs

```bash
# Create custom configs
for batch in 16 32 64; do
    sed "s/batch_size: .*/batch_size: $batch/" configs/config_a.yaml > configs/batch_${batch}.yaml
done

# Profile all
./profiling/run_all.sh --config configs/batch_16.yaml --config configs/batch_32.yaml --config configs/batch_64.yaml
```

### Export Results for External Analysis

```python
# Load Kineto results in Python
import json
with open('results/kineto_config_a/kineto_stats_config_a.json') as f:
    stats = json.load(f)

# Access metrics
print(f"TFLOP/s: {stats['estimated_tflops']}")
print(f"Memory: {stats['memory_stats']['allocated_gb']} GB")
```

### Compare Against Baseline

```python
# Create baseline measurement
python -c "
import torch
from model.attention_benchmark import AttentionBenchmark
import time

model = AttentionBenchmark(batch_size=32, seq_len=4096, hidden_dim=256, num_heads=8)

# Warmup
for _ in range(10):
    model.benchmark_step()
torch.cuda.synchronize()

# Measure
start = time.perf_counter()
for _ in range(100):
    model.benchmark_step()
torch.cuda.synchronize()
end = time.perf_counter()

print(f'Baseline: {(end-start)/100*1000:.2f} ms/iter')
"
```

---

## Next Steps

1. **Read the architectural comparison**: See [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for in-depth understanding of the differences between Kineto and NSight Compute.

2. **Optimize your model**: Based on profiling results:
   - Use Kineto to identify slow operators
   - Use NSight to optimize specific kernels
   - Consider using Tensor Cores (FP16/BF16)
   - Implement kernel fusion

3. **Profile on H200**: Transfer this directory to your H200 container and run the profiling there for realistic performance measurements.

4. **Share results**: The generated reports can be used for:
   - Technical documentation
   - Performance analysis presentations
   - Optimization planning
   - Comparative studies

---

## Questions or Issues?

- Check [ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md) for conceptual questions
- Review [README.md](README.md) for project overview
- Check NSight Compute docs: https://docs.nvidia.com/nsight-compute/
- PyTorch Profiler docs: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
