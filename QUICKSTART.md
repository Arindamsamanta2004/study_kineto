# Quick Start Guide

## Overview

This benchmark compares **Kineto** (PyTorch's profiler) vs **NVIDIA NSight Compute** for GPU profiling.

**Benchmark:** Vector addition (C = A + B)

**Configs:**
- **Config A**: Batch=32, Seq=4K, Hidden=256 (FP32) - 33.5M elements
- **Config B**: Batch=64, Seq=8K, Hidden=512 (FP32) - 268.4M elements

## Setup (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup:**
   ```bash
   python3 test_setup.py
   ```

## Setup (H200 Container)

1. **Clone/copy repository:**
   ```bash
   git clone <your-repo-url>
   cd Kineto
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify CUDA:**
   ```bash
   python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

## Running Profiling

### Option 1: Run Everything (Recommended)

```bash
./run_all_profilers.sh
```

This will:
- Run Kineto profiling for both configs
- Run NSight profiling for both configs
- Generate comparison report

### Option 2: Run Individual Profilers

**Kineto only:**
```bash
./run_all_profilers.sh --kineto-only
```

**NSight only:**
```bash
./run_all_profilers.sh --nsight-only
```

**Single config:**
```bash
./run_all_profilers.sh -c config_a
```

### Option 3: Manual Execution

**Test benchmark (no profiling):**
```bash
python3 benchmarks/vector_add_benchmark.py --config configs/config_a.yaml --iterations 100
```

**Kineto profiling:**
```bash
python3 profiling/profile_kineto.py --config configs/config_a.yaml --output-dir results
```

**NSight profiling:**
```bash
python3 profiling/profile_nsight.py --config configs/config_a.yaml --output-dir results
```

**Compare results:**
```bash
python3 analysis/compare_profilers.py --configs config_a config_b --results-dir results
```

## Viewing Results

### Kineto Results

**Chrome Trace (visual timeline):**
1. Open Chrome
2. Go to `chrome://tracing`
3. Load `results/kineto_*/kineto_trace_*.json`

**Text summary:**
```bash
cat results/kineto_*/kineto_stacks_*.txt
```

**JSON data:**
```bash
cat results/kineto_*/kineto_stats_*.json | jq .
```

### NSight Results

**GUI viewer:**
```bash
ncu-ui results/nsight_*/nsight_*.ncu-rep
```

**Command line:**
```bash
ncu --import results/nsight_*/nsight_*.ncu-rep --print-summary per-kernel
```

### Comparison Report

```bash
cat results/profiler_comparison_report.md
```

## Expected Results

### Kineto Profiler

**Pros:**
- Low overhead (~5-10%)
- Easy Python integration
- Python stack traces
- Chrome trace visualization

**Cons:**
- Limited hardware counters
- Less detailed kernel analysis

### NVIDIA NSight Compute

**Pros:**
- Comprehensive hardware counters
- Deep kernel analysis
- Memory throughput metrics
- Roofline analysis

**Cons:**
- Higher profiling overhead
- External tool setup required
- No Python-level context

## Troubleshooting

### PyTorch not found
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### CUDA not available
```bash
# Check GPU
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

### NSight not found
```bash
# Check if installed
which ncu

# Or specify path
python3 profiling/profile_nsight.py --config configs/config_a.yaml --ncu-path /path/to/ncu
```

### Out of memory
Edit config files to reduce batch size:
```yaml
model:
  batch_size: 16  # Reduce from 32/64
```

## Next Steps

After profiling:

1. **Analyze Chrome trace** - Identify GPU idle time, kernel durations
2. **Review NSight metrics** - Check memory bandwidth utilization, SM efficiency
3. **Compare overhead** - Kineto vs NSight profiling overhead
4. **Read comparison report** - Understand trade-offs between profilers

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review code comments in source files
- Consult PyTorch profiler docs: https://pytorch.org/docs/stable/profiler.html
- NSight docs: https://docs.nvidia.com/nsight-compute/
