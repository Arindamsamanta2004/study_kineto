# Kineto vs NVIDIA NSight Compute - Profiling Comparison

A comprehensive benchmark suite for comparing PyTorch's Kineto profiler with NVIDIA NSight Compute (NCU) for GPU kernel profiling.

## Overview

This project provides a comparative analysis of two GPU profiling tools using vector addition kernels:

- **Kineto**: PyTorch's built-in profiler with Python-level integration
- **NVIDIA NSight Compute**: NVIDIA's low-level kernel profiling tool

## Configurations

### Config A: Small shapes - cache friendly
- Batch size: 32
- Sequence length: 4096
- Hidden dimension: 256
- Precision: FP32

### Config B: Medium shapes - cache friendly
- Batch size: 64
- Sequence length: 8192
- Hidden dimension: 512
- Precision: FP32

## Project Structure

```
Kineto/
├── benchmarks/
│   └── vector_add_benchmark.py    # Vector addition benchmark kernel
├── configs/
│   ├── config_a.yaml              # Config A: Batch=32, Seq=4K, Hidden=256
│   └── config_b.yaml              # Config B: Batch=64, Seq=8K, Hidden=512
├── profiling/
│   ├── profile_kineto.py          # Kineto profiler script
│   └── profile_nsight.py          # NSight profiler script
├── analysis/
│   └── compare_profilers.py       # Comparison analysis tool
├── results/                        # Profiling results (generated)
├── run_all_profilers.sh           # Main execution script
├── README.md                       # This file
└── requirements.txt               # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run All Profilers

Run both Kineto and NSight profiling on all configurations:

```bash
./run_all_profilers.sh
```

### Run Individual Profilers

**Kineto only:**
```bash
./run_all_profilers.sh --kineto-only
```

**NSight only:**
```bash
./run_all_profilers.sh --nsight-only
```

**Specific configuration:**
```bash
./run_all_profilers.sh -c config_a
```

### Run Profilers Manually

**Kineto profiling:**
```bash
python3 profiling/profile_kineto.py --config configs/config_a.yaml --output-dir results
```

**NSight profiling:**
```bash
python3 profiling/profile_nsight.py --config configs/config_a.yaml --output-dir results
```

**Comparison analysis:**
```bash
python3 analysis/compare_profilers.py --configs config_a config_b --results-dir results
```

### Test Benchmark Only

Run the benchmark without profiling:
```bash
python3 benchmarks/vector_add_benchmark.py --config configs/config_a.yaml --iterations 100
```

## Viewing Results

### Kineto Results

**Chrome Trace (Timeline view):**
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load `results/kineto_*/kineto_trace_*.json`

**Text Summary:**
```bash
cat results/kineto_*/kineto_stacks_*.txt
```

**JSON Stats:**
```bash
cat results/kineto_*/kineto_stats_*.json | jq .
```

### NSight Compute Results

**NCU-UI (Graphical):**
```bash
ncu-ui results/nsight_*/nsight_*.ncu-rep
```

**Command Line:**
```bash
ncu --import results/nsight_*/nsight_*.ncu-rep --print-summary per-kernel
```

### Comparison Report

```bash
cat results/profiler_comparison_report.md
```

## Performance Metrics

For vector addition (C = A + B):

**Memory Bandwidth:**
- Theoretical: Read A + Read B + Write C = 3N elements
- Config A: ~134 MB × 3 = ~402 MB per iteration
- Config B: ~1.07 GB × 3 = ~3.22 GB per iteration

**FLOPS:**
- Config A: 33.5M FP32 additions
- Config B: 268.4M FP32 additions

**Expected Performance (H200):**
- Memory bandwidth: ~4.8 TB/s (theoretical)
- Compute: ~67 TFLOPS FP32

## Requirements

### Software Dependencies

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU (tested on H200)
- NVIDIA NSight Compute (optional, for NSight profiling)
- PyYAML

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify PyTorch CUDA:**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Install NVIDIA NSight Compute (optional):**
   - Download from: https://developer.nvidia.com/nsight-compute
   - Or install via CUDA Toolkit
   - Verify: `ncu --version`

## Comparative Analysis

### When to Use Kineto

✅ PyTorch model development
✅ Python-level bottleneck identification
✅ Quick iteration during development
✅ Operator-level performance analysis
✅ Low-overhead continuous profiling

### When to Use NSight Compute

✅ Kernel optimization
✅ Hardware counter analysis
✅ Memory access pattern investigation
✅ Production performance tuning
✅ Deep dive into specific kernels

## Running on H200 Container

1. **Pull code to container:**
   ```bash
   git clone <your-repo-url>
   cd Kineto
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run profiling:**
   ```bash
   ./run_all_profilers.sh
   ```

4. **Copy results back:**
   ```bash
   # From container
   tar -czf results.tar.gz results/

   # Copy to local machine
   docker cp <container-id>:/path/to/results.tar.gz .
   ```

## References

- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA NSight Compute](https://developer.nvidia.com/nsight-compute)
- [Kineto GitHub](https://github.com/pytorch/kineto)
