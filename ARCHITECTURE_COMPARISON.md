# Kineto vs NVIDIA NSight Compute: Architectural Comparison

## Executive Summary

This document provides a comprehensive architectural comparison between **PyTorch Kineto** and **NVIDIA NSight Compute (NCU)**, two profiling tools with fundamentally different design philosophies and use cases.

---

## 1. Overview and Design Philosophy

### Kineto (PyTorch Profiler)
- **Origin**: Developed by Meta (Facebook) for PyTorch ecosystem
- **Philosophy**: Application-level profiling with ML framework integration
- **Primary Goal**: Understand ML workload behavior in the context of PyTorch operations
- **Scope**: End-to-end performance from Python → PyTorch → CUDA kernels
- **Target Users**: ML engineers, researchers, PyTorch developers

### NVIDIA NSight Compute (NCU)
- **Origin**: NVIDIA's official low-level GPU profiling tool
- **Philosophy**: Hardware-centric kernel analysis and optimization
- **Primary Goal**: Deep dive into individual GPU kernel performance
- **Scope**: GPU kernel execution with hardware counter analysis
- **Target Users**: CUDA developers, GPU performance engineers, HPC specialists

---

## 2. Architectural Layers

### Kineto Architecture Stack

```
┌─────────────────────────────────────────────────────────┐
│                  Python Layer (User Code)                │
├─────────────────────────────────────────────────────────┤
│              PyTorch Autograd Engine                     │
│         (torch.profiler.profile context)                 │
├─────────────────────────────────────────────────────────┤
│              Kineto Profiler Library                     │
│  - CPU events (via libkineto)                           │
│  - Python tracing                                        │
│  - Memory tracking                                       │
├─────────────────────────────────────────────────────────┤
│              CUPTI (CUDA Profiling Tools Interface)      │
│  - GPU kernel traces                                     │
│  - CUDA API calls                                        │
│  - Runtime events                                        │
├─────────────────────────────────────────────────────────┤
│                    CUDA Driver                           │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
1. **libkineto**: Core C++ library for event collection
2. **CUPTI Integration**: GPU tracing via NVIDIA's profiling API
3. **ActivityProfiler**: Collects CPU/GPU activity traces
4. **Python Bindings**: Direct integration with PyTorch autograd
5. **Timeline Generator**: Creates Chrome Trace format output

### NSight Compute Architecture Stack

```
┌─────────────────────────────────────────────────────────┐
│              ncu CLI / GUI Interface                     │
├─────────────────────────────────────────────────────────┤
│           Profiling Frontend & Orchestration             │
│  - Metric collection scheduling                         │
│  - Multi-pass kernel replay                             │
│  - Result aggregation                                    │
├─────────────────────────────────────────────────────────┤
│              Hardware Performance Counters               │
│  - SM (Streaming Multiprocessor) metrics                │
│  - Memory subsystem counters                            │
│  - Instruction pipeline stats                           │
│  - Warp scheduler metrics                               │
├─────────────────────────────────────────────────────────┤
│                  CUPTI / NVPA                            │
│  (NVIDIA Performance Analysis API)                      │
├─────────────────────────────────────────────────────────┤
│                  GPU Hardware                            │
│  - Performance monitoring units                         │
│  - Sampling units                                        │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
1. **Multi-pass Profiler**: Replays kernels to collect different metric sets
2. **Hardware Counter Collection**: Direct access to GPU performance counters
3. **Roofline Model**: Performance analysis based on compute/memory bounds
4. **Source Correlation**: Maps metrics to SASS/PTX/CUDA source
5. **Expert Systems**: Automated bottleneck detection and recommendations

---

## 3. Data Collection Mechanisms

### Kineto: Timeline-Based Tracing

**Mechanism:**
- **Single-pass collection**: Captures events during one execution
- **Event-based**: Records start/end timestamps for operations
- **Low overhead**: ~5-10% runtime overhead (typical)
- **Always-on capability**: Can profile production workloads

**What it captures:**
```
Timeline View:
├── Python function calls (stack traces)
├── PyTorch operators (conv2d, linear, softmax, etc.)
├── GPU kernel launches
│   ├── Kernel name
│   ├── Duration
│   ├── Grid/block dimensions
│   └── Memory transfers
├── CPU-GPU synchronization
└── Memory allocation/deallocation events
```

**Advantages:**
- See full execution context (Python → CUDA)
- Understand operator-level bottlenecks
- Identify CPU-GPU synchronization issues
- Correlate GPU work with PyTorch ops

**Limitations:**
- Limited hardware counter access
- No detailed per-SM metrics
- Cannot see instruction-level details
- No memory bank conflicts, cache hit rates, etc.

### NSight Compute: Multi-pass Kernel Profiling

**Mechanism:**
- **Multi-pass collection**: Replays kernels multiple times
- **Counter-based**: Collects 100+ hardware metrics per kernel
- **Higher overhead**: 10-100x slowdown (depends on metric sets)
- **Targeted profiling**: Profile specific kernels or ranges

**What it captures:**
```
Per-Kernel Analysis:
├── Compute Metrics
│   ├── SM utilization (active warps, eligible warps)
│   ├── IPC (instructions per cycle)
│   ├── Pipeline stalls (memory, execution, sync)
│   └── FP32/FP64/INT operations
├── Memory Metrics
│   ├── L1/L2 cache hit rates
│   ├── Global memory bandwidth utilization
│   ├── Shared memory bank conflicts
│   ├── Uncoalesced accesses
│   └── Memory transaction efficiency
├── Warp Metrics
│   ├── Warp divergence
│   ├── Predicated-off threads
│   └── Warp stall reasons
└── Instruction Analysis
    ├── SASS instruction mix
    ├── Register usage
    └── Occupancy limiters
```

**Advantages:**
- Deep hardware-level insights
- Precise bottleneck identification
- Roofline analysis for optimization guidance
- Source-level attribution (SASS/PTX/CUDA)

**Limitations:**
- No Python/framework context
- High overhead (unsuitable for full training runs)
- Requires kernel replay (non-deterministic kernels problematic)
- Steeper learning curve

---

## 4. Use Case Alignment

### When to Use Kineto

**Ideal Scenarios:**
1. **Initial profiling**: First-pass analysis of training/inference pipeline
2. **Operator-level optimization**: Identify slow PyTorch operations
3. **Data loading bottlenecks**: Find CPU preprocessing issues
4. **Multi-GPU profiling**: Understand distributed training behavior
5. **Production monitoring**: Low-overhead profiling in deployed systems
6. **Memory debugging**: Track GPU memory allocation patterns

**Example Questions Kineto Answers:**
- "Why is my training loop slow?"
- "Which PyTorch operator takes the most time?"
- "Are my GPU kernels compute-bound or memory-bound?" (high-level)
- "Is CPU data loading blocking GPU execution?"
- "Where are my CPU-GPU synchronization points?"

### When to Use NSight Compute

**Ideal Scenarios:**
1. **Kernel optimization**: Deep dive into custom CUDA kernels
2. **Hardware efficiency**: Maximize SM utilization, memory throughput
3. **Memory access patterns**: Diagnose cache issues, coalescing problems
4. **Register/occupancy tuning**: Optimize resource usage
5. **Instruction-level analysis**: Understand warp scheduling, stalls
6. **Comparative analysis**: Compare kernel implementations

**Example Questions NCU Answers:**
- "Why is my GEMM kernel slower than cuBLAS?"
- "What percentage of peak memory bandwidth am I achieving?"
- "Are my warp stalls due to memory latency or compute?"
- "Do I have shared memory bank conflicts?"
- "Is my kernel compute-bound or memory-bound?" (precise, with roofline)
- "What's limiting my occupancy - registers, shared memory, or blocks?"

---

## 5. Technical Deep Dive: Key Differentiators

### 5.1 Profiling Granularity

| Aspect | Kineto | NSight Compute |
|--------|--------|----------------|
| **Minimum unit** | PyTorch operator | Individual kernel invocation |
| **Time resolution** | Microseconds | Nanoseconds (hardware counters) |
| **Spatial detail** | Kernel-level | Warp/SM-level |
| **Context** | Full execution stack | Isolated kernel |

### 5.2 Overhead and Performance Impact

**Kineto:**
- **Tracing mode**: 5-10% overhead
- **With memory tracking**: 10-20% overhead
- **With stack traces**: 15-30% overhead
- **Mechanism**: Event buffering, no kernel replay

**NSight Compute:**
- **Basic metrics**: 10-50x slower
- **Full metric sets**: 50-200x slower
- **Mechanism**: Kernel replay with performance counter multiplexing
- **Note**: Some metrics require disabling optimizations

### 5.3 Data Output and Visualization

**Kineto:**
- **Format**: Chrome Trace (JSON), TensorBoard-compatible
- **Visualization**:
  - TensorBoard PyTorch Profiler plugin
  - Chrome's `chrome://tracing` viewer
  - VS Code Python extension
- **Structure**: Timeline with nested operations, stack traces
- **Export**: JSON, text summary

**NSight Compute:**
- **Format**: `.ncu-rep` (proprietary), SQLite database
- **Visualization**:
  - NSight Compute GUI (rich, interactive)
  - CLI text reports
  - Python API for custom analysis
- **Structure**: Per-kernel metrics with rules-based analysis
- **Export**: CSV, JSON, HTML

### 5.4 Integration with Frameworks

**Kineto:**
```python
# Native PyTorch integration
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    model(input)

# Automatic operator attribution
print(prof.key_averages().table())
```

**NSight Compute:**
```bash
# External tool, requires wrapping
ncu --set full --target-processes all \
    python train.py

# No native PyTorch operator awareness
# Must correlate kernel names manually
```

---

## 6. Hardware Counter Access

### Kineto Metrics (via CUPTI)

**Available:**
- Kernel duration
- Grid/block dimensions
- Memory copy volume and duration
- Device memory bandwidth (estimated)
- Basic occupancy
- SM clock cycles (limited)

**NOT Available:**
- Cache hit rates
- Memory transaction efficiency
- Warp stall breakdown
- Instruction mix
- Pipeline utilization

### NSight Compute Metrics (100+ metrics)

**Categories:**
1. **Compute**: IPC, pipeline utilization, FP/INT ops
2. **Memory**: L1/L2 hit rates, bandwidth efficiency, transaction counts
3. **Scheduler**: Warp states, stall reasons, eligible warps
4. **Instructions**: SASS mix, control flow efficiency
5. **Occupancy**: Theoretical vs achieved, limiting factors

**Example Metrics:**
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__sass_thread_inst_executed_op_ffma_pred_on.sum`
- `l1tex__t_sector_hit_rate.pct`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`

---

## 7. Profiling Workflow Comparison

### Kineto Workflow

```
1. Instrument code with torch.profiler.profile()
   ↓
2. Run training/inference (single pass)
   ↓
3. Export trace to Chrome Trace format
   ↓
4. Visualize in TensorBoard or Chrome
   ↓
5. Identify slow operators/kernels
   ↓
6. Optimize PyTorch code (model architecture, op fusion)
```

**Time investment**: Minutes to hours

### NSight Compute Workflow

```
1. Identify target kernel from initial profiling
   ↓
2. Run ncu with specific kernel filter
   ↓
3. Collect relevant metric sections (compute, memory, etc.)
   ↓
4. Analyze in NSight Compute GUI
   ↓
5. Use roofline model to identify bottleneck
   ↓
6. Examine source-level attribution (SASS/PTX)
   ↓
7. Optimize CUDA kernel code
   ↓
8. Re-profile and compare
```

**Time investment**: Hours to days (per kernel)

---

## 8. Complementary Usage Pattern

**Best Practice: Use Both Tools Sequentially**

```
Phase 1: Kineto (Macro-level)
├── Identify which parts of the model are slow
├── Find CPU-GPU synchronization issues
├── Detect data loading bottlenecks
└── Get operator-level breakdown
        ↓
Phase 2: NSight Compute (Micro-level)
├── Profile specific slow kernels identified by Kineto
├── Analyze hardware-level bottlenecks
├── Optimize kernel implementation
└── Validate optimizations with metrics
```

**Example:**
1. Kineto shows `aten::linear` takes 60% of training time
2. Kineto shows underlying GEMM kernel names
3. NSight Compute profiles those specific GEMM kernels
4. NCU reveals 40% DRAM bandwidth utilization → memory-bound
5. Optimize with better tiling, use Tensor Cores
6. Kineto confirms overall speedup in `aten::linear`

---

## 9. Architecture-Specific Considerations

### GPU Architecture Evolution

**Kineto:**
- Works across all CUDA-capable GPUs
- Metrics are architecture-agnostic (kernel names, durations)
- No architecture-specific tuning required
- Limited insight into arch-specific features (Tensor Cores, MIG)

**NSight Compute:**
- Metrics are architecture-specific
  - Ampere (A100): New metrics for TF32 Tensor Cores
  - Hopper (H100/H200): FP8 Tensor Cores, Thread Block Clusters
  - Ada: RT Cores, DLSS-specific metrics
- Roofline models adapted per architecture
- Detailed memory hierarchy specific to GPU model
- Can profile MIG instances independently

### H200-Specific Capabilities

**Kineto on H200:**
- Standard CUDA profiling (same as A100/H100)
- Can track memory usage up to 141GB HBM3e
- Timeline shows kernel execution

**NSight Compute on H200:**
- **New Hopper metrics**:
  - Thread Block Cluster metrics
  - TMA (Tensor Memory Accelerator) utilization
  - Asynchronous copy performance
  - Warp group specialization efficiency
- **Memory subsystem**:
  - HBM3e bandwidth analysis (4.8 TB/s peak)
  - L2 cache efficiency (50MB)
- **Tensor Core analysis**:
  - FP8/FP16/BF16/TF32 Tensor Core utilization
  - Sparse matrix acceleration

---

## 10. Pros and Cons Summary

### Kineto

**Pros:**
- ✅ Seamless PyTorch integration
- ✅ Low profiling overhead (~5-10%)
- ✅ Full execution context (Python → GPU)
- ✅ Easy to use, gentle learning curve
- ✅ Excellent for identifying high-level bottlenecks
- ✅ Multi-GPU/distributed training support
- ✅ Memory profiling with stack traces
- ✅ Production-ready (can profile in deployment)
- ✅ TensorBoard integration

**Cons:**
- ❌ Limited hardware metrics
- ❌ No instruction-level analysis
- ❌ Cannot diagnose low-level kernel inefficiencies
- ❌ No cache hit rates, memory transaction details
- ❌ Kernel-level insights are superficial
- ❌ Cannot compare against theoretical roofline
- ❌ No SASS/PTX correlation

### NVIDIA NSight Compute

**Pros:**
- ✅ Deep hardware-level analysis
- ✅ 100+ performance metrics
- ✅ Roofline model for optimization guidance
- ✅ Source-level correlation (CUDA/PTX/SASS)
- ✅ Precise bottleneck identification
- ✅ Expert system recommendations
- ✅ Architecture-specific insights
- ✅ Essential for CUDA kernel development

**Cons:**
- ❌ High profiling overhead (10-200x slower)
- ❌ No Python/framework awareness
- ❌ Steep learning curve (100+ metrics to understand)
- ❌ Requires kernel replay (problematic for non-deterministic code)
- ❌ Not suitable for full training runs
- ❌ Cannot see CPU-GPU interaction context
- ❌ Manual correlation with PyTorch operators needed

---

## 11. Conclusion: Which Tool to Choose?

### Decision Matrix

| Your Goal | Recommended Tool | Why |
|-----------|------------------|-----|
| Speed up PyTorch training | **Kineto** first | Identify slow ops, data loading issues |
| Optimize custom CUDA kernel | **NSight Compute** | Need hardware counters for tuning |
| Debug OOM errors | **Kineto** | Memory timeline with stack traces |
| Maximize memory bandwidth | **NSight Compute** | Analyze transaction efficiency |
| Profile distributed training | **Kineto** | Multi-GPU support, NCCL integration |
| Tune kernel occupancy | **NSight Compute** | Detailed occupancy analysis |
| Production monitoring | **Kineto** | Low overhead, continuous profiling |
| Compare vs cuBLAS/cuDNN | **NSight Compute** | Benchmark against library kernels |

### Recommended Workflow

**For ML Engineers:**
```
1. Start with Kineto
   - Identify slow model layers
   - Fix Python/data loading issues
   - Optimize model architecture
2. Use NSight Compute only if:
   - Custom CUDA kernels are slow
   - Need to optimize fused kernels
   - Library kernels underperform
```

**For CUDA Developers:**
```
1. Use Kineto to validate integration
   - Ensure kernel is actually being called
   - Check kernel launch overhead
2. Deep dive with NSight Compute
   - Analyze every new kernel
   - Iterate until metrics meet targets
   - Compare against roofline model
```

---

## 12. References and Further Reading

### Kineto
- [PyTorch Profiler Documentation](https://pytorch.org/docs/stable/profiler.html)
- [Kineto GitHub Repository](https://github.com/pytorch/kineto)
- [TensorBoard PyTorch Profiler Guide](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)

### NSight Compute
- [NSight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [Hopper Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)

### Performance Analysis
- [NVIDIA Blog: Understanding GPU Roofline Model](https://developer.nvidia.com/blog/)
- [Dissecting GPU Memory Hierarchy (Paper)](https://arxiv.org/abs/2108.08457)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
