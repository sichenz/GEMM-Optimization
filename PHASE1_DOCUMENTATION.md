# Phase 1: Foundation & Benchmarking - Complete Documentation

## Overview

Phase 1 establishes the foundation for our GEMM optimization project. The goal is to:
1. Understand our target hardware (NVIDIA T4 GPU)
2. Benchmark our baseline implementation (Lab-1 tiled GEMM)
3. Establish performance targets (cuBLAS and CUTLASS)
4. Analyze performance gaps and identify optimization opportunities

This document explains how each component works and how they fit together.

---

## 1. Hardware Analysis (`src/gpu_specs.cu`)

### Purpose
**Phase 1.1.1**: Document GPU specifications and calculate theoretical performance limits.

### How It Works

The `gpu_specs.cu` program queries the GPU using CUDA runtime APIs and calculates key metrics:

#### 1.1 Basic Information Collection
- **GPU Name**: Identifies the specific GPU model
- **Compute Capability**: Architecture version (7.5 for T4 = Turing)
- **Memory**: Total global memory, memory clock rate, bus width
- **Clock Rates**: Base clock rate for compute units

#### 1.2 Compute Resources
- **SMs (Streaming Multiprocessors)**: Number of compute units
- **CUDA Cores**: Total number of FP32 cores
- **Thread Limits**: Max threads per block, per SM, grid dimensions
- **Warp Size**: Always 32 threads (fundamental execution unit)

#### 1.3 Memory Hierarchy
- **L2 Cache**: Large shared cache (helps with data reuse)
- **Shared Memory**: Fast on-chip memory per SM (used for tiling)
- **Registers**: Fastest memory, per-thread storage
- **Constant Memory**: Read-only memory for constants

#### 1.4 Performance Characteristics Calculation

**Peak Memory Bandwidth**:
```
Bandwidth (GB/s) = 2 × Memory Clock (MHz) × Bus Width (bits) / 8 / 1000
```
- The "2" comes from DDR (Double Data Rate)
- This is the maximum rate we can read/write from global memory
- **Example**: T4 with 5000 MHz memory clock, 256-bit bus = ~320 GB/s

**Peak FP32 TFLOPS**:
```
TFLOPS = (CUDA Cores × Clock Rate (GHz) × 2) / 1000
```
- The "2" comes from FMA (Fused Multiply-Add) = 2 operations per cycle
- **Example**: T4 with 2560 cores at 1.59 GHz = ~8.1 TFLOPS

**Peak FP16 TensorCore TFLOPS**:
- TensorCores are specialized units for matrix operations
- Much faster than regular FP32 cores (8x on T4)
- **Example**: T4 = ~65 TFLOPS for FP16 TensorCore operations

#### 1.5 Arithmetic Intensity Analysis

**Arithmetic Intensity (AI)** = FLOPS / Bytes Transferred

This tells us if a workload is:
- **Memory-bound** (AI < ridge point): Limited by memory bandwidth
- **Compute-bound** (AI > ridge point): Limited by compute throughput

**Ridge Point** = Peak TFLOPS / Peak Bandwidth
- **FP32 Ridge Point**: ~23.9 FLOPS/Byte (for T4)
- **FP16 Ridge Point**: ~104.2 FLOPS/Byte (for T4)

Workloads with AI below the ridge point are memory-bound (need better memory access patterns).
Workloads with AI above the ridge point are compute-bound (need more compute operations per data loaded).

### Output
- `results/gpu_specs.txt`: Complete GPU specifications report
- Used by `roofline_analysis.py` to calculate theoretical performance limits

---

## 2. Benchmarking Framework (`src/benchmark_gemm.cu`)

### Purpose
**Phase 1.1.3**: Set up timing harness to measure FLOPS, memory bandwidth, and latency across different matrix sizes.

### How It Works

#### 2.1 Timing Infrastructure

**CudaTimer Class**:
- Uses CUDA events (`cudaEventRecord`) for accurate GPU timing
- More accurate than CPU timers because:
  - Measures GPU execution time directly
  - Accounts for async execution
  - Avoids CPU-GPU synchronization overhead

**Timing Pattern**:
```cpp
timer.start();                    // Record start event
for (int i = 0; i < iters; i++) {
    kernel<<<...>>>();            // Launch kernel
}
cudaDeviceSynchronize();          // Wait for all kernels to finish
float time_ms = timer.stop();     // Calculate elapsed time
```

#### 2.2 Performance Metrics

**GFLOPS Calculation**:
```cpp
GFLOPS = (2 × M × N × K) / (time_ms / 1000.0) / 1e9
```
- GEMM requires 2×M×N×K operations (each element needs K multiply-adds = 2 ops)
- Converts to Giga-FLOPS per second

**Memory Bandwidth**:
```cpp
Bandwidth (GB/s) = (M×K + K×N + M×N) × bytes_per_element / (time_ms / 1000.0) / 1e9
```
- Total bytes = read A matrix + read B matrix + write C matrix
- Measures how efficiently we're using memory bandwidth

#### 2.3 Benchmarking Process

1. **Warmup Runs**: 
   - GPU needs a few iterations to "warm up"
   - Clock speeds stabilize, caches warm up
   - Ensures consistent timing

2. **Benchmark Runs**:
   - Multiple iterations (typically 20)
   - Average time for stability
   - Reduces measurement noise

3. **Matrix Configurations**:
   - **Square**: 128², 256², 512², 1024², 2048², 4096², 8192²
   - **Rectangular**: Common ML shapes (e.g., 4096×256×1024 for attention)

#### 2.4 Kernels Benchmarked

1. **Lab-1 Tiled GEMM** (`benchmarkLab1GEMM`):
   - Our baseline implementation
   - Uses 32×32 tiling with shared memory
   - FP32 precision

2. **cuBLAS SGEMM** (`benchmarkCublasSGEMM`):
   - NVIDIA's optimized FP32 GEMM
   - Performance target for our FP32 implementation
   - Uses column-major format (Fortran-style)

3. **cuBLAS HGEMM** (`benchmarkCublasHGEMM`):
   - Mixed precision: FP16 input, FP32 accumulation
   - Uses TensorCores (much faster)
   - Performance target for our TensorCore implementation

### Output
- `results/benchmark_results.csv`: All benchmark results in CSV format
- Used by `roofline_analysis.py` for visualization

---

## 3. Lab-1 Tiled GEMM (`src/ops/op_mm.cuh`)

### Purpose
**Phase 1.1.3**: Baseline matrix multiplication kernel using tiling/blocking optimization.

### How It Works

#### 3.1 Tiling Strategy

**Problem**: Naive GEMM has poor memory access patterns
- Each element of A and B is read multiple times
- Global memory is slow (~400 cycles latency)
- No data reuse between threads

**Solution**: Blocked/Tiled GEMM
- Load small tiles into shared memory (fast, ~20 cycles latency)
- Reuse data within tile multiple times
- Reduces global memory traffic

#### 3.2 Kernel Structure

**Tile Size**: 32×32
- 32×32 = 1024 threads per block (fits GPU limits)
- 32×32×4 bytes = 4KB per tile (fits in shared memory)
- Warp-aligned (32 threads = 1 warp)

**Thread Mapping**:
- Each threadblock computes one 32×32 output tile
- Each thread computes one element of the output
- Grid size: `(ceil(M/32), ceil(N/32))` blocks

**Algorithm**:
```
For each threadblock (computing one 32×32 output tile):
  For each K-tile (K dimension in chunks of 32):
    1. Load 32×32 tile from A into shared memory
    2. Load 32×32 tile from B into shared memory
    3. Synchronize (wait for all loads)
    4. Compute partial dot product using shared memory
    5. Synchronize (before loading next tiles)
  Write result to global memory
```

#### 3.3 Memory Access Pattern

**Global Memory → Shared Memory**:
- Coalesced access: threads in a warp access consecutive memory
- Each thread loads one element
- 32 threads = 128 bytes (one cache line) = efficient

**Shared Memory → Registers**:
- All threads read from shared memory simultaneously
- No bank conflicts (if properly aligned)
- Very fast (~20 cycles)

**Computation**:
- Accumulate in registers (fastest memory)
- FMA operations (multiply-add in one instruction)
- High arithmetic intensity within tile

#### 3.4 Boundary Handling

For matrices not divisible by 32:
- Check bounds before loading: `if (row < A.h && kA < A.w)`
- Pad with zeros if out of bounds
- Check bounds before writing: `if (row < C.h && col < C.w)`

### Performance Characteristics

**Strengths**:
- Much better than naive implementation
- Good for medium-to-large matrices (2048×2048+)
- Achieves ~2 TFLOPS (13-15% of peak FP32)

**Limitations**:
- Still far from cuBLAS performance
- Not using TensorCores
- No advanced optimizations (pipelining, vectorization, etc.)

---

## 4. cuBLAS Baseline (`src/baselines/cublas_bench.cu`)

### Purpose
**Phase 1.2.1**: Establish performance ceiling targets using NVIDIA's optimized library.

### How It Works

#### 4.1 cuBLAS SGEMM (FP32)

**What it is**:
- NVIDIA's highly optimized FP32 GEMM implementation
- Uses advanced techniques: software pipelining, vectorization, optimal tile sizes
- Represents the "best case" for FP32 performance

**Usage**:
```cpp
cublasSgemm(handle, 
            CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
            M, N, K,                   // Matrix dimensions
            &alpha,                    // Scaling factor
            A_d, M,                    // Matrix A, leading dimension
            B_d, K,                    // Matrix B, leading dimension
            &beta,                     // Scaling factor for C
            C_d, M);                   // Matrix C, leading dimension
```

**Performance**:
- T4: ~8-13 TFLOPS (60-80% of peak)
- Much better than our Lab-1 kernel
- This is our target for Phase 2 FP32 optimizations

#### 4.2 cuBLAS GemmEx (Mixed Precision)

**What it is**:
- FP16 input, FP32 accumulation, FP32 output
- Uses TensorCores when available (Compute 7.0+)
- Much faster than FP32 (5-7x speedup on T4)

**Usage**:
```cpp
cublasGemmEx(handle,
             CUBLAS_OP_N, CUBLAS_OP_N,
             M, N, K,
             &alpha,
             A16_d, CUDA_R_16F, M,    // FP16 input
             B16_d, CUDA_R_16F, K,    // FP16 input
             &beta,
             C32_d, CUDA_R_32F, M,     // FP32 output
             CUDA_R_32F,               // Accumulation type
             CUBLAS_GEMM_DEFAULT_TENSOR_OP);  // Use TensorCores
```

**Performance**:
- T4: ~50-90 TFLOPS (70-90% of peak TensorCore)
- This is our target for Phase 2 TensorCore implementation

### Why cuBLAS is Fast

1. **Optimal Tile Sizes**: Tuned for each GPU architecture
2. **Software Pipelining**: Overlaps computation with memory loads
3. **Vectorization**: Uses 128-bit loads (float4) for coalesced access
4. **Register Tiling**: Multiple tiles in registers for better reuse
5. **TensorCores**: Specialized hardware for matrix operations

---

## 5. Performance Analysis (`src/roofline_analysis.py`)

### Purpose
**Phase 1.3**: Generate roofline model and performance comparison visualizations.

### How It Works

#### 5.1 Roofline Model

**What is a Roofline Model?**
- X-axis: Arithmetic Intensity (FLOPS/Byte)
- Y-axis: Performance (GFLOPS)
- Roofline curve: Theoretical peak performance at each AI

**Two Regions**:
1. **Memory-Bound** (left side, low AI):
   - Performance = AI × Bandwidth (slope)
   - Limited by memory bandwidth
   - Optimization: Improve memory access patterns

2. **Compute-Bound** (right side, high AI):
   - Performance = Peak FLOPS (flat line)
   - Limited by compute throughput
   - Optimization: Increase arithmetic intensity

**Ridge Point**: Where memory-bound and compute-bound regions meet
- AI at ridge point = Peak FLOPS / Peak Bandwidth
- Workloads with AI > ridge point are compute-bound
- Workloads with AI < ridge point are memory-bound

#### 5.2 Arithmetic Intensity Calculation

For GEMM:
```
AI = (2 × M × N × K) / ((M×K + K×N + M×N) × bytes_per_element)
```

- **FLOPS**: 2×M×N×K (each element needs K multiply-adds)
- **Bytes**: Read A (M×K) + Read B (K×N) + Write C (M×N)

**Example**: 2048×2048×2048 FP32 GEMM
- FLOPS = 2 × 2048³ = 17.18 billion
- Bytes = (2048² + 2048² + 2048²) × 4 = 50.33 MB
- AI = 17.18e9 / 50.33e6 = 341 FLOPS/Byte

This is well above the ridge point (~24), so it's compute-bound.

#### 5.3 Visualization

**Roofline Plot**:
- Shows theoretical performance limits (roofline)
- Shows actual kernel performance (points)
- Identifies which kernels are memory-bound vs compute-bound
- Shows optimization opportunities (distance from roofline)

**Performance Comparison Plot**:
- GFLOPS vs matrix size
- Efficiency vs cuBLAS
- Shows where our kernel performs well/poorly

**Analysis Report**:
- Summary statistics
- Efficiency calculations
- Optimization recommendations

### Output
- `results/roofline_plot.png`: Roofline visualization
- `results/performance_comparison.png`: Performance comparison
- `results/analysis_report.txt`: Detailed analysis

---

## 6. Profiling with Nsight Compute (`src/analyze_nsight_profile.py`)

### Purpose
**Phase 1.1.4**: Profile kernel to identify bottlenecks (occupancy, memory throughput, compute utilization).

### How It Works

#### 6.1 Nsight Compute

**What it is**:
- NVIDIA's GPU kernel profiler
- Provides detailed performance metrics
- Helps identify optimization opportunities

**Key Metrics**:
1. **Occupancy**: % of maximum warps per SM
   - Low occupancy = underutilized GPU
   - Causes: Too many registers, too much shared memory

2. **Memory Throughput**: % of peak bandwidth achieved
   - Low throughput = memory access inefficiencies
   - Causes: Poor coalescing, bank conflicts

3. **Compute Utilization**: % of peak FLOPS achieved
   - Low utilization = compute inefficiencies
   - Causes: Poor instruction mix, dependencies

4. **Warp Stall Reasons**: Why warps are waiting
   - Memory stalls: Waiting for memory
   - Compute stalls: Waiting for compute units
   - Synchronization: Waiting for barriers

#### 6.2 Profiling Process

1. **Build with Debug Info**:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
   ```

2. **Run Nsight Compute**:
   ```bash
   ncu --set full \
       --export results/profiling/lab1_2048 \
       --kernel-name regex:op_mm_kernel \
       ./build_profile/benchmark_gemm
   ```

3. **Analyze Results**:
   - Open `.ncu-rep` file in Nsight Compute UI
   - Or use `analyze_nsight_profile.py` for automated analysis

#### 6.3 Common Issues Found

**Low Occupancy**:
- Too many registers per thread → Reduce register usage
- Too much shared memory → Reduce tile size or use less shared memory

**Low Memory Throughput**:
- Poor coalescing → Ensure consecutive threads access consecutive memory
- Bank conflicts → Align shared memory access patterns

**Low Compute Utilization**:
- Not enough work per thread → Increase tile size
- Instruction dependencies → Better instruction scheduling

### Output
- `results/profiling/lab1_*.ncu-rep`: Profiling data files
- `results/profiling/profiling_summary.txt`: Automated analysis

---

## 7. CUTLASS Baseline

### Purpose
**Phase 1.2.2**: Benchmark CUTLASS (CUDA Templates for Linear Algebra) to see state-of-the-art performance.

### How It Works

#### 7.1 CUTLASS

**What it is**:
- NVIDIA's template library for GEMM
- Provides highly optimized, customizable kernels
- Often matches or exceeds cuBLAS performance

**Why Benchmark It**:
- Shows what's possible with advanced optimizations
- Provides reference for our Phase 3 analysis
- Helps set realistic performance targets

#### 7.2 Running CUTLASS Profiler

```bash
# Build CUTLASS profiler
cd third_party/cutlass/build
cmake .. -DCUTLASS_NVCC_ARCHS=75
cmake --build . --target cutlass_profiler

# Run benchmarks
./tools/profiler/cutlass_profiler \
    --operation=gemm \
    --m=128,256,512,1024,2048,4096,8192 \
    --n=128,256,512,1024,2048,4096,8192 \
    --k=128,256,512,1024,2048,4096,8192 \
    --precision=f32 \
    --output=results/cutlass/cutlass_fp32.csv
```

#### 7.3 Results

- `results/cutlass/cutlass_fp32.csv`: FP32 GEMM results
- `results/cutlass/cutlass_f16tc.csv`: FP16 TensorCore results

---

## 8. Phase 1 Workflow

### Complete Execution Flow

1. **Run GPU Specs**:
   ```bash
   ./build/gpu_specs
   ```
   → Generates `results/gpu_specs.txt`

2. **Run Benchmarks**:
   ```bash
   ./build/benchmark_gemm
   ```
   → Generates `results/benchmark_results.csv`

3. **Profile Kernels** (optional):
   ```bash
   ncu --set full --export results/profiling/lab1_2048 \
       --kernel-name regex:op_mm_kernel \
       ./build_profile/benchmark_gemm
   ```

4. **Run CUTLASS** (optional):
   ```bash
   ./third_party/cutlass/build/tools/profiler/cutlass_profiler ...
   ```

5. **Generate Analysis**:
   ```bash
   python3 src/roofline_analysis.py
   ```
   → Generates plots and analysis report

### Automated Script

The `scripts/run_phase1_complete.sbatch` script runs all steps automatically:
```bash
sbatch scripts/run_phase1_complete.sbatch
```

---

## 9. Understanding the Results

### Performance Comparison

**Typical Results (T4 GPU)**:

| Kernel | Matrix Size | GFLOPS | % of cuBLAS |
|--------|-------------|--------|-------------|
| Lab-1 Tiled | 512×512 | ~300 | 20% |
| Lab-1 Tiled | 2048×2048 | ~1950 | 15% |
| cuBLAS SGEMM | 2048×2048 | ~13000 | 100% |
| cuBLAS TensorCore | 2048×2048 | ~90000 | - |

**Key Observations**:
1. Lab-1 performs poorly on small matrices (memory-bound)
2. Lab-1 improves on larger matrices (more compute per data)
3. Still far from cuBLAS (13-60% efficiency)
4. TensorCore provides massive speedup (5-7x)

### Roofline Analysis

**Small Matrices (128×128)**:
- Low AI (~10 FLOPS/Byte)
- Memory-bound region
- Limited by bandwidth, not compute
- **Optimization**: Better memory access patterns

**Large Matrices (4096×4096)**:
- High AI (~300 FLOPS/Byte)
- Compute-bound region
- Limited by compute, not bandwidth
- **Optimization**: More compute per data (TensorCores, better tiling)

### Gap Analysis

**Why is Lab-1 slow?**
1. **No software pipelining**: Computation and memory loads not overlapped
2. **Small tile size**: 32×32 may not be optimal
3. **No vectorization**: Not using 128-bit loads
4. **No TensorCores**: Missing 8x speedup opportunity
5. **Register usage**: May limit occupancy

**What to optimize in Phase 2?**
1. Implement TensorCore kernel (biggest win)
2. Increase tile size (better compute utilization)
3. Add software pipelining (overlap compute/memory)
4. Vectorize memory loads (better bandwidth)

---

## 10. Key Takeaways

### What Phase 1 Accomplishes

1. **Hardware Understanding**: Know your target GPU's capabilities
2. **Baseline Performance**: Measure current implementation
3. **Performance Targets**: Know what's possible (cuBLAS, CUTLASS)
4. **Bottleneck Identification**: Understand what's limiting performance
5. **Optimization Roadmap**: Know what to optimize next

### Metrics to Track

- **GFLOPS**: Raw performance
- **Efficiency**: % of peak performance
- **Bandwidth Utilization**: % of peak bandwidth
- **Occupancy**: % of maximum warps per SM
- **Arithmetic Intensity**: Memory-bound vs compute-bound

### Next Steps (Phase 2)

1. Implement TensorCore kernel (target: 40-60% of cuBLAS TensorCore)
2. Optimize tile sizes and memory access
3. Add software pipelining
4. Profile and iterate

---

## 11. Common Questions

**Q: Why is Lab-1 so much slower than cuBLAS?**
A: cuBLAS uses many advanced optimizations we haven't implemented yet: software pipelining, optimal tile sizes, vectorization, register tiling, etc.

**Q: Why does performance vary with matrix size?**
A: Small matrices are memory-bound (not enough compute to hide memory latency). Large matrices are compute-bound (more work per data loaded).

**Q: What's the difference between FP32 and TensorCore?**
A: TensorCores are specialized hardware units that can do 16×16×16 matrix multiplies very efficiently. They're 5-8x faster than regular FP32 cores for matrix operations.

**Q: How do I know if my kernel is memory-bound or compute-bound?**
A: Calculate arithmetic intensity. If AI < ridge point, it's memory-bound. If AI > ridge point, it's compute-bound. Check the roofline plot.

**Q: What's a good efficiency target?**
A: 70-80% of cuBLAS is excellent for a student implementation. 100% is very difficult (cuBLAS is highly optimized).

---

## Summary

Phase 1 establishes the foundation by:
1. **Understanding hardware**: GPU specs, theoretical limits
2. **Measuring baseline**: Lab-1 kernel performance
3. **Setting targets**: cuBLAS and CUTLASS performance
4. **Identifying gaps**: Where we're slow and why
5. **Planning optimizations**: What to implement in Phase 2

The tools and analysis from Phase 1 will be used throughout the project to measure progress and guide optimizations.

