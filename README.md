# GEMM Optimization with TensorCores

High-performance matrix multiplication (GEMM) implementation using NVIDIA TensorCores on the NYU Greene HPC cluster.

**Authors:** Sichen Zhong, Anh Dam  
**Platform:** NVIDIA Quadro RTX 8000 (Turing Architecture, Compute Capability 7.5)  
**Course:** NYU High Performance Machine Learning

---

## Overview

This project implements and optimizes General Matrix Multiply (GEMM) kernels using GPU TensorCores through NVIDIA's WMMA API. We explored various optimization strategies and achieved **5.2% of cuBLAS TensorCore performance** at large matrix sizes (4096×4096), with our best kernel reaching **6.7% efficiency** at smaller scales (1024×1024).

While we initially targeted 40-60% of cuBLAS performance, the final results taught us valuable lessons about GPU programming complexity, the sophistication of vendor-optimized libraries, and the critical importance of factors like register pressure and occupancy.

### Why GEMM Matters

Matrix multiplication is the computational backbone of deep learning, consuming 80-90% of training time in modern neural networks. TensorCores provide specialized hardware acceleration, offering theoretical speedups of 5-8x over standard FP32 computation through mixed-precision operations (FP16 input, FP32 accumulation).

---

## Key Results

### Performance Summary (4096×4096×4096)

| Kernel | GFLOPS | % of cuBLAS TC | Speedup vs FP32 |
|--------|--------|----------------|-----------------|
| **Lab2_TensorCore_Balanced** | **4,619** | **5.2%** | **3.1×** |
| Lab2_TensorCore_Optimized | 2,566 | 2.9% | 1.9× |
| Lab2_TensorCore (Baseline) | 2,650 | 3.0% | 2.0× |
| Lab1_Tiled (FP32) | 1,953 | 2.2% | 1.0× (baseline) |
| cuBLAS SGEMM (FP32) | 13,309 | 15.0% | 6.8× |
| **cuBLAS TensorCore (FP16)** | **92,137** | **100%** | **47×** |

### Performance Across Matrix Sizes

Our **Balanced** kernel showed consistent performance improvements:

| Matrix Size | GFLOPS | % of cuBLAS TC | % of Peak FP16 |
|-------------|--------|----------------|----------------|
| 1024×1024 | 4,472 | 6.7% | 3.7% |
| 2048×2048 | 4,813 | 5.4% | 4.0% |
| 4096×4096 | 4,619 | 5.2% | 3.9% |
| 8192×8192 | 4,674 | 5.1% | 3.9% |

**Key Insight:** Performance stabilizes at ~4,600 GFLOPS for large matrices, suggesting we're hitting a consistent bottleneck (likely memory bandwidth or instruction-level inefficiency).

---

## Technical Implementation

### Architecture Overview

```
Block Organization (Balanced Kernel):
┌─────────────────────────────────┐
│  Thread Block (32×32 output)    │
│  ┌──────────┬──────────┐        │
│  │ Warp 0   │ Warp 1   │        │
│  │ (16×16)  │ (16×16)  │        │
│  ├──────────┼──────────┤        │
│  │ Warp 2   │ Warp 3   │        │
│  │ (16×16)  │ (16×16)  │        │
│  └──────────┴──────────┘        │
│  4 warps × 32 threads = 128     │
└─────────────────────────────────┘
```

### Kernel Variants

#### 1. **Lab2_TensorCore (Baseline)**
- **Configuration:** 4 warps per block, 32×32 output tile
- **Features:** Basic WMMA implementation with shared memory tiling
- **Performance:** 2,650 GFLOPS @ 4096×4096
- **Learning:** Established correct WMMA usage patterns

#### 2. **Lab2_TensorCore_Optimized**
- **Configuration:** 8 warps per block, 64×32 output tile
- **Features:** Double buffering, larger tile sizes
- **Performance:** 2,566 GFLOPS @ 4096×4096 (worse than baseline!)
- **Learning:** More warps ≠ better performance; discovered register pressure issues

#### 3. **Lab2_TensorCore_Balanced** ⭐ Best Performer
- **Configuration:** 4 warps per block, 32×32 output tile
- **Features:** 
  - Optimized memory access patterns (coalesced loads)
  - Double buffering for compute/memory overlap
  - Reduced register pressure vs. Optimized
  - Precomputed address calculations
- **Performance:** 4,619 GFLOPS @ 4096×4096 (**1.74× baseline**)
- **Key Innovation:** Balances occupancy and register usage

### Why "Balanced" Wins

The counterintuitive result that 4 warps outperformed 8 warps revealed critical insights:

1. **Register Pressure:** More warps per block = more registers per SM
2. **Occupancy Impact:** High register usage limits concurrent blocks
3. **Sweet Spot:** 4 warps provided optimal balance between parallelism and resource availability

---

## Implementation Challenges

### 1. WMMA API Complexity
**Challenge:** WMMA requires precise matrix layouts (row-major A, column-major B) and shared memory staging.

**Solution:** Implemented careful transpose handling during B matrix loading:
```cuda
// Load B in column-major order for WMMA
int j = load_idx % WMMA_N;  // Column varies fastest
int i = load_idx / WMMA_N;  // Row varies slower
smem_b[warpId][j * WMMA_K + i] = Index(B, row, col);
```

### 2. Memory Coalescing
**Challenge:** Naive loading patterns caused uncoalesced global memory accesses.

**Solution:** Restructured load patterns so consecutive threads access consecutive memory locations, improving bandwidth utilization by ~30%.

### 3. Register Pressure
**Challenge:** 8-warp kernel unexpectedly underperformed due to register spilling.

**Solution:** Profiling with `nvcc --ptxas-options=-v` revealed 96 registers/thread. Reducing to 4 warps improved occupancy from 25% to 50%, enabling more concurrent blocks.

### 4. Correctness Validation
**Challenge:** Silent numerical errors were hard to debug without proper validation.

**Solution:** Implemented comprehensive validation against cuBLAS with configurable tolerances (1e-2 for FP16 accumulation noise).

### 5. Shared Memory Bank Conflicts
**Challenge:** Simultaneous shared memory accesses to the same bank caused serialization.

**Solution:** Added 8-element padding to shared memory arrays:
```cuda
__shared__ __half smem_a[4][WMMA_M * WMMA_K + 8];  // +8 padding
```

---

## Project Structure
```
GEMM-Optimization/
├── src/
│   ├── benchmark_gemm.cu              # Main benchmarking harness
│   ├── gpu_specs.cu                   # GPU specifications collector
│   ├── roofline_analysis.py           # Performance visualization
│   │
│   ├── baselines/                     # Reference implementations
│   │   └── cublas_bench.cu            # cuBLAS performance baseline
│   │
│   ├── ops/                           # CUDA kernel implementations
│   │   ├── op_mm.cuh                  # Lab-1 tiled GEMM (FP32 baseline)
│   │   ├── op_mm_tensorcore.cuh       # TensorCore baseline (3.0% efficiency)
│   │   ├── op_mm_tensorcore_optimized.cuh  # 8-warp variant (2.9% efficiency)
│   │   ├── op_mm_tensorcore_balanced.cuh   # Best kernel (5.2% efficiency)
│   │   ├── op_elemwise.cuh            # Element-wise operations
│   │   ├── op_reduction.cuh           # Reduction operations
│   │   └── op_cross_entropy.cuh       # Cross-entropy loss
│   │
│   └── utils/                         # Helper utilities
│       ├── tensor.cuh                 # Tensor data structure
│       └── check_error.cuh            # CUDA error checking macros
│
├── scripts/                           # SLURM job scripts for Greene
│   ├── test_quick_benchmark.sbatch    # Quick validation run
│   ├── profile_comparison.sbatch      # Nsight Compute profiling
│   └── run_roofline_analysis.sbatch   # Performance analysis
│
├── results/                           # Benchmark outputs
│   ├── benchmark_results.csv          # Raw performance data
│   ├── gpu_specs.txt                  # GPU hardware specifications
│   ├── analysis_report.txt            # Detailed performance analysis
│   ├── roofline_plot_fp16.png         # TensorCore roofline plot
│   ├── roofline_plot_fp32.png         # FP32 roofline plot
│   ├── performance_comparison.png     # Efficiency comparison charts
│   │
│   └── profiling/                     # Nsight Compute profiling data
│       └── cutlass_4096.txt           # CUTLASS reference profile
│
├── logs/                              # Execution logs
│   ├── test_benchmark.out             # Latest benchmark run
│   ├── profile_comparison.out         # Profiling job output
│   ├── roofline_analysis.out          # Analysis script output
│   └── test_output.txt                # Detailed test results
│
├── third_party/                       # External dependencies
│   └── cutlass/                       # NVIDIA CUTLASS library (submodule)
│
├── .gitignore                         # Git ignore patterns
├── .gitmodules                        # Git submodule configuration
├── CMakeLists.txt                     # Build configuration
└── README.md                          # This file
```

---

## Building and Running

### Prerequisites
- CUDA Toolkit 12.x
- CMake 3.20+
- C++17 compiler
- NYU Greene HPC access (or similar NVIDIA GPU cluster)

### Build Instructions

On Greene with Singularity:

```bash
# Enter container
singularity exec --nv \
    --overlay /scratch/$USER/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash

# Build
cd /scratch/$USER/GEMM-Optimization
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm
```

### Running Benchmarks

```bash
# Quick test (interactive)
./build/benchmark_gemm

# Full benchmark with SLURM (recommended)
sbatch scripts/test_quick_benchmark.sbatch

# Profile with Nsight Compute
sbatch scripts/profile_comparison.sbatch

# Generate analysis plots
sbatch scripts/run_roofline_analysis.sbatch
```

### Output Files

- `results/benchmark_results.csv` - Performance data for all kernels
- `results/roofline_plot_fp16.png` - TensorCore performance visualization
- `results/performance_comparison.png` - Efficiency comparison charts
- `logs/test_benchmark.out` - Execution log with validation results

---

## Performance Analysis

### Roofline Model Insights

Our kernels operate well below the hardware's roofline, indicating room for improvement:

- **Peak FP16 TensorCore:** 119.4 TFLOPS
- **Achieved (Balanced):** 4.6 TFLOPS (3.9% of peak)
- **cuBLAS TensorCore:** 92.1 TFLOPS (77% of peak)

**Bottleneck Analysis:**
- Achieved bandwidth: ~3.4 GB/s
- Peak bandwidth: 624 GB/s (0.5% utilization!)
- **Conclusion:** Likely instruction-level inefficiency or suboptimal memory access patterns, not raw bandwidth

### Optimization Journey

| Optimization Attempt | Result | Key Learning |
|---------------------|--------|--------------|
| Baseline WMMA | 2,650 GFLOPS | Correct but inefficient |
| 8 warps + larger tiles | 2,566 GFLOPS ↓ | Register pressure hurts |
| Double buffering | Minimal impact | Memory not the bottleneck |
| Coalesced loads + 4 warps | **4,619 GFLOPS** ↑ | Balance is key |

---

## Key Learnings

### 1. Vendor Libraries Are Highly Optimized
cuBLAS represents decades of optimization work. Our 5% efficiency demonstrates the enormous gap between academic implementations and production-quality code.

### 2. More Resources ≠ Better Performance
Our 8-warp kernel performed worse than the 4-warp version due to register pressure limiting occupancy. **Occupancy × ILP** matters more than raw thread count.

### 3. WMMA API Has Steep Learning Curve
- Strict layout requirements (row-major A, col-major B)
- Mandatory shared memory staging
- Fragment management complexity
- Limited documentation and error messages

### 4. Validation Is Critical
FP16 operations accumulate numerical error. Without comparing to cuBLAS (tolerance ≤1e-2), we wouldn't have caught several subtle bugs in our matrix loading logic.

### 5. Profiling Tools Are Essential
Nsight Compute revealed that our "optimized" kernel had 96 registers/thread, causing occupancy drops. Without profiling, this would have been impossible to diagnose.

### 6. Memory Access Patterns Matter Greatly
Restructuring B matrix loads for coalescing improved performance by ~30%, despite not being bandwidth-bound. Every wasted transaction hurts.

---

## Future Improvements

Given more time, we would explore:

1. **Software Pipelining:** Overlap WMMA compute with next tile loads using async copy
2. **Warp Specialization:** Dedicate warps to loading while others compute
3. **Persistent Kernels:** Keep data in registers across multiple output tiles
4. **Multi-stage Pipelines:** 3+ buffers to maximize overlap (like CUTLASS)
5. **Better Instruction Scheduling:** Hand-optimize PTX to reduce stalls
6. **Rectangle Tile Tuning:** 32×64 or 64×32 tiles might balance better
7. **CUTLASS Integration:** Learn from NVIDIA's state-of-the-art templates

---

## References

- [NVIDIA CUTLASS Library](https://github.com/NVIDIA/cutlass) - State-of-the-art GEMM templates
- [CUDA Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) - Official WMMA documentation
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/) - Optimized BLAS implementation
- [Dissecting the NVIDIA Volta GPU Architecture](https://arxiv.org/abs/1804.06826) - TensorCore deep dive

---

## Acknowledgments

- **NYU Greene HPC Team** - Compute resources and support
- **Course Instructors** - Guidance on GPU optimization techniques
- **NVIDIA CUTLASS Team** - Open-source reference implementations

---

## License

This project is educational work completed for NYU coursework. Code is provided as-is for reference purposes.
