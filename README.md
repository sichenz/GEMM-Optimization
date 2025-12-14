# GEMM Optimization Project

High-performance General Matrix Multiply (GEMM) implementation using GPU TensorCores on NYU Greene.

**Authors:** Sichen Zhong, Anh Dam  
**Platform:** NVIDIA Quadro RTX 8000 (Turing, Compute Capability 7.5)

---

## Overview

This project implements and optimizes GEMM kernels using GPU TensorCores via the WMMA API. The goal was to achieve 40-60% of cuBLAS TensorCore performance, though we ended up at around 5-8% with our best kernel.

GEMM is the core operation in deep learning and accounts for most of the training time. TensorCores provide 5-7x speedup over regular FP32 computation, so learning to program them is important for high-performance computing.

---

## Project Structure

```
GEMM-Optimization/
├── src/
│   ├── benchmark_gemm.cu          # Main benchmarking harness
│   ├── roofline_analysis.py      # Roofline model visualization
│   │
│   ├── ops/                      # CUDA kernel implementations
│   │   ├── op_mm.cuh            # Lab-1 tiled GEMM (FP32)
│   │   ├── op_mm_tensorcore.cuh # TensorCore GEMM baseline
│   │   ├── op_mm_tensorcore_optimized.cuh  # Optimized version (8 warps)
│   │   ├── op_mm_tensorcore_balanced.cuh   # Best performing kernel (4 warps)
│   │   └── op_elemwise.cuh      # Element-wise operations
│   │
│   └── utils/                    # Utility headers
│       ├── tensor.cuh           # Tensor data structure
│       └── check_error.cuh      # CUDA error checking
│
├── scripts/                      # SLURM job scripts
│   ├── test_quick_benchmark.sbatch  # Quick benchmark test
│   ├── profile_comparison.sbatch   # Profiling with Nsight Compute
│   └── run_roofline_analysis.sbatch # Generate roofline plots
│
├── results/                      # Benchmark results
│   ├── final/                   # Final benchmark results
│   ├── plots/                   # Generated plots (empty, for future use)
│   └── profiling/               # Profiling outputs (empty, for future use)
└── CMakeLists.txt               # Build configuration
```

---

## Performance Results

### Best Kernel: Balanced (Lab2_TensorCore_Balanced)

The Balanced kernel uses 4 warps per block instead of 8, which reduces register pressure and improves occupancy.

**Performance (4096×4096×4096):**
- **Balanced**: 4,591 GFLOPS (**5.18%** of cuBLAS TensorCore)
- HighPerf: 2,789 GFLOPS (3.15% of cuBLAS)
- Optimized: 2,570 GFLOPS (2.90% of cuBLAS)
- Baseline: 2,656 GFLOPS (3.00% of cuBLAS)
- **cuBLAS TensorCore**: 88,621 GFLOPS (100% baseline)

**Performance across sizes:**
- 1024×1024: 4,589 GFLOPS (8.55% of cuBLAS)
- 2048×2048: 4,844 GFLOPS (5.37% of cuBLAS)
- 4096×4096: 4,591 GFLOPS (5.18% of cuBLAS)
- 8192×8192: 4,524 GFLOPS (5.16% of cuBLAS)

### Why Balanced Works Better

The Balanced kernel uses 4 warps instead of 8, which:
- Reduces register pressure → higher occupancy
- Allows more blocks to run concurrently
- Better GPU utilization

Other kernels tried 8 warps but hit register pressure limits, especially for larger matrices.

---

## Implementation Details

### TensorCore GEMM Kernel

**Architecture:**
- 4 warps per block (128 threads) - Balanced kernel
- Each warp computes 16×16 output tile
- Each block computes 32×32 output (2×2 warp arrangement)
- FP16 input matrices, FP32 accumulation (mixed precision)
- Shared memory for tile loading (WMMA requirement)

**Algorithm:**
1. Each warp loads a 16×16 tile from A (row-major)
2. Each warp loads a 16×16 tile from B (transposed to col-major for WMMA)
3. WMMA computes: C_tile = A_tile × B_tile (using TensorCore hardware)
4. Accumulate over K dimension in chunks of 16
5. Store FP32 result to global memory

**Key Features:**
- Proper WMMA API usage with shared memory
- Correct matrix layout handling (row-major A, col-major B)
- Boundary checking for non-multiple-of-16 sizes
- Double buffering for better memory/compute overlap

### Optimizations Tried

1. **Double Buffering (2-Stage Pipeline)**: Overlap loading next tile with computing current tile
2. **Larger Tile Sizes**: Tried 64×64 and 64×128 blocks
3. **More Warps**: Tried 8 and 16 warps per block
4. **Coalesced Memory Access**: Optimized B matrix access pattern
5. **Reduced Register Pressure**: Balanced kernel uses 4 warps for better occupancy

---

## Building and Running

### Prerequisites
- CUDA Toolkit 12.x
- CMake 3.20+
- C++17 compiler
- Access to NYU Greene with GPU nodes

### Build Instructions

On Greene, use the Singularity container:

```bash
# Enter Singularity container
singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash

# Inside container
cd /scratch/$USER/GEMM-Optimization
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm
```

### Running Benchmarks

```bash
# Run comprehensive benchmark
./build/benchmark_gemm

# Or use SLURM script (recommended)
sbatch scripts/run_phase4_final.sbatch
```

### Generating Roofline Analysis

```bash
# Generate roofline plots and analysis
python3 src/roofline_analysis.py

# View generated files
ls -lh results/roofline_plot.png results/performance_comparison.png results/analysis_report.txt
```

---

## What I Learned

1. **WMMA API is tricky**: Requires shared memory, strict layout requirements (col-major for B), and warp-level coordination.

2. **Correctness first**: Spent a lot of time getting correctness right before optimizing. Small bugs can cause huge issues.

3. **Register pressure matters**: Using 8 warps caused register spilling and lower occupancy. 4 warps worked better.

4. **Tile sizes are important**: Larger tiles reduce kernel launch overhead, but need to fit in shared memory (48KB limit).

5. **Profiling is essential**: Need Nsight Compute to understand bottlenecks. Memory bandwidth, occupancy, and register usage all matter.

---

## Challenges

1. **WMMA API learning curve**: Documentation can be sparse, had to figure out layout requirements through trial and error.

2. **Debugging GPU code**: Hard to debug (no easy printf). Validation was key.

3. **Performance tuning**: Many factors affect performance. Iterative optimization needed.

4. **Shared memory limits**: Had to carefully manage shared memory to stay under 48KB per block.

---

## Results

All benchmark results are in the `results/` directory:
- `results/benchmark_results.csv` - Complete benchmark data
- `results/final/performance_summary.txt` - Final performance summary
- `results/final/comparison_table.txt` - Performance comparison

---

## References

- CUTLASS: https://github.com/NVIDIA/cutlass
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- Nsight Compute: https://developer.nvidia.com/nsight-compute
