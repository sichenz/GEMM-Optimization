# GEMM Optimization Project

Implementation and optimization of General Matrix Multiply (GEMM) kernels using GPU TensorCores on NYU Greene HPC cluster.

**Authors:** Sichen Zhong, Anh Dam  
**Platform:** NVIDIA Quadro RTX 8000 (Turing, Compute Capability 7.5)

---

## Overview

This project implements GEMM kernels using GPU TensorCores through the WMMA API. Our initial goal was to achieve 40-60% of cuBLAS TensorCore performance, but we ended up reaching about 5-8% with our best kernel. While this is lower than our target, we learned a lot about GPU programming and optimization.

GEMM is a fundamental operation in deep learning and takes up most of the training time. TensorCores can provide 5-7x speedup over regular FP32 computation, so understanding how to program them is useful for high-performance computing.

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
├── results/                      # Benchmark results and analysis
│   ├── benchmark_results.csv     # Latest benchmark data
│   ├── gpu_specs.txt            # GPU specifications
│   └── *.png                    # Performance plots
└── CMakeLists.txt               # Build configuration
```

---

## Performance Results

### Best Kernel: Balanced (Lab2_TensorCore_Balanced)

The Balanced kernel uses 4 warps per block instead of 8, which reduces register pressure and improves occupancy.

**Performance (4096×4096×4096):**
- **Balanced**: 4,569 GFLOPS (**5.17%** of cuBLAS TensorCore) - best
- Optimized: 2,565 GFLOPS (2.90% of cuBLAS)
- Baseline: 2,646 GFLOPS (2.99% of cuBLAS)
- **cuBLAS TensorCore**: 88,448 GFLOPS (100% baseline)

**Performance across different matrix sizes:**
- 1024×1024: 3,565 GFLOPS (6.7% of cuBLAS)
- 2048×2048: 4,804 GFLOPS (5.3% of cuBLAS)
- 4096×4096: 4,569 GFLOPS (5.2% of cuBLAS)
- 8192×8192: 4,493 GFLOPS (5.1% of cuBLAS)

### Why Balanced Works Better

We found that using 4 warps instead of 8 actually performs better because:
- Less register pressure means higher occupancy
- More blocks can run at the same time
- Better overall GPU utilization

Initially we thought more warps would be better, but the Optimized kernel with 8 warps actually performed slightly worse than the baseline. This was surprising and took some time to figure out - we realized it was due to register pressure limiting how many blocks could run concurrently.

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

### Optimizations We Tried

1. **Double Buffering**: Tried to overlap loading the next tile while computing the current one
2. **Larger Tile Sizes**: Attempted 64×64 blocks but hit shared memory limits (48KB)
3. **More Warps**: Tried 8 warps per block, but it didn't help due to register pressure
4. **Coalesced Memory Access**: Fixed the B matrix access pattern to improve memory bandwidth
5. **Reduced Register Pressure**: The Balanced kernel uses 4 warps which worked best

We also tried some other approaches like vectorized loads and reducing synchronization, but they didn't improve performance much or caused compilation issues.

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
# Run benchmark directly
./build/benchmark_gemm

# Or use SLURM script (recommended for Greene)
sbatch scripts/test_quick_benchmark.sbatch
```

### Generating Roofline Analysis

```bash
# Generate roofline plots and analysis
python3 src/roofline_analysis.py

# View generated files
ls -lh results/roofline_plot.png results/performance_comparison.png results/analysis_report.txt
```

---

## What We Learned

1. **WMMA API is more complex than expected**: It requires shared memory, has strict layout requirements (B matrix must be col-major), and needs careful warp-level coordination. The documentation wasn't always clear, so we had to figure some things out through trial and error.

2. **Correctness before optimization**: We spent a significant amount of time just getting the kernels to produce correct results. Small bugs can cause completely wrong outputs or crashes, so validation was crucial.

3. **Register pressure is a real issue**: We initially thought more warps would always be better, but using 8 warps actually hurt performance because of register pressure. This was counterintuitive but important to understand.

4. **Shared memory is limited**: We tried larger tile sizes but hit the 48KB shared memory limit per SM. This forced us to be more careful about memory usage.

5. **Profiling helps a lot**: Using Nsight Compute revealed that memory bandwidth was our main bottleneck, not compute. This guided our optimization efforts.

---

## Challenges

1. **WMMA API learning curve**: The documentation wasn't always clear, especially about matrix layouts. We had to experiment a lot to get things working correctly.

2. **Debugging GPU code**: Debugging is much harder than CPU code - no easy printf, and errors can be silent. We had to rely heavily on validation against cuBLAS to catch bugs.

3. **Performance tuning**: There are many factors that affect performance (register usage, occupancy, memory access patterns, etc.). It took a lot of iteration to understand what was actually helping.

4. **Shared memory limits**: We tried larger tiles but kept hitting the 48KB limit. Had to be careful about padding and buffer sizes.

5. **HPC environment**: Working on Greene had its own challenges - Singularity containers, SLURM job scheduling, and file transfer issues added complexity.

---

## Results

All benchmark results are in the `results/` directory:
- `results/benchmark_results.csv` - Complete benchmark data for all kernels and matrix sizes
- `results/gpu_specs.txt` - GPU specifications and performance characteristics
- `results/analysis_report.txt` - Performance analysis report
- `results/*.png` - Performance plots and roofline analysis
- `logs/test_benchmark.out` - Latest test run output with verification results

---

## References

- CUTLASS: https://github.com/NVIDIA/cutlass
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- Nsight Compute: https://developer.nvidia.com/nsight-compute
