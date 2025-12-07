# GEMM Optimization Project

High-performance General Matrix Multiply (GEMM) implementation and optimization study using GPU TensorCores on NYU Greene.

**Authors:** Sichen Zhong, Anh Dam  
**Date:** December 2025  
**Platform:** NVIDIA Quadro RTX 8000 (Turing, Compute Capability 7.5)

---

## Project Overview

### Goal
Develop and optimize high-performance GEMM kernels using GPU TensorCores via WMMA API, targeting 40-60% of cuBLAS TensorCore performance.

### Motivation
GEMM is the core operation in deep learning (neural networks, transformers) and accounts for 80-90% of training time in many ML workloads. TensorCores provide 5-7x speedup over regular FP32 computation, so understanding how to program and optimize these specialized units is essential for high-performance computing and machine learning applications.

---

## Project Structure

```
GEMM-Optimization/
├── src/
│   ├── gpu_specs.cu              # GPU specifications collection
│   ├── benchmark_gemm.cu          # Main benchmarking harness
│   ├── generate_final_report.py  # Performance analysis and report generation
│   ├── roofline_analysis.py      # Roofline model visualization
│   │
│   ├── ops/                      # CUDA kernel implementations
│   │   ├── op_mm.cuh            # Lab-1 tiled GEMM (FP32)
│   │   ├── op_mm_tensorcore.cuh # TensorCore GEMM baseline
│   │   ├── op_mm_tensorcore_optimized.cuh  # Optimized (2-stage pipeline)
│   │   ├── op_mm_tensorcore_large_tile.cuh # Large tile variant
│   │   └── op_elemwise.cuh      # Element-wise operations
│   │
│   └── utils/                    # Utility headers
│       ├── tensor.cuh           # Tensor data structure
│       └── check_error.cuh      # CUDA error checking macros
│
├── scripts/                      # SLURM job scripts
│   ├── run_phase1_complete.sbatch  # Phase 1 comprehensive analysis
│   └── run_phase4_final.sbatch     # Phase 4 final benchmarking
│
├── CMakeLists.txt               # Build configuration
└── README.md                    # This file
```

---

## Project Phases

### Phase 1: Foundation & Benchmarking

In this phase, we established baselines and understood the hardware. We documented GPU specifications, benchmarked the Lab-1 tiled GEMM implementation, and established cuBLAS and CUTLASS performance baselines. We also generated roofline models to visualize performance characteristics.

**Key Results:**
- Lab-1 Tiled: 1,400-2,000 GFLOPS (9-13% of FP32 peak)
- cuBLAS SGEMM: 5,000-13,000 GFLOPS (33-87% of FP32 peak)
- cuBLAS TensorCore: 50,000-90,000 GFLOPS (5-7x speedup over FP32)

The main insight was that TensorCores provide significant speedup, and there's a lot of room for optimization in our baseline implementation.

### Phase 2: TensorCore Implementation

This was the core implementation phase. We implemented TensorCore GEMM using the WMMA API with FP16 inputs and FP32 accumulation (mixed precision). The implementation uses 4 warps per block, with each warp computing a 16x16 output tile. Each block computes a 32x32 output using a 2x2 warp arrangement.

**Key Results:**
- Baseline TensorCore: 2,648 GFLOPS (2.93% of cuBLAS TensorCore)
- Optimized (2-stage): 2,664 GFLOPS (2.94% of cuBLAS TensorCore)
- Large Tile: 2,650 GFLOPS (2.93% of cuBLAS TensorCore)

**Status:** The kernels are functionally correct and pass all validation tests, but performance optimization is still needed.

**Challenges we faced:**
- Learning the WMMA API was tricky - it requires shared memory (not local memory) and has strict layout requirements
- Getting correctness right took several iterations - we had to fix warp organization, matrix transpose indexing, and output writing
- Performance debugging was challenging - we prioritized correctness first, which meant using smaller, easier-to-debug tile sizes

### Phase 3: CUTLASS Analysis

We analyzed CUTLASS (NVIDIA's reference GEMM implementation) to understand how high-performance kernels are designed. This helped us identify why our performance was lower than target.

**Key Findings:**
- CUTLASS uses 128x128 block tiles (we use 32x32) - that's 4x larger
- CUTLASS uses 3-5 stage pipelines (we use 2-stage)
- CUTLASS uses 8 warps per block (we use 4)
- The performance gap is primarily due to these architectural differences, not just missing optimizations

**What this means:** To reach the target performance, we'd need to make fundamental architectural changes like increasing tile sizes and implementing more sophisticated pipelines. This is beyond the scope of this project but provides a clear path forward.

### Phase 4: Final Comprehensive Evaluation

We ran comprehensive benchmarks across all kernels and matrix sizes, validated correctness, and generated final performance reports. This phase confirmed that our kernels are correct but need significant optimization to reach the performance target.

---

## Final Performance Results

### Performance Summary (4096x4096x4096)

| Kernel | GFLOPS | Efficiency vs cuBLAS TC | Time (ms) |
|--------|--------|------------------------|-----------|
| cuBLAS TensorCore | 90,472 | 100.00% (baseline) | 1.52 |
| cuBLAS SGEMM | 12,960 | 14.32% | 10.61 |
| Our TensorCore (Optimized) | 2,664 | 2.94% | 51.60 |
| Our TensorCore (Baseline) | 2,648 | 2.93% | 51.91 |
| Our TensorCore (Large Tile) | 2,650 | 2.93% | 51.86 |
| Lab-1 Tiled | 1,947 | 2.15% | 70.61 |

### Key Metrics

- Average Efficiency: 3.10% of cuBLAS TensorCore
- Target: 40-60% of cuBLAS TensorCore
- Status: Target not achieved, but kernels are functionally correct

### Performance Analysis

**Why the performance gap?**

1. Small tile sizes: We use 32x32 blocks vs CUTLASS 128x128 (4x smaller). This means more kernel launches and higher overhead.

2. Limited pipelining: We use 2-stage pipelines vs CUTLASS 3-5 stage. This means less effective latency hiding.

3. Conservative design: We prioritized correctness over aggressive optimization. Smaller blocks are easier to debug and validate, but this limits performance.

4. Architectural differences: Fundamental design choices (tile sizes, number of warps) limit performance compared to highly optimized libraries.

**What works:**
- All kernels pass correctness validation (100%)
- TensorCore implementation demonstrates proper WMMA API usage
- Code is well-documented and maintainable
- Solid foundation for further optimization

---

## Building and Running

### Prerequisites
- CUDA Toolkit 12.x
- CMake 3.20+
- C++17 compiler
- Access to NYU Greene with GPU nodes

### Build Instructions

On Greene, you need to use the Singularity container:

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

### Generating Reports

```bash
# Generate final performance report
python3 src/generate_final_report.py

# View results
cat results/final/performance_summary.txt
cat results/final/comparison_table.txt
```

---

## Implementation Details

### TensorCore GEMM Kernel

**Architecture:**
- 4 warps per block (128 threads)
- Each warp computes 16x16 output tile
- Each block computes 32x32 output (2x2 warp arrangement)
- FP16 input matrices, FP32 accumulation (mixed precision)
- Shared memory for tile loading (WMMA requirement)

**Algorithm:**
1. Each warp loads a 16x16 tile from A (row-major)
2. Each warp loads a 16x16 tile from B (transposed to col-major for WMMA)
3. WMMA computes: C_tile = A_tile × B_tile (using TensorCore hardware)
4. Accumulate over K dimension in chunks of 16
5. Store FP32 result to global memory

**Key Features:**
- Proper WMMA API usage with shared memory
- Correct matrix layout handling (row-major A, col-major B)
- Boundary checking and zero-padding for non-multiple-of-16 sizes
- Efficient shared memory usage within 48KB limit

### Optimizations Attempted

**1. Double Buffering (2-Stage Pipeline):**
- Software pipelining: overlap loading next tile with computing current tile
- Result: +0.4% improvement (minimal - memory wasn't the main bottleneck)

**2. Large Tile Size (64x64):**
- Increased block output from 32x32 to 64x64
- Result: No improvement (possibly shared memory pressure)

**3. 3-Stage Pipeline:**
- Attempted 3-stage pipeline for better latency hiding
- Result: Performance bug (54% slower) - disabled for now

---

## Key Learnings

### What We Learned

1. Hardware understanding is critical: Understanding GPU architecture (TensorCores, memory hierarchy) is essential. TensorCores provide massive speedup (5-7x) when used correctly, and the memory hierarchy (global → shared → registers) is key to performance.

2. Correctness first: Performance is meaningless if results are wrong. Validation is essential at every step. Small bugs can cause huge correctness issues.

3. WMMA API complexity: The WMMA API has strict layout requirements (col-major for B matrix), requires shared memory (not local memory), and needs warp-level coordination (32 threads work together).

4. Performance vs correctness trade-offs: Sometimes correctness fixes hurt performance. We needed to balance both concerns. Optimization can come after correctness is established.

5. Optimization is hard: Many factors affect performance (tile sizes, pipelining, memory access). We need profiling tools to identify bottlenecks. Iterative optimization is necessary, and fundamental architectural changes may be needed.

### Challenges Overcome

1. WMMA API learning curve: Documentation can be sparse, layout requirements not always clear. We used trial and error to get it right.

2. Debugging GPU code: GPU code is hard to debug (no easy printf). Validation was key. We needed a systematic approach.

3. Performance tuning: Many factors affect performance. We need profiling tools and iterative optimization.

---

## Project Status

### Completed
- Phase 1: Foundation & Benchmarking
- Phase 2: TensorCore Implementation (correct and validated)
- Phase 3: CUTLASS Analysis
- Phase 4: Final Comprehensive Evaluation

### Current State
- Correctness: 100% validated (all tests passing)
- Performance: 3.1% of cuBLAS TensorCore (target: 40-60%)
- Code Quality: Well-documented and maintainable
- Integration: Complete benchmark suite

### Future Work
To reach the target performance, significant architectural improvements are needed:
1. Increase tile sizes to 128x128 blocks (requires careful shared memory management)
2. Implement 3-5 stage pipelines (complex but high impact)
3. Add vectorized memory operations
4. Optimize shared memory layouts to avoid bank conflicts
5. Profile with Nsight Compute to identify specific bottlenecks

---

## Results Location

All benchmark results and analysis are in the `results/` directory:
- `results/benchmark_results.csv` - Complete benchmark data
- `results/final/performance_summary.txt` - Final performance summary
- `results/final/comparison_table.txt` - Performance comparison table

---

## References

- CUTLASS: https://github.com/NVIDIA/cutlass - CUDA Templates for Linear Algebra Subroutines
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- Nsight Compute: https://developer.nvidia.com/nsight-compute - GPU Kernel Profiler
- Roofline Model: https://en.wikipedia.org/wiki/Roofline_model - Performance visualization

---

## License

See LICENSE file for details.

---

**Project Repository:** https://github.com/sichenz/GEMM-Optimization  
**Branches:**
- `phase1` - Phase 1 completion
- `phase2` - Phase 2 completion  
- `phase3` - Phase 3 completion
- `phase4` - Final submission (current)
