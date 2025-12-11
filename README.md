# GEMM Optimization Project

High-performance General Matrix Multiply (GEMM) implementation and optimization study on NYU GREENE.

## Project Structure

```
GEMM-Optimization/
├── src/
│   ├── gpu_specs.cu              # GPU specifications collection tool
│   ├── benchmark_gemm.cu         # Main benchmarking harness
│   ├── roofline_analysis.py      # Performance analysis and visualization
│   ├── analyze_nsight_profile.py # Nsight Compute profiling analyzer
│   │
│   ├── ops/                      # CUDA operators
│   │   ├── op_mm.cuh            # Matrix multiplication (Lab-1 tiled GEMM)
│   │   ├── op_elemwise.cuh      # Element-wise operations
│   │   ├── op_reduction.cuh     # Reduction operations
│   │   └── op_cross_entropy.cuh # Cross-entropy loss
│   │
│   ├── utils/                    # Utility headers
│   │   ├── tensor.cuh           # Tensor data structure
│   │   └── check_error.cuh      # Error checking macros
│   │
│   └── baselines/                # Baseline implementations
│       └── cublas_bench.cu      # cuBLAS benchmarking tool
│
├── scripts/                      # SLURM job scripts
│   ├── run_phase1_complete.sbatch   # Master script for Phase 1
│
├── results/                      # Output directory (generated)
│   ├── gpu_specs.txt            # GPU specifications
│   ├── benchmark_results.csv    # Benchmark data
│   ├── roofline_plot.png        # Roofline visualization
│   ├── performance_comparison.png
│   ├── analysis_report.txt      # Detailed analysis
│   │
│   ├── profiling/               # Nsight Compute results
│   │   ├── profiling_*.log      # Profile data
│   │
│   └── cutlass/                 # CUTLASS benchmark results
│       └── cutlass_*.csv
│
├── third_party/                  # External dependencies
│   └── cutlass/                 # CUTLASS library (submodule)
│
├── build/                        # Build directory (generated)
├── build_profile/                # Build directory for Nsight profiling (generated)
├── logs/                         # SLURM job logs (generated)
├── CMakeLists.txt               # Build configuration
└── README.md                    # This file
```

## Phase 1: Foundation & Benchmarking

### Objectives
1. ✅ Document GPU specifications and theoretical performance
2. ✅ Benchmark Lab-1 tiled GEMM implementation
3. ✅ Establish cuBLAS and CUTLASS performance baselines
4. ✅ Profile kernel performance with Nsight Compute
5. ✅ Generate roofline model and performance analysis

### Quick Start

#### 1. Complete Phase 1 Analysis 
```bash
# Run everything: GPU specs, benchmarks, profiling, analysis
sbatch scripts/run_phase1_complete.sbatch

# Wait for completion, then review results
cat results/analysis_report.txt
```

### Phase 1 Key Results

**GPU Specifications (Quadro RTX 8000):**
- Compute Capability: 7.5 (Turing)
- Peak FP32: ~14.9 TFLOPS
- Peak FP16 TensorCore: 65 TFLOPS
- Memory Bandwidth: 624.1 GB/s
- FP32 Ridge Point: ~23.9 FLOPS/Byte
- FP16 Ridge Point: ~104.2 FLOPS/Byte

**Performance Summary:**
- Lab-1 Tiled GEMM: 306 - 1,952 GFLOPS (FP32)
- cuBLAS SGEMM: 512 - 13,207 GFLOPS (FP32)
- cuBLAS TensorCore: 504 - 91,280 GFLOPS (FP16→FP32)

**Efficiency:**
- Lab-1 vs cuBLAS: 13-60% (varies with matrix size)
- TensorCore Speedup: 5-7x over FP32 for large matrices

### Interpreting Results

**Roofline Plot** (`results/roofline_plot.png`)
- Shows achieved performance vs arithmetic intensity
- Identifies memory-bound vs compute-bound regions
- Compares Lab-1 against theoretical peaks

**Performance Comparison** (`results/performance_comparison.png`)
- GFLOPS vs matrix size
- Efficiency relative to cuBLAS
- Bandwidth utilization trends

**Analysis Report** (`results/analysis_report.txt`)
- Detailed metrics for each kernel
- Per-matrix-size breakdown
- Optimization recommendations

**Profiling Summary** (`results/profiling/profiling_summary.txt`)
- Occupancy, memory throughput, compute utilization
- Warp stall analysis
- Register and shared memory usage

## Building the Project

### Prerequisites
- CUDA Toolkit 12.x
- CMake 3.20+
- C++17 compiler
- Python 3.8+ (for analysis scripts)
  - matplotlib
  - pandas
  - numpy

### Build Commands
```bash
# Release build (optimized)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Profile build (with debug info)
mkdir -p build_profile && cd build_profile
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```

### Build Targets
- `gpu_specs` - GPU specification tool
- `benchmark_gemm` - Main benchmark suite
- `cublas_bench` - cuBLAS baseline benchmark
- `test` - Unit tests for tensor operations

## Implementation Details

### Lab-1 Tiled GEMM (`src/ops/op_mm.cuh`)
- **Algorithm**: Tiled matrix multiplication with shared memory
- **Tile Size**: 32×32 (TILE_DIM)
- **Memory Pattern**: Loads tiles into shared memory, computes partial results
- **Optimization**: Reduces global memory accesses via blocking

**Current Performance Characteristics:**
- Best for: Medium to large square matrices (2048×2048+)
- Peak Performance: ~2 TFLOPS (13-15% of theoretical FP32 peak)
- Bottleneck: Memory bandwidth for small matrices, compute utilization for large

### Benchmark Suite (`src/benchmark_gemm.cu`)
**Matrix Configurations:**
- Square: 128², 256², 512², 1024², 2048², 4096², 8192²
- Rectangular: 4096×256×1024, 1024×4096×512, 2048×512×2048, 512×2048×512

**Measured Metrics:**
- Execution time (milliseconds)
- GFLOPS (2×M×N×K operations)
- Memory bandwidth (GB/s)
- Efficiency (% of peak)

## Troubleshooting

**Build Errors:**
```bash
# Clean and rebuild
rm -rf build build_profile
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

**CUDA Out of Memory:**
- Reduce batch size in benchmarks
- Test smaller matrix sizes first

**Profiling Issues:**
- Ensure compute capability matches GPU (75 for RTX 8000)
- Use `--force-overwrite` to replace existing .ncu-rep files
- Check CUDA version compatibility

**Missing CUTLASS Results:**
```bash
# Reinitialize submodule
git submodule update --init --force --recursive
# Rebuild CUTLASS profiler
rm -rf third_party/cutlass/build
sbatch scripts/run_cutlass_profile.sbatch
```

## References

- [CUTLASS](https://github.com/NVIDIA/cutlass) - CUDA Templates for Linear Algebra Subroutines
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - GPU Kernel Profiler
- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) - Performance visualization

## License

See LICENSE file for details.

## Authors

- Sichen Zhong
- Anh Dam
