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
