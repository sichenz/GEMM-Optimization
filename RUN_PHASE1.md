# How to Run Phase 1 - Complete Guide

This guide shows you how to run Phase 1 from scratch to generate all reports and results.

## Prerequisites

- CUDA Toolkit 12.x installed
- CMake 3.20+
- C++17 compiler (g++ or clang++)
- Python 3.8+ with packages: `matplotlib`, `pandas`, `numpy`
- NVIDIA GPU with CUDA support
- Access to NYU GREENE cluster (for SLURM script) OR local GPU setup

---

## Option 1: Run on NYU GREENE (Recommended)

If you have access to NYU GREENE cluster, use the automated SLURM script:

### Step 0: Push Code to Git (On Your Laptop)

**Important**: Before running on GREENE, make sure you've pushed your latest code:

```bash
# On your local laptop
git add .
git commit -m "Phase 1 code ready"
git push origin main
```

### Step 1: Pull Latest Code on GREENE

```bash
# SSH into GREENE
ssh <your-netid>@greene.hpc.nyu.edu

# Navigate to your repo (or clone if first time)
cd /scratch/$USER/GEMM-Optimization

# Pull latest changes
git pull origin main

# Update submodules if needed
git submodule update --init --recursive
```

### Step 2: Submit the Job

```bash
# Make sure you're in the repo directory
cd /scratch/$USER/GEMM-Optimization

# Submit the job
sbatch scripts/run_phase1_complete.sbatch
```

**See `GREENE_WORKFLOW.md` for detailed workflow instructions!**

### Step 2: Monitor Progress

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/phase1_complete.out
```

### Step 3: Check Results

After the job completes (usually 1-2 hours), check:

```bash
# View GPU specifications
cat results/gpu_specs.txt

# View benchmark results
cat results/benchmark_results.csv

# View analysis report
cat results/analysis_report.txt

# View plots (if you have X11 forwarding or download them)
ls -lh results/*.png
```

---

## Option 2: Run Locally (Manual Steps)

If you're running on a local machine with GPU:

### Step 1: Install Python Dependencies

```bash
pip install matplotlib pandas numpy
```

### Step 2: Build the Project

```bash
cd /path/to/GEMM-Optimization

# Clean any old builds
rm -rf build build_profile

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ..
```

### Step 3: Run GPU Specifications

```bash
# Create results directory if it doesn't exist
mkdir -p results

# Run GPU specs collection
./build/gpu_specs

# Check output
cat results/gpu_specs.txt
```

**Expected Output**: GPU name, compute capability, memory specs, peak TFLOPS, etc.

### Step 4: Run Benchmarks

```bash
# Run benchmark suite (takes 10-30 minutes depending on GPU)
./build/benchmark_gemm

# Check results
head -20 results/benchmark_results.csv
```

**Expected Output**: CSV file with columns:
- Kernel (Lab1_Tiled, cuBLAS_SGEMM, cuBLAS_HGEMM_TensorCore)
- DType (FP32, FP16)
- M, N, K (matrix dimensions)
- Time_ms (execution time)
- GFLOPS (performance)
- Bandwidth_GB_s (memory bandwidth)

### Step 5: Generate Analysis and Visualizations

```bash
# Run roofline analysis script
python3 src/roofline_analysis.py
```

**Expected Output**:
- `results/roofline_plot.png` - Roofline model visualization
- `results/performance_comparison.png` - Performance comparison charts
- `results/analysis_report.txt` - Detailed text report

### Step 6: (Optional) Profile with Nsight Compute

If you have Nsight Compute installed:

```bash
# Build profiling version
mkdir -p build_profile
cd build_profile
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8 benchmark_gemm
cd ..

# Profile a specific matrix size (e.g., 2048x2048)
ncu --set full \
    --export results/profiling/lab1_2048 \
    --kernel-name regex:op_mm_kernel \
    --launch-skip 4 \
    --launch-count 1 \
    ./build_profile/benchmark_gemm

# Analyze profiling results
python3 src/analyze_nsight_profile.py
```

### Step 7: (Optional) Run CUTLASS Benchmarks

If you want to compare against CUTLASS:

```bash
# Initialize CUTLASS submodule if needed
git submodule update --init --recursive

# Build CUTLASS profiler
mkdir -p third_party/cutlass/build
cd third_party/cutlass/build

# Detect your GPU architecture (e.g., 75 for T4, 70 for V100)
GPU_ARCH=75  # Change this to match your GPU
cmake .. -DCUTLASS_NVCC_ARCHS=${GPU_ARCH} -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target cutlass_profiler -j8

cd ../../..

# Run CUTLASS FP32 benchmarks
./third_party/cutlass/build/tools/profiler/cutlass_profiler \
    --operation=gemm --providers=cutlass \
    --m=128,256,512,1024,2048,4096,8192 \
    --n=128,256,512,1024,2048,4096,8192 \
    --k=128,256,512,1024,2048,4096,8192 \
    --precision=f32 --accumulator-type=f32 \
    --disposition=equal \
    --output=results/cutlass/cutlass_fp32.csv

# Run CUTLASS FP16 TensorCore benchmarks (if GPU supports it)
./third_party/cutlass/build/tools/profiler/cutlass_profiler \
    --operation=gemm --providers=cutlass --op_class=tensorop \
    --m=128,256,512,1024,2048,4096,8192 \
    --n=128,256,512,1024,2048,4096,8192 \
    --k=128,256,512,1024,2048,4096,8192 \
    --precision=f16 --accumulator-type=f32 \
    --disposition=equal \
    --output=results/cutlass/cutlass_f16tc.csv
```

---

## Understanding the Results

### 1. GPU Specifications (`results/gpu_specs.txt`)

Contains:
- GPU model and compute capability
- Peak FP32 TFLOPS
- Peak FP16 TensorCore TFLOPS
- Memory bandwidth
- Ridge points (arithmetic intensity thresholds)

**Use this to**: Understand your hardware limits and calculate efficiency.

### 2. Benchmark Results (`results/benchmark_results.csv`)

CSV file with performance data for each kernel and matrix size.

**Key columns**:
- `GFLOPS`: Performance in Giga-FLOPS
- `Bandwidth_GB_s`: Memory bandwidth utilization
- Compare `Lab1_Tiled` vs `cuBLAS_SGEMM` to see the gap

**Use this to**: See raw performance numbers and identify which matrix sizes need optimization.

### 3. Roofline Plot (`results/roofline_plot.png`)

Visualization showing:
- X-axis: Arithmetic Intensity (FLOPS/Byte)
- Y-axis: Performance (GFLOPS)
- Roofline curve: Theoretical peak performance
- Points: Actual kernel performance

**Use this to**: 
- Identify if kernels are memory-bound or compute-bound
- See how close you are to theoretical limits
- Find optimization opportunities (points far below roofline)

### 4. Performance Comparison (`results/performance_comparison.png`)

Two plots:
- **Left**: GFLOPS vs matrix size for all kernels
- **Right**: Efficiency (% of cuBLAS) vs matrix size

**Use this to**: See where Lab-1 performs well/poorly compared to cuBLAS.

### 5. Analysis Report (`results/analysis_report.txt`)

Text summary with:
- Average/max/min GFLOPS for each kernel
- Efficiency calculations
- Bandwidth utilization
- Arithmetic intensity analysis

**Use this to**: Get a quick summary of performance without looking at plots.

### 6. Profiling Summary (`results/profiling/profiling_summary.txt`)

If you ran Nsight Compute profiling:
- Occupancy metrics
- Memory throughput
- Compute utilization
- Bottleneck analysis
- Optimization recommendations

**Use this to**: Understand why your kernel is slow and what to optimize.

---

## Quick Start (Minimal Run)

If you just want the basic results quickly:

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..

# Run essentials
mkdir -p results
./build/gpu_specs
./build/benchmark_gemm
python3 src/roofline_analysis.py

# View results
cat results/analysis_report.txt
open results/roofline_plot.png  # macOS
# or: xdg-open results/roofline_plot.png  # Linux
```

---

## Troubleshooting

### Build Errors

```bash
# Clean and rebuild
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### CUDA Out of Memory

- Reduce matrix sizes in `benchmark_gemm.cu`
- Or test smaller sizes first

### Python Import Errors

```bash
pip install --upgrade matplotlib pandas numpy
```

### Nsight Compute Not Found

- Skip profiling step (it's optional)
- Or install Nsight Compute from NVIDIA

### CUTLASS Build Fails

- Check GPU architecture matches: `nvidia-smi --query-gpu=compute_cap --format=csv`
- Use correct architecture: `-DCUTLASS_NVCC_ARCHS=75` (for T4) or `70` (for V100)

---

## Expected Runtime

- **GPU Specs**: < 1 second
- **Benchmarks**: 10-30 minutes (depends on GPU)
- **Analysis**: < 10 seconds
- **Profiling**: 5-10 minutes per size (optional)
- **CUTLASS**: 30-60 minutes (optional)

**Total**: ~1-2 hours for complete Phase 1

---

## Next Steps After Phase 1

1. Review `results/analysis_report.txt` for optimization opportunities
2. Check `results/roofline_plot.png` to see where you're memory-bound vs compute-bound
3. Compare Lab-1 performance vs cuBLAS to set targets
4. Proceed to Phase 2: Implement TensorCore kernel

---

## Files Generated

After running Phase 1, you should have:

```
results/
├── gpu_specs.txt                    # GPU specifications
├── benchmark_results.csv            # All benchmark data
├── analysis_report.txt              # Text summary
├── roofline_plot.png               # Roofline visualization
├── performance_comparison.png       # Performance charts
├── profiling/                       # (Optional) Nsight Compute data
│   ├── lab1_512.ncu-rep
│   ├── lab1_2048.ncu-rep
│   └── profiling_summary.txt
└── cutlass/                        # (Optional) CUTLASS results
    ├── cutlass_fp32.csv
    └── cutlass_f16tc.csv
```

All of these are used to understand your baseline performance and plan Phase 2 optimizations!

