# Testing Guide for Optimized GEMM Kernels

This guide provides step-by-step instructions to test the optimized kernels both locally and on NYU Greene HPC.

## Prerequisites

### Local Testing
- CUDA Toolkit (11.0+)
- NVIDIA GPU with compute capability 7.0+ (V100, RTX series, etc.)
- CMake 3.20+

### Greene HPC
- NYU Greene account
- Access to GPU nodes (H100 or RTX 8000)

---

## Local Testing

### Step 1: Check CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU
nvidia-smi
```

### Step 2: Build the Project

```bash
# Navigate to project directory
cd /path/to/GEMM-Optimization

# Clean previous build
rm -rf build
mkdir build
cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm

# If build succeeds, you'll see:
# [100%] Built target benchmark_gemm
```

### Step 3: Run Benchmarks

```bash
# Run full benchmark suite (takes 5-10 minutes)
cd ..
./build/benchmark_gemm > results/benchmark_output.txt 2>&1

# Or run and see output in real-time
./build/benchmark_gemm | tee results/benchmark_output.txt
```

### Step 4: Check Results

```bash
# View benchmark output
cat results/benchmark_output.txt

# Check CSV results
cat results/benchmark_results.csv

# Generate summary report
python3 src/generate_final_report.py
cat results/final/performance_summary.txt
```

### Step 5: Quick Test (Small Matrix)

```bash
# For quick testing, you can modify the benchmark to test only one size
# Or just wait for the first few results in the full benchmark
```

---

## Greene HPC Testing

### Step 1: Connect to Greene

```bash
# SSH to Greene login node
ssh your_netid@greene.hpc.nyu.edu

# Navigate to your project directory
cd /scratch/$USER/GEMM-Optimization
```

### Step 2: Update SLURM Script (if needed)

The existing script `scripts/run_phase4_final.sbatch` should work, but check:

```bash
# Check if GPU type matches (H100 vs RTX 8000)
cat scripts/run_phase4_final.sbatch | grep gres

# If you need H100, change line 10 to:
# #SBATCH --gres=gpu:h100:1
```

### Step 3: Submit Job

```bash
# Make sure you're in the project root
cd /scratch/$USER/GEMM-Optimization

# Submit the job
sbatch scripts/run_phase4_final.sbatch

# You'll see output like:
# Submitted batch job 12345678
```

### Step 4: Monitor Job

```bash
# Check job status
squeue -u $USER

# Or use the helper script
bash scripts/check_job_status.sh

# View output in real-time (while job is running)
tail -f logs/phase4_final.out
```

### Step 5: Check Results (After Job Completes)

```bash
# View full output
cat logs/phase4_final.out

# Check results directory
ls -lh results/final/

# View performance summary
cat results/final/performance_summary.txt

# View comparison table
cat results/final/comparison_table.txt

# View CSV data
head -20 results/final/benchmark_results.csv
```

### Step 6: Compare Performance

```bash
# Look for the new "Lab2_TensorCore_HighPerf" kernel in results
grep "HighPerf" results/final/benchmark_results.csv

# Compare GFLOPS
grep "4096,4096,4096" results/final/benchmark_results.csv | \
    awk -F',' '{printf "%-30s %10.2f GFLOPS\n", $1, $6}'
```

---

## Manual Testing on Greene (Interactive)

If you want to test interactively:

```bash
# Request an interactive GPU node
srun --gres=gpu:h100:1 --time=1:00:00 --mem=32GB --pty bash

# Enter Singularity container
singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash

# Inside container
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

# Build
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm

# Run quick test (first few matrix sizes)
cd ..
./build/benchmark_gemm 2>&1 | head -100

# Exit container when done
exit
```

---

## What to Look For

### Performance Metrics

1. **GFLOPS**: Should be 2-8× higher than before
   - Old: ~2,600 GFLOPS for 4096×4096
   - Expected: ~8,000-20,000 GFLOPS

2. **Time**: Should be 2-8× faster
   - Old: ~51 ms for 4096×4096
   - Expected: ~6-25 ms

3. **Efficiency vs cuBLAS**: Should improve from ~3% to ~10-25%

### Kernel Comparison

Look for these kernels in results:
- `Lab2_TensorCore` (baseline)
- `Lab2_TensorCore_Optimized` (updated with 8 warps, larger tiles)
- `Lab2_TensorCore_HighPerf` (new high-performance version)
- `cuBLAS_HGEMM_TensorCore` (reference)

### Expected Results for 4096×4096

| Kernel | GFLOPS | Time (ms) | Efficiency |
|--------|--------|-----------|------------|
| cuBLAS TensorCore | ~90,000 | ~1.5 | 100% |
| **HighPerf (NEW)** | **~8,000-20,000** | **~6-25** | **~10-25%** |
| Optimized (UPDATED) | ~5,000-15,000 | ~10-30 | ~5-15% |
| Baseline | ~2,600 | ~51 | ~3% |

---

## Troubleshooting

### Build Errors

```bash
# If you get "undefined reference" errors, make sure all headers are included
# Check that op_mm_tensorcore_high_perf.cuh is in src/ops/

# If you get architecture errors, check CMakeLists.txt supports your GPU
# For H100, you may need to add "90" to CUDA_ARCHITECTURES
```

### Runtime Errors

```bash
# If kernel launch fails, check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# For H100 (compute 9.0), update CMakeLists.txt:
# set_target_properties(benchmark_gemm PROPERTIES CUDA_ARCHITECTURES "70;75;80;90")
```

### Performance Not Improved

```bash
# Check if the new kernels are actually running
grep "HighPerf\|Optimized" results/benchmark_results.csv

# Verify tile sizes in output
# Should see fewer blocks launched (4096/64 = 64 blocks per dimension)
```

---

## Quick Reference Commands

### Local
```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8 benchmark_gemm && cd .. && ./build/benchmark_gemm
```

### Greene (Submit Job)
```bash
sbatch scripts/run_phase4_final.sbatch
```

### Greene (Check Results)
```bash
tail -50 logs/phase4_final.out && cat results/final/performance_summary.txt
```

---

## Next Steps After Testing

1. **Compare Results**: Check if performance improved 2-8×
2. **Profile**: Use Nsight Compute to find remaining bottlenecks
3. **Document**: Update README with new performance numbers
4. **Present**: Use results to answer professor's questions about:
   - Matrix sizes tested
   - Blocks launched (now 4,096 instead of 16,384 for 4096×4096)
   - Warps per block (8 instead of 4)
   - Memory access patterns (coalesced)

