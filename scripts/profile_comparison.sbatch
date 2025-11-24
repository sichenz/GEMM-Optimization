#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=profile_compare
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/profile_comparison.out
#SBATCH --gres=gpu:rtx8000:1

set -euo pipefail

module purge

mkdir -p results/profiling
mkdir -p logs

echo "========================================================================"
echo "PHASE 3: CUTLASS vs Our Kernel Profiling Comparison"
echo "========================================================================"
date
echo ""

singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

echo '========================================================================'
echo 'STEP 1: Build Project'
echo '========================================================================'

cd build
make -j8 benchmark_gemm
cd ..

echo '✓ Build complete'
echo ''

echo '========================================================================'
echo 'STEP 2: Profile Our TensorCore Kernel (4096x4096x4096)'
echo '========================================================================'

# Profile our optimized kernel
# Use exact kernel name from available kernels list
ncu --set full \
    --kernel-name-base function \
    --kernel-name 'op_mm_tensorcore_optimized_kernel' \
    --launch-skip 2 \
    --launch-count 1 \
    --target-processes all \
    --export results/profiling/our_kernel_4096.ncu-rep \
    ./build/benchmark_gemm 2>&1 | tee logs/profile_our_kernel.log

echo '✓ Our kernel profiled'
echo ''

echo '========================================================================'
echo 'STEP 3: Profile CUTLASS (if available)'
echo '========================================================================'

# Check if CUTLASS profiler exists
if [ -f third_party/cutlass/build/tools/profiler/cutlass_profiler ]; then
    echo 'Running CUTLASS profiler...'
    ./third_party/cutlass/build/tools/profiler/cutlass_profiler \
        --operation=Gemm \
        --A=f16:row \
        --B=f16:row \
        --C=f32:row \
        --accumulator-type=f32 \
        --m=4096 --n=4096 --k=4096 \
        --profiling-iterations=10 \
        > results/profiling/cutlass_4096.txt 2>&1
    echo '✓ CUTLASS profiled'
else
    echo '⚠ CUTLASS profiler not found, skipping'
fi

echo ''
echo '========================================================================'
echo 'PROFILING COMPLETE'
echo '========================================================================'
echo ''
echo 'Generated files:'
echo '  • results/profiling/our_kernel_4096.ncu-rep'
echo '  • results/profiling/cutlass_4096.txt (if available)'
echo ''
echo 'To analyze:'
echo '  1. Download .ncu-rep file to laptop'
echo '  2. Open with: ncu-ui results/profiling/our_kernel_4096.ncu-rep'
echo '  3. Compare metrics with CUTLASS'
echo ''
"

echo ""
echo "========================================================================"
echo "PROFILING JOB COMPLETED!"
echo "========================================================================"
date

