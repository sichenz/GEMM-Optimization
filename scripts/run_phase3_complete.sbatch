#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --mem=32GB
#SBATCH --job-name=phase3_complete
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase3_complete.out
#SBATCH --gres=gpu:1

set -euo pipefail

module purge

# Create necessary directories on host
mkdir -p results/profiling_phase3
mkdir -p logs
mkdir -p build

echo "========================================================================"
echo "          PHASE 3: CUTLASS vs Our TensorCore Kernel PROFILING           "
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Rebuild benchmark_gemm"
echo "  2. Profile our optimized TensorCore kernel with Nsight Compute"
echo "  3. Profile CUTLASS GEMM for the same size (if profiler is available)"
echo ""
date
echo ""

cd /scratch/$USER/env

singularity exec --nv \
    --overlay /scratch/$USER/env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
set -euo pipefail

source /ext3/env.sh
conda activate test
cd /scratch/\$USER/GEMM-Optimization

echo '========================================================================'
echo 'STEP 1: Build Project'
echo '========================================================================'

# Clean + rebuild (to pick up any new kernels)
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm
cd ..

echo '✓ Build complete'
echo ''

echo '========================================================================'
echo 'STEP 2: Profile Our Optimized TensorCore Kernel (4096x4096x4096)'
echo '========================================================================'

mkdir -p results/profiling_phase3

# Profile our optimized TensorCore kernel
# Make sure kernel name matches the __global__ symbol in your code
if ncu --set full \
    --kernel-name-base function \
    --kernel-name 'op_mm_tensorcore_optimized_kernel' \
    --launch-skip 2 \
    --launch-count 1 \
    --target-processes all \
    --export results/profiling_phase3/our_kernel_4096 \
    ./build/benchmark_gemm 2>&1 | tee results/profiling_phase3/profile_our_kernel_phase3.log | tail -20; then
    echo '✓ Our optimized TensorCore kernel profiled'
else
    echo '✗ Nsight Compute profiling for our kernel failed'
fi

echo ''
echo '========================================================================'
echo 'STEP 3: Profile CUTLASS GEMM (4096x4096x4096, if available)'
echo '========================================================================'

# Check if CUTLASS profiler exists; if not, try to build it (similar to Phase 1)
if [ ! -f third_party/cutlass/build/tools/profiler/cutlass_profiler ]; then
    echo 'CUTLASS profiler not found, attempting to build it...'
    if [ ! -d 'third_party/cutlass/.git' ]; then
        echo 'Initializing CUTLASS submodule...'
        git submodule update --init --recursive
    fi

    GPU_ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    echo \"Detected GPU compute capability: \${GPU_ARCH}\"

    mkdir -p third_party/cutlass/build
    cd third_party/cutlass/build
    cmake .. -DCUTLASS_NVCC_ARCHS=\${GPU_ARCH} -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target cutlass_profiler -j8
    cd ../../..
fi

if [ -f third_party/cutlass/build/tools/profiler/cutlass_profiler ]; then
    echo 'Running CUTLASS profiler for 4096x4096x4096 FP16->FP32 GEMM...'
    third_party/cutlass/build/tools/profiler/cutlass_profiler \
        --operation=Gemm \
        --A=f16:row \
        --B=f16:row \
        --C=f32:row \
        --accumulator-type=f32 \
        --m=4096 --n=4096 --k=4096 \
        --profiling-iterations=10 \
        > results/profiling_phase3/cutlass_4096.txt 2>&1
    echo '✓ CUTLASS profiled'
else
    echo '⚠ CUTLASS profiler still not available, skipping CUTLASS profiling'
fi

echo ''
echo '========================================================================'
echo 'PHASE 3 PROFILING COMPLETE'
echo '========================================================================'
echo ''
echo 'Generated files:'
echo '  • results/profiling_phase3/our_kernel_4096.ncu-rep'
echo '  • results/profiling_phase3/profile_our_kernel_phase3.log'
echo '  • results/profiling_phase3/cutlass_4096.txt (if CUTLASS profiler built)'
echo ''
echo 'Next steps:'
echo '  1. Download the .ncu-rep file to your laptop'
echo '  2. Open it with: ncu-ui results/profiling_phase3/our_kernel_4096.ncu-rep'
echo '  3. Compare Nsight metrics vs CUTLASS GEMM (results/profiling_phase3/cutlass_4096.txt)'
echo ''
"

echo ""
echo "========================================================================"
echo "PHASE 3 JOB COMPLETED!"
echo "========================================================================"
date
