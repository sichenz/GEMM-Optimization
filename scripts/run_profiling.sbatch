#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=profile_lab1
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/profiling.out
#SBATCH --gres=gpu:1

set -euo pipefail

module purge

# Create necessary directories
mkdir -p results/profiling
mkdir -p logs
mkdir -p build_profile

echo "========================================================================"
echo "                    PROFILING LAB-1 GEMM KERNEL                       "
echo "========================================================================"
echo ""
date
echo ""

singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
cd /scratch/$USER/GEMM-Optimization

echo '========================================================================'
echo 'STEP 1: Building Profiling Executable'
echo '========================================================================'

# Build profiling version with debug info
rm -rf build_profile
mkdir -p build_profile
cd build_profile
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8 benchmark_gemm
cd ..

if [ ! -f 'build_profile/benchmark_gemm' ]; then
    echo '✗ Failed to build profiling executable'
    exit 1
fi
echo '✓ Build successful'
echo ''

echo '========================================================================'
echo 'STEP 2: Profiling Lab-1 Kernel'
echo '========================================================================'
echo 'Profiling key matrix sizes: 512, 2048, 4096'
echo ''

# Profile key sizes
for size in 512 2048 4096; do
    echo \"Profiling \${size}x\${size}x\${size}...\"
    
    # Calculate launch skip based on size
    # Each matrix size runs 3 kernels: Lab1_Tiled, cuBLAS_SGEMM, cuBLAS_HGEMM
    # So for size 512 (3rd size): skip = (0+1+2) * 3 = 9 launches (skip 128, 256, and cuBLAS for 512)
    # We want to profile the Lab-1 kernel for the target size
    if [ \$size -eq 512 ]; then
        # Size 512 is index 2: skip 0,1 (128,256) = 6 launches, then Lab1 for 512 = launch 6
        skip=6
    elif [ \$size -eq 2048 ]; then
        # Size 2048 is index 4: skip 0,1,2,3 (128,256,512,1024) = 12 launches, then Lab1 for 2048 = launch 12
        skip=12
    else
        # Size 4096 is index 5: skip 0,1,2,3,4 = 15 launches, then Lab1 for 4096 = launch 15
        skip=15
    fi
    
    # Profile with Nsight Compute
    # Note: We profile all kernels since the template kernel name might be mangled
    # The Lab-1 kernel will be in the report and can be identified manually
    if ncu --set full \
        --export results/profiling/lab1_\${size} \
        --target-processes all \
        --launch-skip \$skip \
        --launch-count 1 \
        --force-overwrite \
        ./build_profile/benchmark_gemm 2>&1 | tee logs/profiling_\${size}.log; then
        if [ -f results/profiling/lab1_\${size}.ncu-rep ]; then
            echo \"  ✓ Profiling \${size} succeeded\"
        else
            echo \"  ✗ Profiling \${size} failed - no .ncu-rep file created\"
        fi
    else
        echo \"  ✗ Profiling \${size} failed\"
        echo \"  Check logs/profiling_\${size}.log for details\"
    fi
    echo ''
done

echo '✓ Profiling complete'
echo ''

echo '========================================================================'
echo 'PROFILING COMPLETE!'
echo '========================================================================'
echo ''
echo 'Generated Files:'
echo '  • results/profiling/lab1_512.ncu-rep'
echo '  • results/profiling/lab1_2048.ncu-rep'
echo '  • results/profiling/lab1_4096.ncu-rep'
echo ''
echo 'To analyze results:'
echo '  1. Download .ncu-rep files to your laptop'
echo '  2. Open with: ncu-ui results/profiling/lab1_*.ncu-rep'
echo '  3. Or run: python3 src/analyze_nsight_profile.py (on laptop with pandas)'
echo ''
"

echo ""
echo "========================================================================"
echo "PROFILING JOB COMPLETED!"
echo "========================================================================"
date

