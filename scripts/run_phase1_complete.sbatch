#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=phase1_complete
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase1_complete.out
#SBATCH --gres=gpu:1

set -euo pipefail

module purge

# Create necessary directories
mkdir -p results/{cutlass,plots}
mkdir -p logs

echo "Phase 1: Complete Baseline Analysis"
echo ""
echo "This script will:"
echo "  1. Collect GPU specifications"
echo "  2. Run GEMM benchmarks (Lab-1, cuBLAS)"
echo "  3. Run CUTLASS benchmarks"
echo "  4. Generate analysis and plots"
echo ""
date
echo ""

singularity exec --nv \
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/$USER/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

echo 'Step 1: Building Project'
echo ''

# Clean and rebuild
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..

if [ \$? -ne 0 ]; then
    echo 'Build failed!'
    exit 1
fi
echo 'Build successful'
echo ''

echo 'Step 2: GPU Specifications'
echo ''

./build/gpu_specs

if [ \$? -ne 0 ]; then
    echo 'GPU specs failed!'
    exit 1
fi
echo 'GPU specs collected'
echo ''

echo 'Step 3: GEMM Benchmarks (Lab-1, cuBLAS)'
echo 'This may take a while...'
echo ''

./build/benchmark_gemm

if [ \$? -ne 0 ]; then
    echo 'Benchmark failed!'
    exit 1
fi
echo 'Benchmarks complete'
echo ''

echo 'Step 4: CUTLASS Benchmarks'
echo ''

# Check if CUTLASS submodule exists
if [ ! -d 'third_party/cutlass/.git' ]; then
    echo 'Initializing CUTLASS submodule...'
    git submodule update --init --recursive
fi

# Detect GPU architecture
GPU_ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
echo \"Detected GPU compute capability: \${GPU_ARCH}\"

# Build CUTLASS profiler for correct architecture
if [ ! -f 'third_party/cutlass/build/tools/profiler/cutlass_profiler' ]; then
    echo 'Building CUTLASS profiler...'
    mkdir -p third_party/cutlass/build
    cd third_party/cutlass/build
    
    # FIXED: Use detected GPU architecture instead of hardcoded 75
    cmake .. -DCUTLASS_NVCC_ARCHS=\${GPU_ARCH} -DCUTLASS_UNITY_BUILD_ENABLED=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build . --target cutlass_profiler -j8
    cd ../../..
fi

PROF=third_party/cutlass/build/tools/profiler/cutlass_profiler

if [ ! -f \$PROF ]; then
    echo 'CUTLASS profiler not found, skipping'
else
    echo 'Running CUTLASS benchmarks...'
    
    # FP32 benchmarks
    if \$PROF --operation=gemm --providers=cutlass \
        --m=128,256,512,1024,2048,4096,8192 \
        --n=128,256,512,1024,2048,4096,8192 \
        --k=128,256,512,1024,2048,4096,8192 \
        --precision=f32 --accumulator-type=f32 \
        --disposition=equal \
        --output=results/cutlass/cutlass_fp32.csv 2>&1 | tail -20; then
        echo 'FP32 benchmarks complete'
    else
        echo 'FP32 benchmarks failed'
    fi
    
    # FP16 TensorCore benchmarks (if GPU supports it)
    if [ \"\${GPU_ARCH}\" -ge 70 ]; then
        if \$PROF --operation=gemm --providers=cutlass --op_class=tensorop \
            --m=128,256,512,1024,2048,4096,8192 \
            --n=128,256,512,1024,2048,4096,8192 \
            --k=128,256,512,1024,2048,4096,8192 \
            --precision=f16 --accumulator-type=f32 \
            --disposition=equal \
            --output=results/cutlass/cutlass_f16tc.csv 2>&1 | tail -20; then
            echo 'FP16 TensorCore benchmarks complete'
        else
            echo 'FP16 TensorCore benchmarks failed'
        fi
    fi
    
    echo 'CUTLASS benchmarks complete'
fi
echo ''

echo 'Step 5: Generating Analysis and Plots'
echo ''

python3 src/roofline_analysis.py

echo ''
echo 'Analysis complete'
echo ''

echo 'Phase 1 Complete!'
echo ''
echo 'Generated Files:'
echo '  results/gpu_specs.txt'
echo '  results/benchmark_results.csv'
echo '  results/cutlass/ (CUTLASS results)'
echo '  results/roofline_plot.png'
echo '  results/performance_comparison.png'
echo '  results/analysis_report.txt'
echo ''
"

echo ""
echo "All jobs completed!"
date