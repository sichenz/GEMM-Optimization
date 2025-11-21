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
mkdir -p results/{profiling,cutlass,plots}
mkdir -p logs
mkdir -p build

echo "========================================================================"
echo "                    PHASE 1: COMPLETE BASELINE ANALYSIS                "
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Collect GPU specifications"
echo "  2. Run GEMM benchmarks (Lab-1, cuBLAS)"
echo "  3. Profile Lab-1 kernel with Nsight Compute"
echo "  4. Run CUTLASS benchmarks"
echo "  5. Generate comprehensive analysis and visualizations"
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

echo '========================================================================'
echo 'STEP 1: Building Project'
echo '========================================================================'

# Clean build directory
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..

if [ \$? -ne 0 ]; then
    echo '✗ Build failed!'
    exit 1
fi
echo '✓ Build successful'
echo ''

echo '========================================================================'
echo 'STEP 2: GPU Specifications'
echo '========================================================================'

./build/gpu_specs

if [ \$? -ne 0 ]; then
    echo '✗ GPU specs failed!'
    exit 1
fi
echo '✓ GPU specs collected'
echo ''

echo '========================================================================'
echo 'STEP 3: GEMM Benchmarks (Lab-1, cuBLAS)'
echo '========================================================================'
echo 'This may take 30-60 minutes...'
echo ''

./build/benchmark_gemm

if [ \$? -ne 0 ]; then
    echo '✗ Benchmark failed!'
    exit 1
fi
echo '✓ Benchmarks complete'
echo ''

echo '========================================================================'
echo 'STEP 4: Nsight Compute Profiling'
echo '========================================================================'
echo 'Profiling Lab-1 kernel for key matrix sizes...'
echo ''

# Create profiling build with proper configuration
rm -rf build_profile
mkdir -p build_profile
cd build_profile
echo 'Building profiling executable...'
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8 benchmark_gemm
cd ..

if [ ! -f 'build_profile/benchmark_gemm' ]; then
    echo '✗ Failed to build profiling executable'
    echo '  Skipping profiling step'
else
    # Profile key sizes with better error handling
    for size in 512 2048 4096; do
        echo \"Profiling \${size}x\${size}x\${size}...\"
        
        # Calculate launch skip based on size
        # Skip indices: 128=0, 256=1, 512=2, 1024=3, 2048=4, 4096=5, 8192=6
        if [ \$size -eq 512 ]; then
            skip=2
        elif [ \$size -eq 2048 ]; then
            skip=4
        else
            skip=5
        fi
        
        # Profile with verbose error output
        if ncu --set full \
            --export results/profiling/lab1_\${size} \
            --target-processes all \
            --kernel-name regex:op_mm_kernel \
            --launch-skip \$skip \
            --launch-count 1 \
            --force-overwrite \
            ./build_profile/benchmark_gemm 2>&1 | tee profiling_\${size}.log | tail -20; then
            echo \"  ✓ Profiling \${size} succeeded\"
        else
            echo \"  ✗ Profiling \${size} failed\"
            echo \"  Check profiling_\${size}.log for details\"
        fi
    done
fi

echo '✓ Profiling step complete'
echo ''

echo '========================================================================'
echo 'STEP 5: CUTLASS Benchmarks'
echo '========================================================================'

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
    echo '✗ CUTLASS profiler not found, skipping CUTLASS benchmarks'
else
    echo 'Running CUTLASS benchmarks...'
    
    # FP32 Square
    if \$PROF --operation=gemm --providers=cutlass \
        --m=128,256,512,1024,2048,4096,8192 \
        --n=128,256,512,1024,2048,4096,8192 \
        --k=128,256,512,1024,2048,4096,8192 \
        --precision=f32 --accumulator-type=f32 \
        --disposition=equal \
        --output=results/cutlass/cutlass_fp32.csv 2>&1 | tail -20; then
        echo '✓ FP32 benchmarks complete'
    else
        echo '✗ FP32 benchmarks failed'
    fi
    
    # FP16 TensorCore Square (only if supported)
    if [ \"\${GPU_ARCH}\" -ge 70 ]; then
        if \$PROF --operation=gemm --providers=cutlass --op_class=tensorop \
            --m=128,256,512,1024,2048,4096,8192 \
            --n=128,256,512,1024,2048,4096,8192 \
            --k=128,256,512,1024,2048,4096,8192 \
            --precision=f16 --accumulator-type=f32 \
            --disposition=equal \
            --output=results/cutlass/cutlass_f16tc.csv 2>&1 | tail -20; then
            echo '✓ FP16 TensorCore benchmarks complete'
        else
            echo '✗ FP16 TensorCore benchmarks failed'
        fi
    fi
    
    echo '✓ CUTLASS benchmarks complete'
fi
echo ''

echo '========================================================================'
echo 'STEP 6: Generating Analysis and Visualizations'
echo '========================================================================'

# Run comprehensive analysis
python3 src/roofline_analysis.py

# Analyze profiling results (only if files exist)
if ls results/profiling/*.ncu-rep 1> /dev/null 2>&1; then
    python3 src/analyze_nsight_profile.py
else
    echo 'No profiling data found, skipping Nsight analysis'
fi

echo ''
echo '✓ Analysis complete'
echo ''

echo '========================================================================'
echo 'PHASE 1 COMPLETE!'
echo '========================================================================'
echo ''
echo 'Generated Files:'
echo '  GPU Specifications:'
echo '    • results/gpu_specs.txt'
echo ''
echo '  Benchmark Results:'
echo '    • results/benchmark_results.csv'
echo '    • results/cutlass/ (CUTLASS results)'
echo ''
echo '  Profiling Data:'
if ls results/profiling/*.ncu-rep 1> /dev/null 2>&1; then
    echo '    • results/profiling/lab1_*.ncu-rep'
    echo '    • results/profiling/profiling_summary.txt'
else
    echo '    • (Profiling data not available)'
fi
echo ''
echo '  Analysis & Visualizations:'
echo '    • results/roofline_plot.png'
echo '    • results/performance_comparison.png'
echo '    • results/analysis_report.txt'
echo ''
echo 'Next Steps:'
echo '  1. Review results/analysis_report.txt for optimization opportunities'
echo '  2. Compare against cuBLAS and CUTLASS baselines'
echo '  3. Proceed to Phase 2 optimizations'
echo ''
"

echo ""
echo "========================================================================"
echo "ALL JOBS COMPLETED!"
echo "========================================================================"
date