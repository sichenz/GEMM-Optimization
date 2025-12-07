#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=phase4_final
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase4_final.out
#SBATCH --gres=gpu:rtx8000:1

set -euo pipefail

module purge

# Create necessary directories
mkdir -p results/{final,plots}
mkdir -p logs

echo "========================================================================"
echo "                    PHASE 4: FINAL COMPREHENSIVE EVALUATION            "
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Build project with all optimizations"
echo "  2. Run comprehensive benchmarks (all kernels, all sizes)"
echo "  3. Validate correctness"
echo "  4. Generate final performance analysis"
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
make -j8 benchmark_gemm

if [ \$? -ne 0 ]; then
    echo '✗ Build failed!'
    exit 1
fi
echo '✓ Build successful'
cd ..
echo ''

echo '========================================================================'
echo 'STEP 2: Comprehensive Benchmarking'
echo '========================================================================'

# Run comprehensive benchmark
./build/benchmark_gemm > results/final/benchmark_output.txt 2>&1

if [ \$? -ne 0 ]; then
    echo '✗ Benchmark failed!'
    exit 1
fi
echo '✓ Benchmark complete'
echo ''

echo '========================================================================'
echo 'STEP 3: Generate Final Analysis'
echo '========================================================================'

# Copy benchmark results
cp results/benchmark_results.csv results/final/benchmark_results.csv

# Generate comprehensive analysis
python3 src/generate_final_report.py

if [ \$? -ne 0 ]; then
    echo '✗ Report generation failed!'
    exit 1
fi
echo '✓ Analysis complete'

echo ''
echo '========================================================================'
echo 'PHASE 4 COMPLETE!'
echo '========================================================================'
echo ''
echo 'Generated Files:'
echo '  • results/final/benchmark_results.csv'
echo '  • results/final/benchmark_output.txt'
echo '  • results/final/performance_summary.txt'
echo ''
echo 'Review results:'
echo '  cat results/final/performance_summary.txt'
echo ''
"

echo ""
echo "========================================================================"
echo "PHASE 4 JOB COMPLETED!"
echo "========================================================================"
date

