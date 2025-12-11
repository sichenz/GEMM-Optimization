#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --mem=32GB
#SBATCH --job-name=phase2_complete
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase2_complete.out
#SBATCH --gres=gpu:1

set -euo pipefail

module purge

# Create necessary directories (host side)
mkdir -p logs
mkdir -p build
mkdir -p results

echo "========================================================================"
echo "                    PHASE 2: TENSORCORE OPTIMIZATION                    "
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Build the project with TensorCore kernels"
echo "  2. Collect GPU specifications"
echo "  3. Run GEMM benchmarks including:"
echo "       - Baseline TensorCore kernel (op_mm_tensorcore)"
echo "       - Optimized TensorCore kernel (op_mm_tensorcore_optimized)"
echo "  4. Generate analysis and plots"
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
echo 'STEP 1: Building Project'
echo '========================================================================'

# Clean build directory
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ..

echo '✓ Build successful'
echo ''

echo '========================================================================'
echo 'STEP 2: GPU Specifications'
echo '========================================================================'

./build/gpu_specs || { echo '✗ GPU specs failed!'; exit 1; }

echo '✓ GPU specs collected'
echo ''

echo '========================================================================'
echo 'STEP 3: TensorCore GEMM Benchmarks'
echo '========================================================================'
echo 'Running GEMM benchmarks including baseline and optimized TensorCore kernels...'
echo ''

./build/benchmark_gemm || { echo '✗ Benchmark failed!'; exit 1; }

echo '✓ Benchmarks complete'
echo ''

echo '========================================================================'
echo 'STEP 4: Generating Analysis and Visualizations'
echo '========================================================================'

python3 src/performance.py

echo ''
echo '✓ Analysis complete'
echo ''

echo '========================================================================'
echo 'STEP 5: Phase 2 File Renaming'
echo '========================================================================'

mkdir -p results

# Rename analysis outputs for Phase 2
if [ -f results/performance_plot.png ]; then
    mv results/performance_plot.png results/performance_plot_phase2.png
fi

if [ -f results/performance_comparison.png ]; then
    mv results/performance_comparison.png results/performance_comparison_phase2.png
fi

if [ -f results/analysis_report.txt ]; then
    mv results/analysis_report.txt results/analysis_report_phase2.txt
fi

# Rename GPU specs and benchmark CSV for Phase 2
if [ -f results/gpu_specs.txt ]; then
    mv results/gpu_specs.txt results/gpu_specs_phase2.txt
fi

if [ -f results/benchmark_results.csv ]; then
    mv results/benchmark_results.csv results/benchmark_results_phase2.csv
fi

echo ''
echo '✓ Phase 2 files renamed'
echo ''

echo '========================================================================'
echo 'PHASE 2 COMPLETE!'
echo '========================================================================'
echo ''
echo 'Generated Files:'
echo '  GPU Specifications:'
echo '    • results/gpu_specs_phase2.txt'
echo ''
echo '  Benchmark Results (including TensorCore kernels):'
echo '    • results/benchmark_results_phase2.csv'
echo ''
echo '  Analysis & Visualizations:'
echo '    • results/performance_plot_phase2.png'
echo '    • results/performance_comparison_phase2.png'
echo '    • results/analysis_report_phase2.txt'
echo ''
echo 'Next Steps:'
echo '  - Compare baseline TensorCore vs optimized TensorCore vs cuBLAS'
echo '  - Inspect results/analysis_report_phase2.txt and plots for Phase 2 discussion'
echo ''
"

echo ""
echo "========================================================================"
echo "ALL PHASE 2 JOBS COMPLETED!"
echo "========================================================================"
date
