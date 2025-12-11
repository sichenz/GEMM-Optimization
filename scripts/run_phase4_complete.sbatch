#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --mem=32GB
#SBATCH --job-name=phase4_complete
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase4_complete.out
#SBATCH --gres=gpu:1

set -euo pipefail

module purge

# Host-side directories (safe even if container also makes them)
mkdir -p results
mkdir -p logs
mkdir -p build

echo "========================================================================"
echo "          PHASE 4: INTEGRATION & COMPREHENSIVE EVALUATION              "
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Build the project (all GEMM and baseline kernels)"
echo "  2. Run comprehensive GEMM benchmarks (FP32 & TensorCore FP16)"
echo "  3. Collect Phase 4 benchmark CSVs and logs"
echo "  4. Run Python analysis to generate Phase 4 plots and summary"
echo "  5. (Optional) Run generate_final_report.py to aggregate all phases"
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

# Ensure result dirs exist inside the container
mkdir -p results
mkdir -p results/final_phase4
mkdir -p results/plots
mkdir -p logs

echo '========================================================================'
echo 'STEP 1: Building Project'
echo '========================================================================'

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd ..
echo '✓ Build successful'
echo ''

echo '========================================================================'
echo 'STEP 2: Comprehensive GEMM Benchmarks (Phase 4)'
echo '========================================================================'
echo 'Running benchmark_gemm to exercise:'
echo '  - Lab-1 tiled FP32 kernel'
echo '  - Our TensorCore kernels (baseline, optimized, large-tile)'
echo '  - cuBLAS FP32 and TensorCore baselines'
echo 'Across:'
echo '  - Square sizes: 128, 256, 512, 1024, 2048, 4096, 8192'
echo '  - Rectangular shapes (e.g. 4096x256x1024, 1024x8192x512, etc.)'
echo ''

./build/benchmark_gemm 2>&1 | tee results/final_phase4/benchmark_output_phase4.txt

if [ ! -f results/benchmark_results.csv ]; then
    echo '✗ benchmark_results.csv not found after benchmark_gemm'
    exit 1
fi

# Preserve Phase 4 CSV with a phase-specific name
cp results/benchmark_results.csv results/benchmark_results_phase4.csv
cp results/benchmark_results_phase4.csv results/final_phase4/benchmark_results_phase4.csv

echo '✓ Benchmarks complete'
echo ''

echo '========================================================================'
echo 'STEP 3: Phase 4 Analysis & Plots'
echo '========================================================================'

# Run generic performance analysis if available
if [ -f src/performance.py ]; then
    python3 src/performance.py || echo '⚠ performance.py reported an error, continuing'
    
    # Rename plots/analysis to phase4-specific versions if generated
    if [ -f results/performance_plot.png ]; then
        mv results/performance_plot.png results/performance_plot_phase4.png
        cp results/performance_plot_phase4.png results/plots/
        cp results/performance_plot_phase4.png results/final_phase4/
    fi
    
    if [ -f results/performance_comparison.png ]; then
        mv results/performance_comparison.png results/performance_comparison_phase4.png
        cp results/performance_comparison_phase4.png results/plots/
        cp results/performance_comparison_phase4.png results/final_phase4/
    fi
    
    if [ -f results/analysis_report.txt ]; then
        mv results/analysis_report.txt results/analysis_report_phase4.txt
        cp results/analysis_report_phase4.txt results/final_phase4/
    fi
else
    echo '⚠ src/performance.py not found, skipping Phase 4 plot generation'
fi

echo '✓ Phase 4 analysis step complete (where scripts were available)'
echo ''

echo '========================================================================'
echo 'STEP 4: Final Cross-Phase Summary (Optional)'
echo '========================================================================'
echo 'If src/generate_final_report.py is present, we will run it to produce'
echo 'an aggregate summary comparing: Lab-1 / our kernels / cuBLAS / CUTLASS.'
echo ''

if [ -f src/generate_final_report.py ]; then
    # Arguments are hypothetical; adjust inside the script as needed
    python3 src/generate_final_report.py --phase4 \
        --out results/final_phase4/performance_summary_phase4.txt || \
        echo '⚠ generate_final_report.py reported an error'
else
    echo '⚠ src/generate_final_report.py not found, skipping final summary script'
fi

echo ''
echo '========================================================================'
echo 'PHASE 4 COMPLETE!'
echo '========================================================================'
echo ''
echo 'Generated / updated files (Phase 4):'
echo '  • results/benchmark_results_phase4.csv'
echo '  • results/final_phase4/benchmark_output_phase4.txt'
echo '  • results/final_phase4/benchmark_results_phase4.csv'
echo '  • results/performance_plot_phase4.png            (if performance.py ran)'
echo '  • results/performance_comparison_phase4.png      (if performance.py ran)'
echo '  • results/analysis_report_phase4.txt             (if performance.py ran)'
echo '  • results/final_phase4/performance_summary_phase4.txt (if generated)'
echo ''
echo 'You can now:'
echo '  - Compare Phase 4 CSV against Phase 1/2 CSVs'
echo '  - Inspect plots in results/plots/ and results/final_phase4/'
echo '  - Use Nsight Compute data from Phase 3 for deeper microarchitectural analysis'
echo ''
"

echo ""
echo "========================================================================"
echo "PHASE 4 JOB COMPLETED!"
echo "========================================================================"
date
