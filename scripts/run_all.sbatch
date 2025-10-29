#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=phase1_complete
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/phase1_complete_%j.out
#SBATCH --gres=gpu:1

module purge

# Create necessary directories
mkdir -p results
mkdir -p logs

echo "======================================"
echo "Phase 1.1: Complete Baseline Analysis"
echo "======================================"
echo ""
date

cd /scratch/$USER/env

singularity exec --nv \
    --overlay /scratch/$USER/env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

echo '======================================'
echo 'Step 1: Building project...'
echo '======================================'

if [ ! -d 'build' ]; then
    mkdir build
fi

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..

if [ \$? -ne 0 ]; then
    echo 'Build failed! Please fix compilation errors.'
    exit 1
fi

echo 'Build successful!'
echo ''

echo '======================================'
echo 'Step 2: Collecting GPU specifications...'
echo '======================================'

./build/gpu_specs

if [ \$? -ne 0 ]; then
    echo 'GPU specs failed!'
    exit 1
fi

echo 'GPU specs complete!'
echo ''

echo '======================================'
echo 'Step 3: Running GEMM benchmarks...'
echo 'This will take 30-60 minutes...'
echo '======================================'

./build/benchmark_gemm

if [ \$? -ne 0 ]; then
    echo 'Benchmark failed!'
    exit 1
fi

echo 'Benchmarks complete!'
echo ''

echo '======================================'
echo 'Step 4: Generating roofline analysis...'
echo '======================================'

python3 src/roofline_analysis.py

if [ \$? -ne 0 ]; then
    echo 'Roofline analysis failed!'
    exit 1
fi

echo 'Analysis complete!'
echo ''

echo '======================================'
echo 'Phase 1.1 Complete!'
echo '======================================'
echo ''
echo 'Results available in:'
echo '  - results/gpu_specs.txt'
echo '  - results/benchmark_results.csv'
echo '  - results/roofline_plot.png'
echo '  - results/performance_comparison.png'
echo '  - results/analysis_report.txt'
echo ''
echo 'All jobs completed! Check results/ directory for outputs.'
"


