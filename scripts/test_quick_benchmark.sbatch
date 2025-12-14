#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --mem=16GB
#SBATCH --job-name=test_bench
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/test_benchmark.out
#SBATCH --gres=gpu:rtx8000:1

set -euo pipefail

module purge

mkdir -p logs

echo "Quick Benchmark Test"
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

echo 'Step 1: Pull latest changes'
echo ''
git pull origin performance-optimizations

echo ''
echo 'Step 2: Build'
echo ''

rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 benchmark_gemm

if [ \$? -ne 0 ]; then
    echo 'Build failed!'
    exit 1
fi

echo 'Build successful'
cd ..

echo ''
echo 'Step 3: Quick test (first few matrix sizes)'
echo ''

# Run benchmark and check output
./build/benchmark_gemm 2>&1 | tee logs/test_output.txt

echo ''
echo 'Step 4: Verify kernels ran'
echo ''

if grep -q 'Lab2_TensorCore_Balanced' logs/test_output.txt; then
    echo '✓ Balanced kernel ran'
else
    echo '✗ Balanced kernel did NOT run'
    exit 1
fi

if grep -q 'Lab2_TensorCore_Optimized' logs/test_output.txt; then
    echo '✓ Optimized kernel ran'
else
    echo '✗ Optimized kernel did NOT run'
    exit 1
fi

if grep -q 'Lab2_TensorCore[^_]' logs/test_output.txt || grep -qE 'Lab2_TensorCore[,\s]' logs/test_output.txt; then
    echo '✓ Baseline kernel ran'
else
    echo '✗ Baseline kernel did NOT run'
    echo 'Checking what kernels actually ran:'
    grep -o 'Lab2_TensorCore[^,]*' logs/test_output.txt | sort -u || echo 'No Lab2 kernels found'
    exit 1
fi

echo ''
echo 'Step 5: Check results'
echo ''

if [ -f results/benchmark_results.csv ]; then
    echo '✓ Results file generated'
    echo ''
    echo 'Kernels in results:'
    grep -o 'Lab2_TensorCore[^,]*' results/benchmark_results.csv | sort -u
else
    echo '✗ Results file not found'
    exit 1
fi

echo ''
echo 'Test completed successfully!'
echo ''
"

echo ""
echo "Test job completed!"
date

