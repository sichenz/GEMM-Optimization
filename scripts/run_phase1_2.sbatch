#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=phase1_2_cublas
#SBATCH --output=logs/phase1_2.out
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load cuda/11.6.2
module load cmake/3.28.0

mkdir -p build
mkdir -p results/cublas
mkdir -p logs

echo "=============================================="
echo "          PHASE 1.2 — cuBLAS Baseline Only"
echo "=============================================="
date

# Build
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "Running cuBLAS benchmark..."
./build/cublas_bench | tee results/cublas/cublas_output.txt

echo "✓ Phase 1.2 completed"
date