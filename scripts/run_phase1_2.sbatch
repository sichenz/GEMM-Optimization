#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=phase1_2_cublas
#SBATCH --output=logs/phase1_2.out
#SBATCH --gres=gpu:1

# FASTEST queue options (same as Phase 1.1)
#SBATCH --partition=rtx8000,a100_1,a100_2,h100_1,v100,gpu_misc_v100

set -euo pipefail

module purge
module load cuda/11.6.2
module load cmake/3.28.0

mkdir -p build
mkdir -p results/cublas
mkdir -p logs

echo "=============================================="
echo "        PHASE 1.2 — cuBLAS Baseline Only"
echo "=============================================="
date

echo "[1/2] Cleaning & Building..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "[2/2] Running cuBLAS benchmark..."
./build/cublas_bench | tee results/cublas/cublas_output.txt

echo "Phase 1.2 completed"
date