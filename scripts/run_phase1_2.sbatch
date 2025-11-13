#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=phase1_2_lab1
#SBATCH --output=logs/phase1_2.out
#SBATCH --gres=gpu:1

# Use the most available partitions (same as Phase 1.1 success)
#SBATCH --partition=rtx8000

set -euo pipefail

module purge
module load cuda/11.6.2
module load cmake/3.28.0

mkdir -p build
mkdir -p results/lab1
mkdir -p logs

echo "=============================================="
echo "   PHASE 1.2 — cuBLAS Baseline (FP32 & FP16)"
echo "=============================================="
date

echo "[1/2] Building project..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "[2/2] Running cuBLAS-only benchmark..."
./build/benchmark_cublas | tee results/lab1/lab1_cublas_output.txt

echo "Phase 1.2 completed"
date