#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=phase1_1_lab1
#SBATCH --output=logs/phase1_1.out
#SBATCH --gres=gpu:1

# FASTEST QUEUE OPTIONS
#SBATCH --partition=gpu_misc_v100,gpu_misc_a100,gpu_misc

set -euo pipefail

module purge
module load cuda/11.6.2
module load cmake/3.28.0

mkdir -p build
mkdir -p results/lab1
mkdir -p logs

echo "=============================================="
echo "     PHASE 1.1 — Lab-1 GEMM Baseline Only"
echo "=============================================="
date

echo "[1/2] Cleaning & Building..."
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

echo "[2/2] Running Lab-1 benchmark..."
./build/benchmark_gemm | tee results/lab1/lab1_gemm_output.txt

echo "✓ Phase 1.1 completed"
date