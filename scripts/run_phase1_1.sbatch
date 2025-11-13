#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=phase1_1_lab1
#SBATCH --output=logs/phase1_1.out
#SBATCH --gres=gpu:1

# FASTEST QUEUE OPTIONS
#SBATCH --partition=gpu_misc_v100,v100,rtx8000,a100_1,a100_2,h100_1

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

<<<<<<< HEAD
echo "✓ Phase 1.1 completed"
date
=======
echo "Phase 1.1 completed"
date
>>>>>>> ee3389f (Add Phase 1.2 sbatch script)
