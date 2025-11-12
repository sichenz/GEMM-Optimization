#!/bin/bash
#SBATCH -J baselines
#SBATCH -p rtx8000
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

module purge
module load cuda/11.6
module load cmake

# (optional) show GPU to confirm node type
nvidia-smi --query-gpu=name --format=csv

# Build your repo (cuBLAS bench)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

mkdir -p results

# --- cuBLAS CSVs ---
./build/cublas_bench --csv               > results/cublas_results.csv
./build/cublas_bench --csv --mixed-only  > results/cublas_mixed.csv
./build/cublas_bench --csv --fp32-only   > results/cublas_fp32.csv

# --- CUTLASS profiler from submodule (pinned) ---
cmake -S third_party/cutlass -B third_party/cutlass/build \
      -DCUTLASS_NVCC_ARCHS="75" -DCUTLASS_UNITY_BUILD_ENABLED=ON
cmake --build third_party/cutlass/build -j cutlass_profiler
PROF=third_party/cutlass/build/tools/profiler/cutlass_profiler

# Square sizes — FP16->FP32 (tensorop) and FP32
for s in 128 256 512 1024 1536 2048 4096 8192; do
  "$PROF" --operation=gemm --op_class=tensorop \
          --m=$s --n=$s --k=$s \
          --precision=f16 --accumulator=f32 --arch=75 \
          --output=results/cutlass_f16f32_${s}.csv

  "$PROF" --operation=gemm \
          --m=$s --n=$s --k=$s \
          --precision=f32 --accumulator=f32 --arch=75 \
          --output=results/cutlass_fp32_${s}.csv
done

# Rectangular samples
"$PROF" --operation=gemm --op_class=tensorop \
        --m=4096 --n=256 --k=1024 --precision=f16 --accumulator=f32 --arch=75 \
        --output=results/cutlass_f16f32_rect1.csv

"$PROF" --operation=gemm \
        --m=1024 --n=8192 --k=512 --precision=f32 --accumulator=f32 --arch=75 \
        --output=results/cutlass_fp32_rect2.csv