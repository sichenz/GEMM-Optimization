#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=gemm_benchmark
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --gres=gpu:1

module purge

# Create results directory
mkdir -p results
mkdir -p logs

cd /scratch/$USER/env

singularity exec --nv \
    --overlay /scratch/$USER/env/overlay-50G-10M.ext3:rw \
    /scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate test
cd /scratch/$USER/GEMM-Optimization

echo 'Running GEMM Benchmark Suite...'
echo 'This may take 30-60 minutes...'
./build/benchmark_gemm

echo 'Benchmark complete!'
echo 'Results saved to results/benchmark_results.csv'
"