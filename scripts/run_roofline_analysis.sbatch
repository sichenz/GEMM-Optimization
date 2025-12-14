#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem=8GB
#SBATCH --job-name=roofline
#SBATCH --mail-user=sz4972@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/roofline_analysis.out

set -euo pipefail

module purge

mkdir -p logs
mkdir -p results

echo "Roofline Analysis"
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

echo 'Running roofline analysis...'
echo ''

python3 src/roofline_analysis.py

if [ \$? -ne 0 ]; then
    echo 'Roofline analysis failed!'
    exit 1
fi

echo ''
echo 'Roofline analysis complete!'
echo ''
echo 'Generated files:'
echo '  - results/roofline_plot.png'
echo '  - results/performance_comparison.png'
echo '  - results/analysis_report.txt'
echo ''
"

echo ""
echo "Roofline analysis job completed!"
date

