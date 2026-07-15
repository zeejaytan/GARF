#!/bin/bash
#SBATCH --job-name=garf-diag-9pc
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=garf_diag_9pc_%j.out

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

echo "=== GARF matched 9-piece diagnostic ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/garf_matched_diagnostic.py \
  --pieces 9 \
  --seed 42

echo "=== DONE ==="
