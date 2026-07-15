#!/bin/bash
#SBATCH --job-name=garf_exp11_smoke
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp11_smoke_%j.out
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1
echo "SMOKE: 2-epoch FracSeg worn-aug fine-tune on pig"; date
python scripts/finetune_frac_seg.py \
    --init-ckpt output/feature_extractor.ckpt \
    --data-root input/Fractura/bone_synthetic.hdf5 \
    --categories pig \
    --out-dir output/frac_seg_smoke_$(date +%H%M%S) \
    --epochs 2 --batch-size 8 --lr 2e-5 --erode-prob 0.6 \
    --num-workers 6 --save-every 1
echo "SMOKE DONE"; date
