#!/bin/bash
#SBATCH --job-name=debug_bone_srun
#SBATCH --output=debug_bone_srun_%j.out
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30:00

echo "=== DEBUG BONE TEST WITH SRUN ==="
echo "Dataset: input/Fractura/bone_real.hdf5"
echo "Using srun for proper SLURM integration"
echo ""

# Activate environment
source .venv/bin/activate

echo "Starting debug evaluation with srun..."

# Run with srun as suggested by Lightning
srun python eval.py \
    experiment=denoiser_flow_matching \
    data.data_root=input/Fractura/bone_real.hdf5 \
    data.categories="[pig_bone]" \
    ckpt_path=output/GARF.ckpt \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    loggers=csv \
    trainer.devices=1 \
    ++seed=42

echo "Debug evaluation completed!"