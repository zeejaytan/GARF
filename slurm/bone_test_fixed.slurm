#!/bin/bash
#SBATCH --job-name=bone_test_fixed
#SBATCH --output=bone_test_fixed_%j.out
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "=== BONE TEST WITH FIXED CONFIGURATION ==="
echo "Dataset: input/Fractura/bone_real.hdf5"
echo "Fix: Override num_nodes=1 to match SLURM config"
echo ""

source .venv/bin/activate

echo "Starting fixed bone evaluation..."

srun python eval.py \
    experiment=denoiser_flow_matching \
    data.data_root=input/Fractura/bone_real.hdf5 \
    data.categories="[pig_bone]" \
    ckpt_path=output/GARF.ckpt \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    loggers=csv \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    ++seed=42

echo "Bone evaluation completed!"