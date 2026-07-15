#!/bin/bash
#SBATCH --job-name=garf_bone_real
#SBATCH --output=garf_bone_real_%j.out
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "=== TESTING GARF ON REAL BONE FRACTURES ==="
echo "Dataset: input/Fractura/bone_real.hdf5 (pig_bone category)"
echo "Expected: Domain mismatch - model trained on everyday objects"
echo "Samples: Real bone fracture validation cases"
echo ""

echo "Starting GARF evaluation on real bone fractures..."

# Activate environment
source .venv/bin/activate

# Run evaluation
HYDRA_FULL_ERROR=1 python eval.py \
    experiment=denoiser_flow_matching \
    data.data_root=input/Fractura/bone_real.hdf5 \
    data.categories="[pig_bone]" \
    ckpt_path=output/GARF.ckpt \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    loggers=csv \
    trainer.devices=1 \
    ++seed=42

echo "Real bone fracture evaluation completed!"
echo "Check logs/GARF/real_bone_fractures/ for results"