#!/bin/bash
#SBATCH --job-name=bone_original
#SBATCH --output=/data/gpfs/projects/punim2657/GARF/logs/slurm/bone_original_%j.out
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

GARF_WORK=/data/gpfs/projects/punim2657/GARF        # working area: input/ output/ logs/ (data, untracked)
GARF_REPO=$GARF_WORK/repo   # the code: fetch-only checkout of zeejaytan/GARF
cd "$GARF_WORK"

echo "=== BONE TEST WITH ORIGINAL CODE ==="
echo "Dataset: input/Fractura/bone_real.hdf5"
echo "Using original scheduler and config"
echo ""

source /data/gpfs/projects/punim2657/CRAG/.venv/bin/activate  # shared env (GARF+CRAG, 2026-09-18)

echo "Starting bone evaluation with original code..."

srun python $GARF_REPO/eval.py \
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