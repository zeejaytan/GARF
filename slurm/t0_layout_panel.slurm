#!/bin/bash
#SBATCH --job-name=pfpp-t0-panel
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=t0_layout_panel_%j.out

# T0 — like-for-like layout quality panel (PFPP_JUGLET_SUCCESS_EXPERIMENT_PLAN.md).
# Step 1: fresh baseline-GARF deploy on the ORIGINAL juglet (fixed GLB export)
#         — the recent 9pc runs used exp14 de-weathered geometry or the exp15
#         self-sup model, so the baseline arm needs its own run.
# Step 2: score PF++ final layout vs GARF vs random-compact and identity-
#         permuted nulls with scripts/pfpp_layout_probes.py.

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="t0_garf_base_${TS}"
PF=/data/gpfs/projects/punim2657/Puzzlefusion

echo "=== T0 step 1: baseline GARF deploy on original juglet ==="
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/deploy \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=input/juglet_deploy.hdf5 \
    data.categories="['artifact']" \
    data.min_parts=2 \
    data.max_parts=20 \
    data.batch_size=1 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=output/GARF.ckpt \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true \
    ++model.inference_config.anchor_free=true \
    ++model.inference_config.deploy_mode=true \
    ++model.inference_config.save_assembly=true

GLB="logs/deploy/$EXPERIMENT_NAME/version_0/assembly_results/artifact/Juglet-000/predicted_assembly.glb"
ls -l "$GLB"

echo "=== T0 step 2: layout panel ==="
python scripts/pfpp_layout_probes.py \
    --pfpp-dir "$PF/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0" \
    --pc-npz   "$PF/data/pc_data/juglet_deploy/val/00000.npz" \
    --mesh-hdf5 /data/gpfs/projects/punim2657/GARF/input/juglet_deploy.hdf5 --mesh-key artifact/Juglet-000 \
    --garf-glb "$GLB" \
    --out      "logs/diagnostics/pfpp_t0_${TS}" \
    --tag t0 \
    --n-null 5

echo "=== DONE: logs/diagnostics/pfpp_t0_${TS} ==="
