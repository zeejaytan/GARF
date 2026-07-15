#!/bin/bash
#SBATCH --job-name=garf_exp13_coadapt
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=exp13_coadapt_%j.out

# Exp 13 — co-adapt the denoiser to the worn encoder (fixes the Exp 12 confound:
# swapping only the encoder fed the denoiser features it was never trained on).
# Full denoiser fine-tune from GARF_worn_encoder.ckpt (worn frozen encoder +
# original denoiser) on worn-augmented synthetic fractures, then evaluate on the
# decisive Juglet pairwise chamfer oracle.
#
# EPOCHS and SMOKE overridable via env: SMOKE=1 runs a 1-epoch shakeout.

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

EPOCHS="${EPOCHS:-40}"
SMOKE="${SMOKE:-0}"
STAMP=$(date +%Y%m%d_%H%M%S)
EXP="exp13_coadapt_${STAMP}"
if [ "${SMOKE}" = "1" ]; then EPOCHS=1; EXP="exp13_smoke_${STAMP}"; fi

echo "=== Exp 13 denoiser co-adaptation (epochs=${EPOCHS}, smoke=${SMOKE}) ==="; date
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Full denoiser fine-tune. finetuning=true loads GARF_worn_encoder.ckpt (worn
# encoder stays frozen in DenoiserBase.__init__); denoiser params train.
python train.py \
    experiment=denoiser_flow_matching \
    experiment_name="${EXP}" \
    seed=42 \
    loggers=csv \
    trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=${EPOCHS} \
    +trainer.limit_val_batches=0 \
    +trainer.num_sanity_val_steps=0 \
    ckpt_path=output/GARF_worn_encoder.ckpt \
    finetuning=true \
    model.feature_extractor_ckpt=null \
    model.optimizer.lr=1e-5 \
    callbacks.model_checkpoint.save_on_train_epoch_end=true \
    data.data_root=input/Fractura/bone_synthetic.hdf5 \
    data.categories="[pig,hip,leg,rib,vertebra]" \
    data.batch_size=16 \
    data.num_workers=12 \
    data.frac_erode_prob=0.6

CKPT="output/${EXP}/last.ckpt"
echo "trained checkpoint: ${CKPT}"
ls -la "output/${EXP}/" || true
if [ "${SMOKE}" = "1" ]; then echo "SMOKE done"; date; exit 0; fi

# --- evaluate co-adapted model on the Juglet pairwise chamfer oracle ---
PAIR_HDF5=input/juglet_pairs_local02.hdf5
ETAG="exp13_pairs_${STAMP}"
PFPP_DIR=/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0
MESH_DIR=/data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000

RUN_DIRS=()
for SEED in 41 42 43; do
    NAME="${ETAG}_s${SEED}"
    echo "--- eval pairs (co-adapted, seed=${SEED}) ---"
    python eval.py \
        seed="${SEED}" experiment=denoiser_flow_matching \
        experiment_name="${NAME}" \
        loggers=csv loggers.csv.save_dir=logs/deploy \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${PAIR_HDF5}" data.categories="['artifact']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 \
        data.multi_ref=False \
        ckpt_path="${CKPT}" \
        ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=true \
        ++model.inference_config.save_assembly=true
    RUN_DIRS+=("logs/deploy/${NAME}")
done

OUT=logs/diagnostics/pair_chamfer_juglet_coadapt_${STAMP}
python scripts/pair_reference_chamfer.py juglet \
    --run-dirs "${RUN_DIRS[@]}" \
    --pairs-hdf5 "${PAIR_HDF5}" \
    --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
    --pfpp-dir "${PFPP_DIR}" --mesh-dir "${MESH_DIR}" \
    --out "${OUT}"

echo "=== RESULT vs baseline (0.070 true-mate == 0.073 non-mate, no separation) ==="
sed -n '1,10p' "${OUT}/summary.md"
echo "=== DONE -> ${OUT}/summary.md ==="; date
