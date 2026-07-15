#!/bin/bash
#SBATCH --job-name=garf_exp12_wornasm
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp12_wornasm_%j.out

# Exp 12 — does the worn-adapted encoder let GARF assemble the Juglet?
# Swap ONLY the encoder in GARF.ckpt for the Exp 11 worn fine-tuned one, then
# rerun the decisive pairwise oracle (Exp 6) + symmetry-invariant chamfer scoring
# and the full 9-piece Juglet deploy. Baseline (frozen encoder): true-mate
# chamfer 0.070 == non-mate 0.073 (NO separation). Success: true mates drop and
# separate from non-mates -> GARF now perceives Juglet mates.

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

# Locate the newest worn encoder (or pass WORN_CKPT=... to override).
WORN_CKPT="${WORN_CKPT:-$(ls -t output/frac_seg_worn_*/feature_extractor_worn.ckpt 2>/dev/null | head -1)}"
if [ -z "${WORN_CKPT}" ] || [ ! -f "${WORN_CKPT}" ]; then
    echo "ERROR: no worn encoder checkpoint found (output/frac_seg_worn_*/feature_extractor_worn.ckpt)"; exit 1
fi
echo "=== Exp 12: Juglet assembly with worn encoder ==="; date
echo "worn encoder: ${WORN_CKPT}"

MERGED=output/GARF_worn_encoder.ckpt
python scripts/merge_worn_encoder.py --base output/GARF.ckpt --worn "${WORN_CKPT}" --out "${MERGED}"

PAIR_HDF5=input/juglet_pairs_local02.hdf5
STAMP=$(date +%Y%m%d_%H%M%S)
TAGROOT="exp12_wornpairs_${STAMP}"
PFPP_DIR=/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0
MESH_DIR=/data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000

RUN_DIRS=()
for SEED in 41 42 43; do
    NAME="${TAGROOT}_s${SEED}"
    echo "--- eval all pairs with worn encoder (seed=${SEED}) ---"
    python eval.py \
        seed="${SEED}" \
        experiment=denoiser_flow_matching \
        experiment_name="${NAME}" \
        loggers=csv loggers.csv.save_dir=logs/deploy \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${PAIR_HDF5}" \
        data.categories="['artifact']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 \
        data.multi_ref=False \
        ckpt_path="${MERGED}" \
        ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=true \
        ++model.inference_config.save_assembly=true
    RUN_DIRS+=("logs/deploy/${NAME}")
done

OUT=logs/diagnostics/pair_chamfer_juglet_worn_${STAMP}
echo "=== symmetry-invariant chamfer scoring (worn encoder) ==="
python scripts/pair_reference_chamfer.py juglet \
    --run-dirs "${RUN_DIRS[@]}" \
    --pairs-hdf5 "${PAIR_HDF5}" \
    --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
    --pfpp-dir "${PFPP_DIR}" \
    --mesh-dir "${MESH_DIR}" \
    --out "${OUT}"

echo "=== full 9-piece Juglet deploy with worn encoder (GLB) ==="
python eval.py \
    seed=42 experiment=denoiser_flow_matching \
    experiment_name="exp12_j9pc_${STAMP}" \
    loggers=csv loggers.csv.save_dir=logs/deploy \
    trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
    data.data_root=input/juglet_deploy_local02.hdf5 \
    data.categories="['artifact']" \
    data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path="${MERGED}" \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true \
    ++model.inference_config.anchor_free=true \
    ++model.inference_config.deploy_mode=true \
    ++model.inference_config.save_assembly=true || echo "9pc deploy failed (non-fatal)"

echo "=== RESULT vs baseline (Exp 6: true-mate 0.070 == non-mate 0.073) ==="
sed -n '1,12p' "${OUT}/summary.md"
echo "=== DONE -> ${OUT}/summary.md ==="; date
