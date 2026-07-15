#!/bin/bash
#SBATCH --job-name=garf-exp7-erode
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp7_rim_erosion_bridge_%j.out

# Exp 7 — RIM-EROSION DOMAIN BRIDGE (decisive confirmation of the Juglet
# root cause, per JUGLET_ROOTCAUSE_FINDINGS.md).
#
# Mechanism under test: GARF's Juglet failure is a pairwise PERCEPTION failure
# on worn archaeological fracture rims. Test: take the known-good control
# ceramics (true-mate pairwise chamfer 0.024 vs non-mates 0.039), erode their
# TRUE fracture contact bands toward Juglet's worn relief level (relief_p90
# ~0.171) with GT pose preserved, and re-run the identical pairwise oracle.
#   CONFIRMED if true-mate chamfer collapses toward Juglet's 0.070 and the
#   mate/non-mate separation vanishes as relief approaches Juglet's level.
# Calibration (login-node probe, blue_pot): strength 0.5 lands mean relief_p90
# almost exactly at Juglet's 0.171; strength 1.0 overshoots to ~0.112.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

SRC_HDF5=input/Fractura/fractura_real.hdf5
CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="logs/diagnostics/exp7_erosion_${STAMP}"
mkdir -p "$OUT_ROOT"

echo "=== Exp 7: rim-erosion domain bridge (control ceramics pairs) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

SUMMARY_ENTRIES=()

# strength / filename tag pairs (tag = strength*100, zero padded)
for ST in "0.00 000" "0.25 025" "0.50 050" "0.75 075" "1.00 100"; do
    read -r STRENGTH TAG <<< "$ST"
    PAIR_HDF5="input/control_ceramics_pairs_erode${TAG}.hdf5"
    ADJ="${OUT_ROOT}/adjacency_e${TAG}.json"
    CHAMFER_OUT="${OUT_ROOT}/chamfer_e${TAG}"

    echo ""
    echo "=== strength ${STRENGTH} (tag ${TAG}) ==="
    python scripts/build_control_pairs_hdf5.py \
        --source "$SRC_HDF5" \
        --objects ceramics/pink_bowl ceramics/narrow_bottle2 \
                  ceramics/narrow_bottle4 ceramics/blue_pot \
        --out "$PAIR_HDF5" \
        --category control \
        --adjacency-out "$ADJ" \
        --erode-strength "$STRENGTH"

    RUN_DIRS=()
    for SEED in 41 42 43; do
        NAME="exp7_erode_${STAMP}_e${TAG}_s${SEED}"
        echo "--- eval eroded pairs (strength=${STRENGTH}, seed=${SEED}) ---"
        HYDRA_FULL_ERROR=1 python eval.py \
            seed="${SEED}" \
            experiment=denoiser_flow_matching \
            experiment_name="${NAME}" \
            loggers=csv \
            loggers.csv.save_dir=logs/deploy \
            trainer.num_nodes=1 \
            trainer.devices=1 \
            trainer.precision=bf16-mixed \
            data.data_root="${PAIR_HDF5}" \
            data.categories="['control']" \
            data.min_parts=2 \
            data.max_parts=20 \
            data.batch_size=1 \
            data.num_workers=4 \
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

    echo "--- symmetry-invariant chamfer scoring (strength=${STRENGTH}) ---"
    python scripts/pair_reference_chamfer.py control \
        --run-dirs "${RUN_DIRS[@]}" \
        --pairs-hdf5 "$PAIR_HDF5" \
        --adjacency "$ADJ" \
        --out "$CHAMFER_OUT"

    SUMMARY_ENTRIES+=(--entry "$STRENGTH" "$CHAMFER_OUT" "$ADJ")

    # each eroded pairs file is ~1.2G; reference geometry has been consumed
    # by the chamfer scoring above, so drop it to keep input/ clean.
    rm -f "$PAIR_HDF5"
done

echo ""
echo "=== Exp 7 bridge summary ==="
python scripts/summarize_erosion_bridge.py \
    "${SUMMARY_ENTRIES[@]}" \
    --out "$OUT_ROOT"

echo "Summary: ${OUT_ROOT}/summary.md"
echo "=== DONE ==="
