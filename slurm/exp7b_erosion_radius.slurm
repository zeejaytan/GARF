#!/bin/bash
#SBATCH --job-name=garf-exp7b-radius
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp7b_erosion_radius_%j.out

# Exp 7b — RIM-EROSION BRIDGE, KERNEL-RADIUS SWEEP (break the relief plateau).
#
# Exp 7 (job 27038450) came back NOT confirmed, but with a method caveat: the
# achieved fracture-band relief_p90 SATURATED at ~0.201 for strengths 0.5-1.0
# and never reached Juglet's measured 0.171 — the mollifier stops carving once
# the band is locally smooth at its fixed physical radius (strength * 0.05 *
# piece_scale), and knn=48 additionally capped the effective kernel at large
# radii (only the 48 nearest of ~20k surface samples are averaged, so the
# kernel silently shrinks). Exp 7's negative verdict therefore does not yet
# rule out full-depth wear as the mechanism.
#
# This sweep holds strength=1.0 (full blend) and grows the PHYSICAL kernel
# radius past the plateau: kernel_frac in {0.10, 0.15, 0.20} with knn scaled
# ~quadratically (192/432/768) so the neighbour budget matches the radius.
# The bridge curve is indexed by effective mollify radius = strength *
# kernel_frac (exp7's strength-1.0 arm = radius 0.05), reusing exp7's already
# scored radius-0.00 and radius-0.05 entries as the curve's left end.
#
#   CONFIRMED if, once relief_p90 actually reaches <= Juglet's 0.171, the
#   true-mate chamfer collapses toward Juglet's 0.070 and mate/non separation
#   dies. If relief reaches Juglet's level and chamfer still plateaus well
#   below 0.070, full-depth wear is ruled out as the sufficient mechanism.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

SRC_HDF5=input/Fractura/fractura_real.hdf5
CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="logs/diagnostics/exp7b_radius_${STAMP}"
mkdir -p "$OUT_ROOT"

# Exp 7 (2026-07-10) already scored the radius-0.00 (no erosion) and
# radius-0.05 (strength 1.0, kernel 0.05) points of the same bridge curve.
EXP7_ROOT="logs/diagnostics/exp7_erosion_20260710_141409"
for f in "$EXP7_ROOT/chamfer_e000/pairs.json" "$EXP7_ROOT/adjacency_e000.json" \
         "$EXP7_ROOT/chamfer_e100/pairs.json" "$EXP7_ROOT/adjacency_e100.json"; do
    [[ -f $f ]] || { echo "missing exp7 baseline artifact: $f" >&2; exit 1; }
done

echo "=== Exp 7b: rim-erosion bridge, kernel-radius sweep ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

SUMMARY_ENTRIES=(
    --entry 0.00 "$EXP7_ROOT/chamfer_e000" "$EXP7_ROOT/adjacency_e000.json"
    --entry 0.05 "$EXP7_ROOT/chamfer_e100" "$EXP7_ROOT/adjacency_e100.json"
)

# radius (= strength 1.0 * kernel_frac) / filename tag / knn budget
for RT in "0.10 r010 192" "0.15 r015 432" "0.20 r020 768"; do
    read -r KFRAC TAG KNN <<< "$RT"
    PAIR_HDF5="input/control_ceramics_pairs_${TAG}.hdf5"
    ADJ="${OUT_ROOT}/adjacency_${TAG}.json"
    CHAMFER_OUT="${OUT_ROOT}/chamfer_${TAG}"

    echo ""
    echo "=== mollify radius ${KFRAC} (knn ${KNN}) ==="
    python scripts/build_control_pairs_hdf5.py \
        --source "$SRC_HDF5" \
        --objects ceramics/pink_bowl ceramics/narrow_bottle2 \
                  ceramics/narrow_bottle4 ceramics/blue_pot \
        --out "$PAIR_HDF5" \
        --category control \
        --adjacency-out "$ADJ" \
        --erode-strength 1.0 \
        --erode-kernel-frac "$KFRAC" \
        --erode-knn "$KNN"

    RUN_DIRS=()
    for SEED in 41 42 43; do
        NAME="exp7b_radius_${STAMP}_${TAG}_s${SEED}"
        echo "--- eval eroded pairs (radius=${KFRAC}, seed=${SEED}) ---"
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

    echo "--- symmetry-invariant chamfer scoring (radius=${KFRAC}) ---"
    python scripts/pair_reference_chamfer.py control \
        --run-dirs "${RUN_DIRS[@]}" \
        --pairs-hdf5 "$PAIR_HDF5" \
        --adjacency "$ADJ" \
        --out "$CHAMFER_OUT"

    SUMMARY_ENTRIES+=(--entry "$KFRAC" "$CHAMFER_OUT" "$ADJ")

    # each eroded pairs file is ~1.2G; reference geometry has been consumed
    # by the chamfer scoring above, so drop it to keep input/ clean.
    rm -f "$PAIR_HDF5"
done

echo ""
echo "=== Exp 7b bridge summary ==="
python scripts/summarize_erosion_bridge.py \
    "${SUMMARY_ENTRIES[@]}" \
    --x-label "mollify radius / piece scale" \
    --out "$OUT_ROOT"

echo "Summary: ${OUT_ROOT}/summary.md"
echo "=== DONE ==="
