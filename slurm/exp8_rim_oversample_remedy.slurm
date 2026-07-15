#!/bin/bash
#SBATCH --job-name=garf-exp8-rim
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp8_rim_oversample_remedy_%j.out

# Exp 8 — FRACTURE-RIM OVERSAMPLING REMEDY test (Juglet root cause fix,
# per JUGLET_ROOTCAUSE_FINDINGS.md).
#
# Mechanism being remedied: GARF's encoder gets no usable mating signal from
# Juglet's worn fracture rims; area/poisson sampling additionally starves the
# thin rim band of points. Remedy: data.rim_oversample_frac forces a fraction
# of each part's point budget onto the geometrically detected rim band
# (relief-at-physical-radius detection, assembly/data/breaking_bad/base.py).
#
# Tests, all with the identical pairwise-oracle protocol (deploy, anchor_free,
# seeds 41/42/43, symmetry-invariant chamfer):
#   1. Juglet 36 pairs at frac 0.35 and 0.60  (baseline: mates 0.073 = non 0.073)
#   2. Control ceramics 22 pairs at frac 0.35 (regression: mates must stay ~0.024)
#   3. Full 9-piece Juglet deploy at the sweep fracs (qualitative GLB output)

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
JUGLET_PAIRS=input/juglet_pairs_local02.hdf5
CONTROL_PAIRS=input/control_ceramics_pairs.hdf5
JUGLET_FULL=input/juglet_deploy_local02.hdf5
PFPP_DIR=/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0
MESH_DIR=/data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="logs/diagnostics/exp8_rim_${STAMP}"
mkdir -p "$OUT_ROOT"

echo "=== Exp 8: fracture-rim oversampling remedy ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

run_eval () {  # NAME DATA_ROOT CATEGORIES SEED FRAC
    local NAME=$1 DATA_ROOT=$2 CATS=$3 SEED=$4 FRAC=$5
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${SEED}" \
        experiment=denoiser_flow_matching \
        experiment_name="${NAME}" \
        loggers=csv \
        loggers.csv.save_dir=logs/deploy \
        trainer.num_nodes=1 \
        trainer.devices=1 \
        trainer.precision=bf16-mixed \
        data.data_root="${DATA_ROOT}" \
        data.categories="${CATS}" \
        data.min_parts=2 \
        data.max_parts=20 \
        data.batch_size=1 \
        data.num_workers=4 \
        data.multi_ref=False \
        data.rim_oversample_frac="${FRAC}" \
        ckpt_path="${CKPT}" \
        ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=true \
        ++model.inference_config.save_assembly=true
}

SUMMARY_ARGS=()

# --- 1. Juglet pairs at each oversampling fraction --------------------------
for FT in "0.35 035" "0.60 060"; do
    read -r FRAC TAG <<< "$FT"
    RUN_DIRS=()
    for SEED in 41 42 43; do
        NAME="exp8_jpairs_${STAMP}_f${TAG}_s${SEED}"
        echo "--- Juglet pairs (frac=${FRAC}, seed=${SEED}) ---"
        run_eval "$NAME" "$JUGLET_PAIRS" "['artifact']" "$SEED" "$FRAC"
        RUN_DIRS+=("logs/deploy/${NAME}")
    done
    CHAMFER_OUT="${OUT_ROOT}/juglet_chamfer_f${TAG}"
    python scripts/pair_reference_chamfer.py juglet \
        --run-dirs "${RUN_DIRS[@]}" \
        --pairs-hdf5 "$JUGLET_PAIRS" \
        --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
        --pfpp-dir "$PFPP_DIR" \
        --mesh-dir "$MESH_DIR" \
        --out "$CHAMFER_OUT"
    SUMMARY_ARGS+=(--juglet-entry "$FRAC" "$CHAMFER_OUT")
done

# --- 2. Control ceramics pairs regression at frac 0.35 ----------------------
RUN_DIRS=()
for SEED in 41 42 43; do
    NAME="exp8_cpairs_${STAMP}_f035_s${SEED}"
    echo "--- Control pairs regression (frac=0.35, seed=${SEED}) ---"
    run_eval "$NAME" "$CONTROL_PAIRS" "['control']" "$SEED" "0.35"
    RUN_DIRS+=("logs/deploy/${NAME}")
done
CHAMFER_OUT="${OUT_ROOT}/control_chamfer_f035"
python scripts/pair_reference_chamfer.py control \
    --run-dirs "${RUN_DIRS[@]}" \
    --pairs-hdf5 "$CONTROL_PAIRS" \
    --adjacency logs/diagnostics/control_ceramics_adjacency.json \
    --out "$CHAMFER_OUT"
SUMMARY_ARGS+=(--control-entry "0.35" "$CHAMFER_OUT")

# --- 3. Full 9-piece Juglet deploy at each fraction (qualitative) -----------
for FT in "0.35 035" "0.60 060"; do
    read -r FRAC TAG <<< "$FT"
    NAME="exp8_j9pc_${STAMP}_f${TAG}"
    echo "--- Full 9-piece Juglet (frac=${FRAC}, seed=42) ---"
    run_eval "$NAME" "$JUGLET_FULL" "['artifact']" 42 "$FRAC"
    echo "GLB: logs/deploy/${NAME}/version_0/assembly_results/"
done

echo ""
echo "=== Exp 8 remedy summary ==="
python scripts/summarize_rim_remedy.py \
    "${SUMMARY_ARGS[@]}" \
    --out "$OUT_ROOT"

echo "Summary: ${OUT_ROOT}/summary.md"
echo "=== DONE ==="
