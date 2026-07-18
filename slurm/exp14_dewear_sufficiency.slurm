#!/bin/bash
#SBATCH --job-name=garf_exp14_dewear
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp14_dewear_%j.out

# Exp 14 — de-weathering sufficiency test ("un-wear the Juglet").
# Proves the encoder-blindness cause in the SUFFICIENCY direction, on the
# failing object itself: invert the causally-validated mollifier wear model
# (surface unsharp mask on the pose-free relief-detected band) and ask
#   arm A: does the frozen encoder's fracture response return?   (probe)
#   arm C: sharpening OFF-band only must NOT help                (specificity)
#   arm B: does pairwise mating return? (3-seed chamfer oracle)  (decisive)
#   + full 9-piece deploy on de-weathered data (GLB).
# Baselines: Juglet fired% 0.57% (control 3.4%); pairwise true-mate 0.070 ==
# non-mate 0.073 (control 0.024 vs 0.039). Success = fired% >= 1.5% AND
# true-mate median <= 0.045 with mate/non separation >= 1.25x.
# If confirmed, the transform doubles as remedy path 1: an inference-time
# de-weathering preprocessor (no retraining). Optional second eval arm stacks
# it with the Exp 13 co-adapted model (COADAPT_CKPT to override autodetect).
# RUN_CTRL_REGRESSION=1 adds arm D (de-weathered control ceramics must stay good).

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

STAMP=$(date +%Y%m%d_%H%M%S)
MESH_DIR=/data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000
PFPP_DIR=/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0
echo "=== Exp 14 de-weathering sufficiency ==="; date

# ---------- Stage A: perception sweep (band) + arm C (offband specificity) ----------
PROBE_BAND=logs/diagnostics/exp14_probe_band_${STAMP}
python scripts/exp14_dewear_sufficiency.py probe \
    --ckpt output/feature_extractor.ckpt \
    --juglet-mesh-dir "${MESH_DIR}" \
    --control-source input/Fractura/fractura_real.hdf5 \
    --control-objects ceramics/pink_bowl ceramics/blue_pot \
    --strengths 0.0 0.5 1.0 2.0 3.0 --region band \
    --out "${PROBE_BAND}"

PROBE_OFF=logs/diagnostics/exp14_probe_offband_${STAMP}
python scripts/exp14_dewear_sufficiency.py probe \
    --ckpt output/feature_extractor.ckpt \
    --juglet-mesh-dir "${MESH_DIR}" \
    --strengths 2.0 --region offband \
    --out "${PROBE_OFF}"

BEST=$(python - "${PROBE_BAND}" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
print(f"{s['best_strength']:.2f}" if s["best_juglet_fired"] >= 0.015 else "NONE")
PY
)
if [ "${BEST}" = "NONE" ]; then
    echo "=== VERDICT: arm A gate FAILED — de-weathering does not restore the"
    echo "    encoder's fracture response. Unsharp amplification cannot recreate"
    echo "    the fresh-break texture class from worn geometry; sufficiency by"
    echo "    geometric inversion is NOT achievable -> proceed to Exp 15"
    echo "    (Juglet-domain self-supervised encoder adaptation)."
    exit 0
fi
echo "=== arm A gate PASSED: best de-weathering strength ${BEST} ==="

# ---------- Stage B: de-weathered pairs, 3-seed oracle, chamfer scoring ----------
DEPLOY_DW=input/juglet_deploy_dewear.hdf5
PAIRS_DW=input/juglet_pairs_dewear.hdf5
python scripts/exp14_dewear_sufficiency.py build \
    --source input/juglet_deploy_local02.hdf5 --strength "${BEST}" --out "${DEPLOY_DW}"
python scripts/build_juglet_pairs_hdf5.py --source "${DEPLOY_DW}" --out "${PAIRS_DW}"

# Model arms: baseline GARF + (optional) Exp 13 co-adapted stack.
declare -A ARMS=( [base]=output/GARF.ckpt )
COADAPT_CKPT="${COADAPT_CKPT:-$(ls -t output/*coadapt*/last.ckpt output/*coadapt*/*/last.ckpt 2>/dev/null | head -1 || true)}"
if [ -n "${COADAPT_CKPT}" ] && [ -f "${COADAPT_CKPT}" ]; then
    ARMS[coadapt]="${COADAPT_CKPT}"
    echo "co-adapt arm enabled: ${COADAPT_CKPT}"
fi

for ARM in "${!ARMS[@]}"; do
    CKPT="${ARMS[$ARM]}"
    RUN_DIRS=()
    for SEED in 41 42 43; do
        NAME="exp14_dwpairs_${ARM}_${STAMP}_s${SEED}"
        echo "--- eval de-weathered pairs (${ARM}, seed=${SEED}) ---"
        python eval.py \
            seed="${SEED}" \
            experiment=denoiser_flow_matching \
            experiment_name="${NAME}" \
            loggers=csv loggers.csv.save_dir=logs/deploy \
            trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
            data.data_root="${PAIRS_DW}" \
            data.categories="['artifact']" \
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

    OUT=logs/diagnostics/pair_chamfer_juglet_dewear_${ARM}_${STAMP}
    echo "=== chamfer scoring (${ARM}) — reference = original geometry ==="
    python scripts/pair_reference_chamfer.py juglet \
        --run-dirs "${RUN_DIRS[@]}" \
        --pairs-hdf5 input/juglet_pairs_local02.hdf5 \
        --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
        --pfpp-dir "${PFPP_DIR}" \
        --mesh-dir "${MESH_DIR}" \
        --out "${OUT}"

    echo "=== 9-piece de-weathered Juglet deploy (${ARM}, GLB) ==="
    python eval.py \
        seed=42 experiment=denoiser_flow_matching \
        experiment_name="exp14_j9pc_${ARM}_${STAMP}" \
        loggers=csv loggers.csv.save_dir=logs/deploy \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${DEPLOY_DW}" \
        data.categories="['artifact']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 \
        data.multi_ref=False \
        ckpt_path="${CKPT}" \
        ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=true \
        ++model.inference_config.save_assembly=true || echo "9pc deploy failed (non-fatal)"

    echo "=== RESULT ${ARM} (baseline: true-mate 0.070 == non-mate 0.073) ==="
    sed -n '1,14p' "${OUT}/summary.md"
done

# ---------- Arm D (optional): control-ceramics regression ----------
if [ "${RUN_CTRL_REGRESSION:-0}" = "1" ]; then
    CTRL_DW=input/control_ceramics_pairs_dewear.hdf5
    python scripts/exp14_dewear_sufficiency.py build \
        --source input/control_ceramics_pairs.hdf5 --strength "${BEST}" --out "${CTRL_DW}"
    RUN_DIRS=()
    for SEED in 41 42 43; do
        NAME="exp14_ctrlpairs_${STAMP}_s${SEED}"
        python eval.py \
            seed="${SEED}" \
            experiment=denoiser_flow_matching \
            experiment_name="${NAME}" \
            loggers=csv loggers.csv.save_dir=logs/deploy \
            trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
            data.data_root="${CTRL_DW}" \
            data.categories="['control']" \
            data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 \
            data.multi_ref=False \
            ckpt_path=output/GARF.ckpt \
            ++data.random_anchor=false \
            ++model.inference_config.one_step_init=true \
            ++model.inference_config.write_to_json=true \
            ++model.inference_config.anchor_free=true \
            ++model.inference_config.deploy_mode=true \
            ++model.inference_config.save_assembly=true
        RUN_DIRS+=("logs/deploy/${NAME}")
    done
    python scripts/pair_reference_chamfer.py control \
        --run-dirs "${RUN_DIRS[@]}" \
        --pairs-hdf5 input/control_ceramics_pairs.hdf5 \
        --adjacency logs/diagnostics/control_ceramics_adjacency.json \
        --out logs/diagnostics/pair_chamfer_ctrl_dewear_${STAMP}
    echo "=== arm D regression (gate: true-mate median <= 0.030) ==="
    sed -n '1,14p' logs/diagnostics/pair_chamfer_ctrl_dewear_${STAMP}/summary.md
fi

echo "=== DONE ==="; date
