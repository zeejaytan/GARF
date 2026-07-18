#!/bin/bash
#SBATCH --job-name=garf_exp15_selfsup
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=exp15_selfsup_%j.out

# Exp 15 — Juglet-domain self-supervised encoder adaptation (the last
# in-environment lever). Exp 14 ruled out sufficiency-by-geometric-inversion:
# re-triggering the fracture head on amplified worn texture does not restore
# mating. Surviving hypothesis (a): representation mismatch — the worn geometry
# still carries mating signal, but only an encoder adapted to REAL worn ceramic
# texture can extract it. Test: fine-tune FracSeg on Juglet's own sherds with
# relief-detector pseudo fracture labels (geometric, no circularity) mixed ~1:4
# with the worn-augmented synthetic replay stream; merge; co-adapt the denoiser
# (Exp 13 recipe, synthetic-only — Juglet has no GT poses); re-run the pairwise
# oracle + 9-piece deploy.
# Gates: guard = synthetic probe AUC >= 0.90 (no forgetting); decisive = Juglet
# true-mate median chamfer <= 0.045 AND mate/non separation >= 1.25x
# (baseline 0.070 == 0.073; Exp 13 best so far 0.0645 vs 0.0671, no separation).
# If this fails, hypothesis (b) stands: abrasion destroyed the complementarity
# signal — real weathered supervision or non-fracture cues are required.
#
# Env overrides: EPOCHS_SEG (60), EPOCHS_DEN (40), REPLICAS (80), INIT_CKPT.

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

STAMP=$(date +%Y%m%d_%H%M%S)
EPOCHS_SEG="${EPOCHS_SEG:-60}"
EPOCHS_DEN="${EPOCHS_DEN:-40}"
REPLICAS="${REPLICAS:-80}"
MESH_DIR=/data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000
PFPP_DIR=/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0
echo "=== Exp 15 self-supervised Juglet adaptation ==="; date

# ---------- Stage 1: training mix (synthetic replay + pseudo-labeled Juglet) ----------
MIX=input/juglet_selfsup_mix.hdf5
if [ ! -f "${MIX}" ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
    python scripts/build_juglet_selfsup_hdf5.py \
        --juglet-source input/juglet_deploy_local02.hdf5 \
        --synth-source input/Fractura/bone_synthetic.hdf5 \
        --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
        --replicas "${REPLICAS}" \
        --out "${MIX}"
fi

# ---------- Stage 2: FracSeg adaptation (warm-start from the Exp 11 worn encoder) ----------
INIT_CKPT="${INIT_CKPT:-$(ls -t output/frac_seg_worn_*/feature_extractor_worn.ckpt 2>/dev/null | head -1)}"
[ -f "${INIT_CKPT}" ] || INIT_CKPT=output/feature_extractor.ckpt
echo "init encoder: ${INIT_CKPT}"
SEG_DIR=output/frac_seg_selfsup_${STAMP}
python scripts/finetune_frac_seg.py \
    --init-ckpt "${INIT_CKPT}" \
    --data-root "${MIX}" \
    --categories pig hip leg rib vertebra artifact \
    --out-dir "${SEG_DIR}" \
    --epochs "${EPOCHS_SEG}" --batch-size 16 --lr 2e-5 \
    --erode-prob 0.6
SS_CKPT=${SEG_DIR}/feature_extractor_worn.ckpt

# ---------- Stage 3: probe gate (forgetting guard + perception readout) ----------
PROBE=logs/diagnostics/exp15_probe_${STAMP}
python scripts/fracseg_introspection.py \
    --ckpt "${SS_CKPT}" \
    --synth-source input/Fractura/bone_synthetic.hdf5 \
    --synth-objects pig/synthetic_fracture/20/mode_5 \
                    pig/synthetic_fracture/20/mode_4 \
                    pig/synthetic_fracture/20/mode_2 \
    --control-source input/Fractura/fractura_real.hdf5 \
    --control-objects ceramics/pink_bowl ceramics/blue_pot \
    --juglet-mesh-dir "${MESH_DIR}" \
    --out "${PROBE}" \
    --n-per-piece 5000
GATE=$(python - "${PROBE}" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
auc, jf = s["synth_mean_auc_stored"], s["juglet_fired_frac"]
print(f"auc={auc:.3f} juglet_fired={jf*100:.2f}%", file=sys.stderr)
print("PASS" if auc >= 0.90 else "FAIL")
PY
)
echo "=== probe gate: ${GATE} (guard: synth AUC >= 0.90) ==="
sed -n '1,20p' "${PROBE}/summary.md" || true
if [ "${GATE}" = "FAIL" ]; then
    echo "=== VERDICT: forgetting guard FAILED — adapted encoder lost synthetic"
    echo "    fracture competence. Rerun with fewer epochs / lower lr before any"
    echo "    assembly conclusions. Stopping."
    exit 0
fi

# ---------- Stage 4: merge adapted encoder into GARF ----------
MERGED=output/GARF_selfsup_encoder.ckpt
python scripts/merge_worn_encoder.py --base output/GARF.ckpt \
    --worn "${SS_CKPT}" --out "${MERGED}"

# ---------- Stage 5: denoiser co-adaptation (synthetic-only — Juglet has no GT poses) ----------
COEXP="exp15_coadapt_${STAMP}"
python train.py \
    experiment=denoiser_flow_matching \
    experiment_name="${COEXP}" \
    seed=42 \
    loggers=csv \
    trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
    trainer.max_epochs=${EPOCHS_DEN} \
    +trainer.limit_val_batches=0 \
    +trainer.num_sanity_val_steps=0 \
    ckpt_path="${MERGED}" \
    finetuning=true \
    model.feature_extractor_ckpt=null \
    model.optimizer.lr=1e-5 \
    callbacks.model_checkpoint.save_on_train_epoch_end=true \
    data.data_root=input/Fractura/bone_synthetic.hdf5 \
    data.categories="[pig,hip,leg,rib,vertebra]" \
    data.batch_size=16 \
    data.num_workers=12 \
    data.frac_erode_prob=0.6
CKPT="output/${COEXP}/last.ckpt"
echo "co-adapted checkpoint: ${CKPT}"

# ---------- Stage 6: decisive pairwise oracle + 9-piece deploy ----------
PAIR_HDF5=input/juglet_pairs_local02.hdf5
RUN_DIRS=()
for SEED in 41 42 43; do
    NAME="exp15_pairs_${STAMP}_s${SEED}"
    echo "--- eval pairs (self-sup adapted, seed=${SEED}) ---"
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

OUT=logs/diagnostics/pair_chamfer_juglet_selfsup_${STAMP}
python scripts/pair_reference_chamfer.py juglet \
    --run-dirs "${RUN_DIRS[@]}" \
    --pairs-hdf5 "${PAIR_HDF5}" \
    --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
    --pfpp-dir "${PFPP_DIR}" --mesh-dir "${MESH_DIR}" \
    --out "${OUT}"

echo "=== 9-piece Juglet deploy (self-sup adapted, GLB) ==="
python eval.py \
    seed=42 experiment=denoiser_flow_matching \
    experiment_name="exp15_j9pc_${STAMP}" \
    loggers=csv loggers.csv.save_dir=logs/deploy \
    trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
    data.data_root=input/juglet_deploy_local02.hdf5 \
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

echo "=== RESULT (baseline 0.070==0.073; Exp 13 0.0645 vs 0.0671; gates <=0.045 + >=1.25x) ==="
sed -n '1,12p' "${OUT}/summary.md"
echo "=== DONE -> ${OUT}/summary.md ==="; date
