#!/bin/bash
#SBATCH --job-name=garf-exp6b-ctrl
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp6b_control_pairs_%j.out

# Exp 6b — POSITIVE CONTROL for the Juglet pairwise oracle.
# Runs GARF (deploy mode, anchor_free — identical config to Exp 6) on every
# 2-piece subset of KNOWN-GOOD Fractura ceramics objects (part_acc >= 0.92).
# If the true-mate pairs here show LOW cross-seed relative-pose dispersion /
# real contact while Juglet's do not, the 2-piece proxy is valid and Juglet is
# a genuine perception failure. If these are ALSO ~70 deg dispersed, the proxy
# itself cannot register a mate and Exp 6 is inconclusive.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

SRC_HDF5=input/Fractura/fractura_real.hdf5
PAIR_HDF5=input/control_ceramics_pairs.hdf5
ADJ=logs/diagnostics/control_ceramics_adjacency.json
CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
TAGROOT="exp6b_ctrl_${STAMP}"
PROBE_OUT="logs/diagnostics/ctrl_pairs_${STAMP}"
mkdir -p "$PROBE_OUT"

echo "=== Exp 6b: control (known-good ceramics) pairwise oracle ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# (Re)build the control pairs HDF5 + GT adjacency labels.
python scripts/build_control_pairs_hdf5.py \
    --source "$SRC_HDF5" \
    --objects ceramics/pink_bowl ceramics/narrow_bottle2 \
              ceramics/narrow_bottle4 ceramics/blue_pot \
    --out "$PAIR_HDF5" \
    --category control \
    --adjacency-out "$ADJ"

RUN_DIRS=()
for SEED in 41 42 43; do
    NAME="${TAGROOT}_s${SEED}"
    echo "--- eval control pairs (seed=${SEED}) ---"
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

echo "=== control pairwise probe summary ==="
python scripts/summarize_control_pairs.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --adjacency "$ADJ" \
    --juglet-adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
    --juglet-pairs logs/diagnostics/pairs_20260610_162659/pairs.json \
    --out "$PROBE_OUT"

echo "Summary: ${PROBE_OUT}/summary.md"
echo "=== DONE ==="
