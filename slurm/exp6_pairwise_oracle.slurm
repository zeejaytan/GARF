#!/bin/bash
#SBATCH --job-name=garf-exp6-pairs
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp6_pairwise_oracle_%j.out

# Exp 6 — pairwise oracle decomposition for "why GARF fails on Juglet".
# Runs GARF (deploy mode, anchor_free) on every 2-piece Juglet subset
# (36 pairs, one HDF5) x 3 seeds, then probes each pair:
#   contact / interpenetration (watertight meshes -> meaningful) +
#   cross-seed relative-pose stability of the single pair.
#
#   no pair ever mates            -> perception failure (no rim signal)
#   true pairs mate, 9pc fails    -> joint-inference/search failure
#
# Final judgment: GLB gallery (viz_output/exp6_pairs_*) + physical adjacency.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

SRC_HDF5=input/juglet_deploy_local02.hdf5
PAIR_HDF5=input/juglet_pairs_local02.hdf5
CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
TAGROOT="exp6_pairs_${STAMP}"
PROBE_OUT="logs/diagnostics/pairs_${STAMP}"
mkdir -p "$PROBE_OUT"

echo "=== Exp 6: Juglet pairwise oracle ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Build (or rebuild) the 36-pair HDF5 from the canonical rebuilt geometry.
python scripts/build_juglet_pairs_hdf5.py \
    --source "$SRC_HDF5" \
    --sample artifact/Juglet-000 \
    --out "$PAIR_HDF5"

RUN_DIRS=()
for SEED in 41 42 43; do
    NAME="${TAGROOT}_s${SEED}"
    echo "--- eval all pairs (seed=${SEED}) ---"
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
        data.categories="['artifact']" \
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

echo "=== Pairwise probe summary ==="
python scripts/summarize_pairwise_oracle.py \
    --run-dirs "${RUN_DIRS[@]}" \
    --out "$PROBE_OUT"

echo "=== GLB gallery (seed 42) ==="
python generate_viz.py \
    --results_dir "logs/deploy/${TAGROOT}_s42/version_0/json_results" \
    --hdf5 "$PAIR_HDF5" \
    --all --output_dir "viz_output/${TAGROOT}" \
    --proposed_only --aug_seed 42 || echo "viz generation failed (non-fatal)"

echo "Summary: ${PROBE_OUT}/summary.md"
echo "Gallery: viz_output/${TAGROOT}/index.html"
echo "=== DONE ==="
