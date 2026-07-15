#!/bin/bash
#SBATCH --job-name=garf-thinwall
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp4c_real_thinwall_%j.out

# Exp 4c — control for the user's challenge: BB/Fractura DO contain real thin-walled
# objects (Fractura_real: egg fill~0.02, ceramics fill~0.08-0.18 ~ Juglet's 0.13).
# These are REAL scans WITH GT. If GARF assembles them well, thin-wall alone is NOT
# the cause and the Exp 4b synthetic-shell collapse was a marching-cubes remesh
# confound. If GARF also fails here, thin-wall (H1) is confirmed on real data.
# bones = same-domain non-thin control.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="thinwall_${STAMP}"
SRC=input/Fractura/fractura_real.hdf5
OUT="logs/diagnostics/thinwall_${STAMP}"
mkdir -p "$OUT"

echo "Date: $(date)"; echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"; echo "PREFIX=$PREFIX"

run_bench () {
    local exp="$1"; local seed="$2"; local cat="$3"
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${seed}" experiment=denoiser_flow_matching experiment_name="${exp}" \
        loggers=csv loggers.csv.save_dir=logs/diagnostics \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${SRC}" data.categories="['${cat}']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 data.multi_ref=False \
        ckpt_path="${CKPT}" ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=false \
        ++model.inference_config.save_assembly=false
}

for CAT in egg ceramics bones; do
    for SEED in 41 42 43; do
        echo "=== ${CAT} seed ${SEED} ==="
        run_bench "${PREFIX}_${CAT}_init1_s${SEED}" "$SEED" "$CAT"
    done
done

echo "=== aggregate ==="
python scripts/summarize_benchmark_metrics.py --root logs/diagnostics --prefix "$PREFIX" --out "$OUT" || \
    echo "(summarizer prefix-mode may differ; raw CSVs under logs/diagnostics/${PREFIX}_*)"
echo "Summary dir: $OUT"
echo "=== DONE ==="
