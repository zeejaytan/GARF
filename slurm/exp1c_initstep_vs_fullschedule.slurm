#!/bin/bash
#SBATCH --job-name=garf-initcmp
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp1c_initstep_vs_fullschedule_%j.out

# Exp 1c — does one_step_init vs full-schedule change correctness on GT objects?
# Runs BB (6pc, 9pc) and Fractura (pig 9pc) in BENCHMARK mode (deploy_mode=false,
# so part_acc/rmse/shape_cd are real), for one_step_init in {true,false}, across
# seeds, then aggregates. This tells us if our deploy inference config is a
# confound before we attribute Juglet's failure to geometry.
#
# NOTE: one_step_init=true = one warmup step THEN the full 20-step schedule.
#       one_step_init=false = full 20-step schedule only.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="bench_cmp_${STAMP}"
SUBSET_DIR="logs/diagnostics/initcmp_subsets/${STAMP}"
OUT="logs/diagnostics/initcmp_${STAMP}"
mkdir -p "$SUBSET_DIR" "$OUT"

BB_SRC=/data/gpfs/projects/punim2657/TORA/dataset/breaking_bad_vol.hdf5
FR_SRC=input/Fractura/bone_synthetic.hdf5

echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "PREFIX=$PREFIX"

# args: exp_name seed hdf5 one_step_init
run_bench () {
    local exp_name="$1"; local seed="$2"; local hdf5="$3"; local osi="$4"
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${seed}" \
        experiment=denoiser_flow_matching \
        experiment_name="${exp_name}" \
        loggers=csv \
        loggers.csv.save_dir=logs/diagnostics \
        trainer.num_nodes=1 \
        trainer.devices=1 \
        trainer.precision=bf16-mixed \
        data.data_root="${hdf5}" \
        data.categories="['diag']" \
        data.min_parts=2 \
        data.max_parts=20 \
        data.batch_size=1 \
        data.num_workers=4 \
        data.multi_ref=False \
        ckpt_path="${CKPT}" \
        ++data.random_anchor=false \
        ++model.inference_config.one_step_init="${osi}" \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=false \
        ++model.inference_config.save_assembly=false
}

# args: label source split pieces
do_label () {
    local label="$1"; local src="$2"; local key="$3"; local pieces="$4"
    local subset="${SUBSET_DIR}/${label}.hdf5"
    echo "=== build subset ${label} (${pieces} pc) ==="
    python scripts/build_single_sample_subset.py \
        --source "$src" --split-key "$key" --pieces "$pieces" --out "$subset"
    for SEED in 41 42 43; do
        echo "--- ${label} init1 seed=${SEED} ---"
        run_bench "${PREFIX}_${label}_init1_s${SEED}" "$SEED" "$subset" "true"
        echo "--- ${label} init0 seed=${SEED} ---"
        run_bench "${PREFIX}_${label}_init0_s${SEED}" "$SEED" "$subset" "false"
    done
}

do_label "bb_6pc"          "$BB_SRC" "everyday" 6
do_label "bb_9pc"          "$BB_SRC" "everyday" 9
do_label "fractura_pig_9pc" "$FR_SRC" "pig"      9

echo "=== aggregate ==="
python scripts/summarize_benchmark_metrics.py \
    --root logs/diagnostics --prefix "$PREFIX" --out "$OUT"

echo "Summary: $OUT/summary.md"
echo "=== DONE ==="
