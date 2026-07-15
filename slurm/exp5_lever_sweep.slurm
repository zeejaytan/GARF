#!/bin/bash
#SBATCH --job-name=garf-levers
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp5_lever_sweep_%j.out

# Exp 5 - Inference lever sweep on UNSUCCESSFUL thin-wall Fractura objects (with GT).
#
# Question: are there inference-time knobs (flow-matching / sampler / schedule)
# that change the outcome on the objects GARF currently fails on?
#
# Test set (real scans WITH GT, so better/worse is measured objectively):
#   FAIL  : ceramics/narrow_bottle1 (0.08), narrow_bottle3 (0.50), plate (0.72),
#           egg/egg1 (0.33), egg2 (0.20), egg3 (0.25)
#   CTRL  : ceramics/pink_bowl, narrow_bottle2, blue_pot, narrow_bottle4 (>=0.92)
# Run categories egg + ceramics; per-object metrics come from json_results.
#
# Method: change ONE lever at a time vs the exp4c baseline, 3 seeds each.
# Baseline = init1, anchor_free=true, steps20, max_iters=1, deterministic(ODE), sigma=linear.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="lever_${STAMP}"
SRC=input/Fractura/fractura_real.hdf5
OUT="logs/diagnostics/${PREFIX}"
mkdir -p "$OUT"

echo "Date: $(date)"; echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"; echo "PREFIX=$PREFIX"

# Map a config label -> extra hydra overrides (one lever changed vs baseline).
cfg_overrides () {
    case "$1" in
        baseline)   echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1" ;;
        init0)      echo "++model.inference_config.one_step_init=false ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1" ;;
        steps50)    echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=50 ++model.inference_config.max_iters=1" ;;
        steps100)   echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=100 ++model.inference_config.max_iters=1" ;;
        iters2)     echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=2" ;;
        iters3)     echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=3" ;;
        anchor_on)  echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=false ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1" ;;
        sde)        echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1 ++model.val_noise_scheduler.stochastic_paths=true ++model.val_noise_scheduler.stochastic_level=0.1" ;;
        sde_hi)     echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1 ++model.val_noise_scheduler.stochastic_paths=true ++model.val_noise_scheduler.stochastic_level=0.2" ;;
        sigma_exp)  echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1 ++model.val_noise_scheduler.sigma_schedule=exponential" ;;
        sigma_pwl)  echo "++model.inference_config.one_step_init=true ++model.inference_config.anchor_free=true ++model.inference_config.num_inference_steps=20 ++model.inference_config.max_iters=1 ++model.val_noise_scheduler.sigma_schedule=piecewise-linear" ;;
        *) echo "BAD_CFG" ; return 1 ;;
    esac
}

run_bench () {
    local cfg="$1"; local seed="$2"; local cat="$3"
    local exp="${PREFIX}_${cfg}_${cat}_s${seed}"
    local extra; extra=$(cfg_overrides "$cfg")
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${seed}" experiment=denoiser_flow_matching experiment_name="${exp}" \
        loggers=csv loggers.csv.save_dir=logs/diagnostics \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${SRC}" data.categories="['${cat}']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 data.multi_ref=False \
        ckpt_path="${CKPT}" ++data.random_anchor=false \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.deploy_mode=false \
        ++model.inference_config.save_assembly=false \
        ${extra}
}

CONFIGS="baseline init0 steps50 steps100 iters2 iters3 anchor_on sde sde_hi sigma_exp sigma_pwl"

for CFG in $CONFIGS; do
    for CAT in ceramics egg; do
        for SEED in 41 42 43; do
            echo "=== cfg=${CFG} cat=${CAT} seed=${SEED} ==="
            run_bench "${CFG}" "${SEED}" "${CAT}"
        done
    done
done

echo "=== aggregate ==="
python scripts/summarize_lever_sweep.py --root logs/diagnostics --prefix "$PREFIX" --out "$OUT"
echo "Summary dir: $OUT"
echo "=== DONE ==="
