#!/bin/bash
#SBATCH --job-name=garf-bridge2
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp4b_thin_shell_bridge_%j.out

# Exp 4b — thin-shell domain bridge (H1).
# Take solid BB/Fractura objects GARF assembles well, HOLLOW each fragment into a
# thin shell (wall thickness swept around Juglet's thick/scale~0.075, fill~0.13)
# while preserving GT pose, and measure part_acc/shape_cd. If correctness collapses
# at Juglet-like wall thickness, thin-walledness is the mechanism behind Juglet's
# failure. base = untouched sanity gate; must reproduce the known-good baseline.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="bridge2_${STAMP}"
SUBSETS="logs/diagnostics/bridge_subsets/${STAMP}"
OUT="logs/diagnostics/bridge2_${STAMP}"
mkdir -p "$SUBSETS" "$OUT"

BB_SRC=/data/gpfs/projects/punim2657/TORA/dataset/breaking_bad_vol.hdf5
FR_SRC=input/Fractura/bone_synthetic.hdf5

echo "Date: $(date)"; echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"; echo "PREFIX=$PREFIX"

run_bench () {
    local exp="$1"; local seed="$2"; local hdf5="$3"
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${seed}" experiment=denoiser_flow_matching experiment_name="${exp}" \
        loggers=csv loggers.csv.save_dir=logs/diagnostics \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${hdf5}" data.categories="['diag']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 data.multi_ref=False \
        ckpt_path="${CKPT}" ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=false \
        ++model.inference_config.save_assembly=false
}

# args: src_tag source split pieces transform strength tag
build_and_eval () {
    local src="$1"; local source="$2"; local key="$3"; local pieces="$4"; local tf="$5"; local str="$6"; local tag="$7"
    local hdf5="${SUBSETS}/${src}_${tag}.hdf5"
    echo "=== build ${src}_${tag} (transform=${tf} strength=${str}) ==="
    python scripts/domain_bridge.py --source "$source" --split-key "$key" --pieces "$pieces" \
        --transform "$tf" --strength "$str" --seed 0 --out "$hdf5"
    for SEED in 41 42 43; do
        run_bench "${PREFIX}_${src}_${tag}_s${SEED}" "$SEED" "$hdf5"
    done
}

# wall thickness as fraction of frag scale; Juglet ~0.075 (fill~0.13)
sweep_source () {
    local src="$1"; local source="$2"; local key="$3"; local pieces="$4"
    build_and_eval "$src" "$source" "$key" "$pieces" none  0.0  base
    build_and_eval "$src" "$source" "$key" "$pieces" shell 0.08 shell08
    build_and_eval "$src" "$source" "$key" "$pieces" shell 0.06 shell06
    build_and_eval "$src" "$source" "$key" "$pieces" shell 0.04 shell04
    build_and_eval "$src" "$source" "$key" "$pieces" shell 0.02 shell02
}

sweep_source "bb6"   "$BB_SRC" "everyday" 6
sweep_source "frpig" "$FR_SRC" "pig"      9

echo "=== aggregate ==="
python scripts/summarize_bridge_metrics.py --root logs/diagnostics --prefix "$PREFIX" --out "$OUT"
echo "Summary: $OUT/summary.md"
echo "=== DONE ==="
