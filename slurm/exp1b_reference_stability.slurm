#!/bin/bash
#SBATCH --job-name=garf-exp1b-ref
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp1b_reference_stability_%j.out

# Exp 1b — known-good reference baseline for the stability/contact probes.
# Runs BB (6- and 9-piece) and Fractura (9-piece) objects that GARF assembles
# well, across several seeds, with save_assembly=true, then probes them with the
# SAME no-GT metrics used on Juglet. This gives the contrast needed to interpret
# Juglet's dispersion/contact numbers.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
SUBSET_DIR="logs/diagnostics/ref_subsets/${STAMP}"
PROBE_ROOT="logs/diagnostics/probes/exp1b_ref_${STAMP}"
mkdir -p "$SUBSET_DIR" "$PROBE_ROOT"

BB_SRC=/data/gpfs/projects/punim2657/TORA/dataset/breaking_bad_vol.hdf5
FR_SRC=input/Fractura/bone_synthetic.hdf5

run_eval () {
    local exp_name="$1"; local seed="$2"; local hdf5="$3"
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${seed}" \
        experiment=denoiser_flow_matching \
        experiment_name="${exp_name}" \
        loggers=csv \
        loggers.csv.save_dir=logs/deploy \
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
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=true \
        ++model.inference_config.save_assembly=true
}

# args: label  source_hdf5  split_key  pieces
do_reference () {
    local label="$1"; local src="$2"; local key="$3"; local pieces="$4"
    local subset="${SUBSET_DIR}/${label}.hdf5"
    echo "=== Reference ${label} (${pieces} pieces) ==="
    local out
    out=$(python scripts/build_single_sample_subset.py \
            --source "$src" --split-key "$key" --pieces "$pieces" --out "$subset")
    echo "$out"
    local sample
    sample=$(echo "$out" | sed -n 's/^sample=//p')

    local jsons=()
    for SEED in 41 42 43 44; do
        local name="ref_${label}_${STAMP}_s${SEED}"
        echo "--- ${name} (seed=${SEED}) ---"
        run_eval "$name" "$SEED" "$subset"
        jsons+=("logs/deploy/${name}/version_0/assembly_results/${sample}/predicted_assembly.json")
    done

    python scripts/no_gt_probes.py stability \
        --jsons "${jsons[@]}" \
        --out "$PROBE_ROOT" --tag "stability_${label}"

    local glb="logs/deploy/ref_${label}_${STAMP}_s42/version_0/assembly_results/${sample}/predicted_assembly.glb"
    python scripts/no_gt_probes.py contact --glb "$glb" --out "$PROBE_ROOT" --tag "contact_${label}" || true
}

echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

do_reference "bb_6pc"  "$BB_SRC" "everyday" 6
do_reference "bb_9pc"  "$BB_SRC" "everyday" 9
do_reference "fractura_pig_9pc" "$FR_SRC" "pig" 9

echo "Probe outputs: $PROBE_ROOT"
echo "=== DONE ==="
