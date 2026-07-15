#!/bin/bash
#SBATCH --job-name=garf-exp1-stab
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp1_seed_stability_%j.out

# Exp 1 — seed/anchor stability discriminator for "why GARF fails on Juglet".
# Runs the SAME Juglet object several times with different seeds (and a
# random-anchor variant), exports predicted_assembly.json each time, then
# measures relative-pose dispersion across runs (gauge-invariant).
#
#   high dispersion  -> model has no reliable mating signal (H1/H2)
#   low dispersion, wrong -> systematic bias (H3 / normals / encoder)

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

HDF5=input/juglet_deploy.hdf5
CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
TAGROOT="exp1_stab_${STAMP}"
PROBE_OUT="logs/diagnostics/probes/${TAGROOT}"
mkdir -p "$PROBE_OUT"

echo "=== Exp 1: Juglet seed/anchor stability ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

run_eval () {
    local exp_name="$1"; local seed="$2"; local rand_anchor="$3"
    echo "--- run ${exp_name} (seed=${seed}, random_anchor=${rand_anchor}) ---"
    HYDRA_FULL_ERROR=1 python eval.py \
        seed="${seed}" \
        experiment=denoiser_flow_matching \
        experiment_name="${exp_name}" \
        loggers=csv \
        loggers.csv.save_dir=logs/deploy \
        trainer.num_nodes=1 \
        trainer.devices=1 \
        trainer.precision=bf16-mixed \
        data.data_root="${HDF5}" \
        data.categories="['artifact']" \
        data.min_parts=2 \
        data.max_parts=20 \
        data.batch_size=1 \
        data.num_workers=4 \
        data.multi_ref=False \
        ckpt_path="${CKPT}" \
        ++data.random_anchor="${rand_anchor}" \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=true \
        ++model.inference_config.save_assembly=true
}

JSONS=()
# 6 fixed-anchor seeds
for SEED in 41 42 43 44 45 46; do
    NAME="${TAGROOT}_s${SEED}"
    run_eval "$NAME" "$SEED" "false"
    JSONS+=("logs/deploy/${NAME}/version_0/assembly_results/artifact/Juglet-000/predicted_assembly.json")
done
# 3 random-anchor seeds (tests anchor sensitivity)
for SEED in 42 43 44; do
    NAME="${TAGROOT}_ra_s${SEED}"
    run_eval "$NAME" "$SEED" "true"
    JSONS+=("logs/deploy/${NAME}/version_0/assembly_results/artifact/Juglet-000/predicted_assembly.json")
done

echo "=== Stability probe (fixed-anchor seeds only) ==="
python scripts/no_gt_probes.py stability \
    --jsons "${JSONS[@]:0:6}" \
    --out "$PROBE_OUT" --tag stability_fixed_anchor

echo "=== Stability probe (all runs incl. random anchor) ==="
python scripts/no_gt_probes.py stability \
    --jsons "${JSONS[@]}" \
    --out "$PROBE_OUT" --tag stability_all

echo "=== Contact probe per run (descriptive context) ==="
for SEED in 41 42 43 44 45 46; do
    NAME="${TAGROOT}_s${SEED}"
    GLB="logs/deploy/${NAME}/version_0/assembly_results/artifact/Juglet-000/predicted_assembly.glb"
    python scripts/no_gt_probes.py contact --glb "$GLB" --out "$PROBE_OUT" --tag "contact_s${SEED}" || true
done

echo "Probe outputs: $PROBE_OUT"
echo "=== DONE ==="
