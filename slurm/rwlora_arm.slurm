#!/bin/bash
#SBATCH --job-name=garf_rwlora
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/rwlora/rwlora_%x_%j.out

# G1 / .scratch/rough-worn-dose ticket 01: one arm of GARF's rough/worn dose sweep.
#   ARM=untouched            base GARF, no training
#   ARM=fresh                u10_generic + u10_ceiling pooled, unaltered
#   ARM=noise_d025 ... d800  rough ; ARM=worn_d025 ... d800  worn
# Train (configs/experiment/rough_worn_lora.yaml, LoRA only), then 20 attempts on the
# Juglet, the 8 Fractura pots and the erosion ladder, anchor = largest sherd at its true
# pose, scored by TORA's own_place on the pot-size ruler, plus the GARF<->own_place
# reconcile gate. SETS overrides the test sets (default all three).
#   sbatch --export=ALL,ARM=fresh slurm/rwlora_arm.slurm   (SEED=7 repeats the training)

set -euo pipefail
ARM="${ARM:?set ARM}"
SETS="${SETS:-juglet_gt fractura_fresh erosion_ceramics}"
DRAWS="${DRAWS:-20}"
SEED="${SEED:-42}"   # training seed only; the 20 attempts are always drawn with seed 42
cd /data/gpfs/projects/punim2657/GARF
source /data/gpfs/projects/punim2657/CRAG/.venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1
DS=/data/gpfs/projects/punim2657/TORA/dataset
TORA=/data/gpfs/projects/punim2657/TORA/repo
STAMP="${SLURM_JOB_ID:-local}"
BASE=output/GARF.ckpt

echo "=== rwlora arm=${ARM} train-seed=${SEED} sets=${SETS} draws=${DRAWS} $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "GARF $(git rev-parse --short HEAD)"

LORA_ARGS=()
if [ "$ARM" != "untouched" ]; then
    EXP="rwlora_${ARM}_s${SEED}_${STAMP}"
    if [ "$ARM" = "fresh" ]; then
        DATA=(data.data_root=$DS/u10_generic.hdf5 "data.categories=[u10_generic,u10_ceiling]"
              "+data.additional_data_root={u10_ceiling:'$DS/u10_ceiling.hdf5'}")
    else
        F="u10_${ARM}"
        [ -f "$DS/$F.hdf5" ] || { echo "no $DS/$F.hdf5"; exit 1; }
        DATA=(data.data_root=$DS/$F.hdf5 "data.categories=[$F]")
    fi
    python train.py \
        experiment=rough_worn_lora experiment_name="$EXP" seed=$SEED \
        loggers=csv loggers.csv.save_dir=logs/rwlora \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        +trainer.limit_val_batches=0 +trainer.num_sanity_val_steps=0 \
        model.feature_extractor_ckpt=null \
        data.num_workers=12 "${DATA[@]}" \
        || { echo "training failed"; exit 1; }
    ADAPTER="output/$EXP/last.ckpt"
    ls -la "$ADAPTER"
    LORA_ARGS=("+lora_ckpt_path=$ADAPTER")
fi

for SET in $SETS; do
    NAME="rwlora_eval_${ARM}_${SET}_${STAMP}"
    python eval.py \
        seed=42 experiment=denoiser_flow_matching experiment_name="$NAME" \
        loggers=csv loggers.csv.save_dir=logs/rwlora \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root=$DS/$SET.hdf5 "data.categories=['$SET']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 \
        data.multi_ref=False \
        model.feature_extractor_ckpt=null \
        ckpt_path=$BASE "${LORA_ARGS[@]}" \
        ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.n_generations=$DRAWS \
        ++model.inference_config.save_clouds=true \
        ++model.inference_config.save_assembly=false \
        || { echo "eval $SET failed"; exit 1; }
    RUN=$(dirname "$(ls -d logs/rwlora/$NAME/*/clouds | head -1)")
    echo "--- $SET run dir: $RUN ($(ls $RUN/clouds | wc -l) objects)"
    python scripts/garf_clouds_reconcile.py "$RUN" --tora $TORA \
        || echo "RECONCILE FAILED on $SET -- do not use these scores"
    POTMM=(); [ "$SET" = "juglet_gt" ] && POTMM=(--pot-mm 65)
    python $TORA/scripts/own_place.py --run "$RUN" "${POTMM[@]}" --json "$RUN/own_place.json" \
        || echo "own_place failed on $SET (clouds kept; score on CPU)"
done
echo "=== done $(date) ==="
