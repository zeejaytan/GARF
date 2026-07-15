#!/bin/bash
#SBATCH --job-name=garf_bone_eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=eval_bone_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

EXPERIMENT_NAME="bone_eval_$(date +%Y%m%d_%H%M%S)"
HDF5_PATH="input/Fractura/bone_synthetic.hdf5"
CHECKPOINT="output/GARF.ckpt"
CATEGORY="pig"

echo "=== GARF Bone Evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Dataset: $HDF5_PATH ($CATEGORY)"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Step 1: Run evaluation
echo "── Step 1: Running GARF evaluation ──"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/bone_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['$CATEGORY']" \
    data.batch_size=16 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

RESULTS_DIR="logs/bone_eval/$EXPERIMENT_NAME/version_0/json_results"

echo ""
echo "── Step 2: Generating visualizations ──"

if [ -d "$RESULTS_DIR" ]; then
    # Generate all sample pages + index
    python generate_viz.py \
        --results_dir "$RESULTS_DIR" \
        --hdf5 "$HDF5_PATH" \
        --all \
        --output_dir "viz_output/bone_${CATEGORY}" \
        --max_samples 50

    echo ""
    echo "=== EVALUATION + VISUALIZATION COMPLETE ==="
    echo "Results JSON: $RESULTS_DIR"
    echo "Viz pages:    viz_output/bone_${CATEGORY}/"
    echo "Open viz_output/bone_${CATEGORY}/index.html in a browser"
else
    echo "WARNING: Results directory not found: $RESULTS_DIR"
    echo "Check the evaluation output above for errors."
fi
