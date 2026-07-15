#!/bin/bash
#SBATCH --job-name=garf_tray_eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=eval_tray_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

EXPERIMENT_NAME="tray_archaeological_$(date +%Y%m%d_%H%M%S)"

echo "=== GARF Tray Archaeological Eval ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=input/tray_archaeological.hdf5 \
    data.categories="['artifact']" \
    data.min_parts=2 \
    data.max_parts=50 \
    data.batch_size=1 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=output/GARF.ckpt \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

RESULTS_DIR="logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo "Results: $RESULTS_DIR"

if [ -d "$RESULTS_DIR" ]; then
    python generate_viz.py \
        --results_dir "$RESULTS_DIR" \
        --hdf5 input/tray_archaeological.hdf5 \
        --all --output_dir viz_output/tray_archaeological
    echo "Viz: viz_output/tray_archaeological/index.html"
fi

echo "=== DONE ==="
