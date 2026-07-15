#!/bin/bash
#SBATCH --job-name=garf_juglet_local02
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=infer_juglet_deploy_local02_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

EXPERIMENT_NAME="juglet_deploy_local02_$(date +%Y%m%d_%H%M%S)"

echo "=== GARF Juglet local02 deploy (anchor-free, NO ref-part GT snap) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/deploy \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=input/juglet_deploy_local02.hdf5 \
    data.categories="['artifact']" \
    data.min_parts=2 \
    data.max_parts=20 \
    data.batch_size=1 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=output/GARF.ckpt \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true \
    ++model.inference_config.anchor_free=true \
    ++model.inference_config.deploy_mode=true \
    ++model.inference_config.save_assembly=true

RESULTS_DIR="logs/deploy/$EXPERIMENT_NAME/version_0/json_results"
echo "JSON: $RESULTS_DIR"
echo "GLB:  logs/deploy/$EXPERIMENT_NAME/version_0/assembly_results/"

if [ -d "$RESULTS_DIR" ]; then
    python generate_viz.py \
        --results_dir "$RESULTS_DIR" \
        --hdf5 input/juglet_deploy_local02.hdf5 \
        --all --output_dir viz_output/juglet_deploy_local02_proposed \
        --proposed_only --aug_seed 42
    echo "Viz: viz_output/juglet_deploy_local02_proposed/index.html (default: Model proposal)"
fi

echo "=== DONE ==="
