#!/bin/bash
#SBATCH --job-name=garf-thinviz
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp4c_viz_%j.out

# Re-run Exp 4c categories with save_assembly=true so GARF exports the assembled
# GLBs (predicted_assembly.glb + gt) for visual inspection of the thin-wall result.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CKPT=output/GARF.ckpt
STAMP=$(date +%Y%m%d_%H%M%S)
SRC=input/Fractura/fractura_real.hdf5

echo "Date: $(date)"; echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for CAT in ceramics egg bones; do
    EXP="thinviz_${STAMP}_${CAT}"
    echo "=== ${CAT} -> ${EXP} ==="
    HYDRA_FULL_ERROR=1 python eval.py \
        seed=41 experiment=denoiser_flow_matching experiment_name="${EXP}" \
        loggers=csv loggers.csv.save_dir=logs/deploy \
        trainer.num_nodes=1 trainer.devices=1 trainer.precision=bf16-mixed \
        data.data_root="${SRC}" data.categories="['${CAT}']" \
        data.min_parts=2 data.max_parts=20 data.batch_size=1 data.num_workers=4 data.multi_ref=False \
        ckpt_path="${CKPT}" ++data.random_anchor=false \
        ++model.inference_config.one_step_init=true \
        ++model.inference_config.write_to_json=true \
        ++model.inference_config.anchor_free=true \
        ++model.inference_config.deploy_mode=false \
        ++model.inference_config.save_assembly=true
    echo "GLB dir: logs/deploy/${EXP}/version_0/assembly_results/"
done
echo "=== DONE ==="
