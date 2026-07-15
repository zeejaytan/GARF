#!/bin/bash
#SBATCH --job-name=garf_smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=smoke_test_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

echo "=== SMOKE TEST: GARF Flow Matching ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

HYDRA_FULL_ERROR=1 python train.py \
    experiment=denoiser_flow_matching \
    experiment_name=smoke_test_$(date +%Y%m%d_%H%M%S) \
    project_name="GARF-smoke" \
    data.data_root=/data/gpfs/projects/punim2657/GARF/input/Fractura/bone_synthetic.hdf5 \
    data.categories="['pig']" \
    data.num_workers=4 \
    data.batch_size=8 \
    data.multi_ref=True \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.max_epochs=5 \
    trainer.check_val_every_n_epoch=5 \
    trainer.precision=bf16-mixed \
    model.feature_extractor_ckpt=output/feature_extractor.ckpt \
    loggers=csv \
    tags='["smoke_test"]'

echo ""
echo "=== SMOKE TEST COMPLETE ==="
