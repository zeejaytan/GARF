#!/bin/bash
#SBATCH --job-name=garf_complex_eval
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=eval_complex_%j.out

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

CHECKPOINT="output/GARF.ckpt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== GARF Complex Fragment Evaluation ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 1: bone_synthetic / pig — 5-20 fragments
# ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME="bone_syn_pig_${TIMESTAMP}"
HDF5_PATH="input/Fractura/bone_synthetic.hdf5"

echo "══ Test 1: bone_synthetic / pig (5-20 frags) ══"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['pig']" \
    data.min_parts=5 \
    data.max_parts=20 \
    data.batch_size=4 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

echo "Results: logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 2: bone_synthetic / rib — 5-50 fragments (up to 47 in dataset)
# ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME="bone_syn_rib_${TIMESTAMP}"

echo "══ Test 2: bone_synthetic / rib (5-50 frags) ══"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['rib']" \
    data.min_parts=5 \
    data.max_parts=50 \
    data.batch_size=1 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

echo "Results: logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 3: fractura_real / ceramics — 3-12 fragments (real-world)
# ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME="fractura_real_ceramics_${TIMESTAMP}"
HDF5_PATH="input/Fractura/fractura_real.hdf5"

echo "══ Test 3: fractura_real / ceramics (3-12 frags) ══"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['ceramics']" \
    data.min_parts=2 \
    data.max_parts=20 \
    data.batch_size=4 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

echo "Results: logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 4: fractura_real / egg — 3-5 fragments
# ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME="fractura_real_egg_${TIMESTAMP}"

echo "══ Test 4: fractura_real / egg (3-5 frags) ══"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['egg']" \
    data.min_parts=2 \
    data.max_parts=10 \
    data.batch_size=4 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

echo "Results: logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 5: fractura_real / bones — 2-3 fragments
# ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME="fractura_real_bones_${TIMESTAMP}"

echo "══ Test 5: fractura_real / bones (2-3 frags) ══"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['bones']" \
    data.min_parts=2 \
    data.max_parts=10 \
    data.batch_size=4 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

echo "Results: logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 6: tray_archaeological / artifact — 40 fragments
# ─────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME="tray_archaeological_${TIMESTAMP}"
HDF5_PATH="input/tray_archaeological.hdf5"

echo "══ Test 6: tray_archaeological / artifact (40 frags) ══"
HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EXPERIMENT_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/complex_eval \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.precision=bf16-mixed \
    data.data_root=$HDF5_PATH \
    data.categories="['artifact']" \
    data.min_parts=2 \
    data.max_parts=50 \
    data.batch_size=1 \
    data.num_workers=4 \
    data.multi_ref=False \
    ckpt_path=$CHECKPOINT \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true \
    ++model.inference_config.write_to_json=true

echo "Results: logs/complex_eval/$EXPERIMENT_NAME/version_0/json_results"
echo ""

echo "=== ALL 6 EVALUATIONS COMPLETE ==="
echo "Date: $(date)"
