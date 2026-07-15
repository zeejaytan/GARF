#!/bin/bash
#SBATCH --job-name=garf_exp11_wornenc
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=exp11_wornenc_%j.out

# Exp 11 — remedy for the Exp 10 encoder blindness. Fine-tune GARF's FracSeg
# feature extractor on labeled synthetic fractures WITH worn-break augmentation
# so it recognizes worn/smoothed breaks as fracture surfaces. Then validate on
# Juglet with the introspection probe: success = Juglet fired% rises toward the
# fresh-ceramic control's ~3.4% (baseline was 0.57%).

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=output/frac_seg_worn_${STAMP}

echo "=== Exp 11 FracSeg worn-break fine-tune ==="; date
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/finetune_frac_seg.py \
    --init-ckpt output/feature_extractor.ckpt \
    --data-root input/Fractura/bone_synthetic.hdf5 \
    --categories pig hip leg rib vertebra \
    --out-dir ${OUT} \
    --epochs 60 \
    --batch-size 16 \
    --lr 2e-5 \
    --erode-prob 0.6 \
    --erode-min 0.3 \
    --erode-max 1.0 \
    --num-workers 12 \
    --save-every 15

NEW_CKPT=${OUT}/feature_extractor_worn.ckpt
echo "=== validating fine-tuned encoder on Juglet (Exp 10 probe) ==="
python scripts/fracseg_introspection.py \
    --ckpt ${NEW_CKPT} \
    --synth-source input/Fractura/bone_synthetic.hdf5 \
    --synth-objects pig/synthetic_fracture/20/mode_5 \
                    pig/synthetic_fracture/20/mode_4 \
                    pig/synthetic_fracture/20/mode_2 \
    --control-source input/Fractura/fractura_real.hdf5 \
    --control-objects ceramics/pink_bowl ceramics/narrow_bottle2 \
                      ceramics/narrow_bottle4 ceramics/blue_pot \
    --juglet-mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
    --out ${OUT}/introspection_after \
    --n-per-piece 5000

echo "=== BEFORE vs AFTER (Juglet fracture response) ==="
python - <<PY
import json
after=json.load(open("${OUT}/introspection_after/summary.json"))
print("fine-tuned encoder:")
print(f"  synth fired% (probe) = {after['synth_mean_fired_frac']*100:.2f}  AUC_gt={after['synth_mean_auc_stored']:.3f}")
print(f"  control fired%       = {after['control_mean_fired_frac']*100:.2f}")
print(f"  JUGLET fired%        = {after['juglet_fired_frac']*100:.2f}  (baseline 0.57)")
print(f"  response ratio J/C   = {after['response_ratio_juglet_over_control']:.2f}")
PY

echo "=== done -> ${OUT} ==="
