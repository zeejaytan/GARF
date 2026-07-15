#!/bin/bash
#SBATCH --job-name=garf_exp10_fracseg
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp10_fracseg_%j.out

# Exp 10 — look INSIDE GARF's frozen fracture-aware encoder.
# Does the pretrained backbone (which feeds the denoiser) recognize Juglet's worn
# archaeological rims as fracture/mating surfaces? Probe validated on labeled
# synthetic fractures; fresh real ceramics vs worn Juglet compared via the
# assembly-free relief-band detector.

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=logs/diagnostics/exp10_fracseg_${STAMP}

echo "=== Exp 10 FracSeg introspection ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python scripts/fracseg_introspection.py \
    --ckpt output/feature_extractor.ckpt \
    --synth-source input/Fractura/bone_synthetic.hdf5 \
    --synth-objects pig/synthetic_fracture/20/mode_5 \
                    pig/synthetic_fracture/20/mode_4 \
                    pig/synthetic_fracture/20/mode_2 \
    --control-source input/Fractura/fractura_real.hdf5 \
    --control-objects ceramics/pink_bowl ceramics/narrow_bottle2 \
                      ceramics/narrow_bottle4 ceramics/blue_pot \
    --juglet-mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
    --out ${OUT} \
    --n-per-piece 5000

echo "=== done -> ${OUT}/summary.md ==="
cat ${OUT}/summary.md
