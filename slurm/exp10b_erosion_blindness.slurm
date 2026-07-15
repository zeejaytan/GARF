#!/bin/bash
#SBATCH --job-name=garf_exp10b_erode
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp10b_erode_%j.out

# Exp 10b — causal bridge for Exp 10. Erode the TRUE fracture bands of labeled
# synthetic breaks toward Juglet-like wear (labels preserved) and measure the
# frozen encoder's fracture response. If fired% / AUC-vs-GT collapse toward
# Juglet's silent level as erosion grows, wear is the CAUSE of the encoder's
# blindness -> training on eroded breaks is the justified remedy.

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate
export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export HYDRA_FULL_ERROR=1

STAMP=$(date +%Y%m%d_%H%M%S)
echo "=== Exp 10b erosion-blindness sweep ==="; date

for S in 0.0 0.5 1.0; do
  OUT=logs/diagnostics/exp10b_erode_${STAMP}/s${S}
  echo "---- synth erode strength ${S} -> ${OUT} ----"
  python scripts/fracseg_introspection.py \
      --ckpt output/feature_extractor.ckpt \
      --synth-source input/Fractura/bone_synthetic.hdf5 \
      --synth-objects pig/synthetic_fracture/20/mode_5 \
                      pig/synthetic_fracture/20/mode_4 \
                      pig/synthetic_fracture/20/mode_2 \
      --control-source input/Fractura/fractura_real.hdf5 \
      --control-objects ceramics/pink_bowl ceramics/blue_pot \
      --juglet-mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
      --out ${OUT} \
      --n-per-piece 5000 \
      --synth-erode-strength ${S}
done

echo "=== erosion-blindness summary ==="
python - <<'PY'
import json, glob, os
root = sorted(glob.glob("logs/diagnostics/exp10b_erode_*"), key=os.path.getmtime)[-1]
print(f"root: {root}")
print(f"{'erode':>6} | {'synth fired%':>12} | {'synth AUC_gt':>12} | {'juglet fired%':>13}")
for d in sorted(glob.glob(root+"/s*")):
    s = json.load(open(d+"/summary.json"))
    print(f"{os.path.basename(d)[1:]:>6} | {s['synth_mean_fired_frac']*100:12.2f} | "
          f"{s['synth_mean_auc_stored']:12.3f} | {s['juglet_fired_frac']*100:13.2f}")
PY
