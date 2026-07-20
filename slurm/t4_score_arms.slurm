#!/bin/bash
#SBATCH --job-name=pfpp-t4-score
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=t4_score_arms_%j.out

# Score the T4 prior-dominance ablation arms with the T0 layout panel.
set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

PF=/data/gpfs/projects/punim2657/Puzzlefusion
INF=$PF/output/denoiser/everyday_epoch2000_bs64/inference
NPZ=$PF/data/pc_data/juglet_deploy/val/00000.npz
TS=$(date +%Y%m%d_%H%M%S)
OUT=logs/diagnostics/pfpp_t4_${TS}

for MODE in none shuffle random zero; do
    echo "=== scoring t4_${MODE} ==="
    python scripts/pfpp_layout_probes.py \
        --pfpp-dir "$INF/t4_${MODE}/0" \
        --pc-npz   "$NPZ" \
        --mesh-hdf5 /data/gpfs/projects/punim2657/GARF/input/juglet_deploy.hdf5 --mesh-key artifact/Juglet-000 \
        --out      "$OUT" \
        --tag      "t4_${MODE}" \
        --n-null 0
done

echo "=== combined ==="
cat $OUT/t4_*_summary.md
echo "=== DONE: $OUT ==="
