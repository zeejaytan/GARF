#!/bin/bash
#SBATCH --job-name=pfpp-t13-score
#SBATCH --partition=sapphire
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=t13_score_%j.out

# Score T1a / T1b / T3 PF++ runs (submit with --dependency=afterok:<gpu job>).
set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

PF=/data/gpfs/projects/punim2657/Puzzlefusion
INF=$PF/output/denoiser/everyday_epoch2000_bs64/inference
TS=$(date +%Y%m%d_%H%M%S)

echo "### T1a arms — layout panel (points-only: geometry differs from OBJs)"
OUT1A=logs/diagnostics/pfpp_t1a_${TS}
for ARM in foreign dup mirror; do
    python scripts/pfpp_layout_probes.py \
        --pfpp-dir "$INF/t1a_${ARM}/0" \
        --pc-npz   "$PF/data/pc_data/t1a_${ARM}/val/00000.npz" \
        --out "$OUT1A" --tag "t1a_${ARM}" --n-null 0
done
cat $OUT1A/t1a_*_summary.md

echo "### T1b — cross-seed stability"
python scripts/pfpp_stability.py \
    --run-dirs $INF/t1b_seed41/0 $INF/t1b_seed42/0 $INF/t1b_seed43/0 \
               $INF/t1b_seed44/0 $INF/t1b_seed45/0 \
    --pc-npz "$PF/data/pc_data/juglet_deploy/val/00000.npz" \
    --out logs/diagnostics/pfpp_t1b_${TS}

echo "### T3 — pairwise oracle (juglet)"
python scripts/pfpp_pair_chamfer.py juglet \
    --pairs-data "$PF/data/pc_data/t3_juglet_pairs" \
    --inference-dirs $INF/t3_juglet_s41 $INF/t3_juglet_s42 $INF/t3_juglet_s43 \
    --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
    --ref-pfpp-dir "$INF/juglet_deploy/0" \
    --ref-npz "$PF/data/pc_data/juglet_deploy/val/00000.npz" \
    --out logs/diagnostics/pfpp_t3_juglet_${TS}

echo "### T3 — pairwise oracle (control)"
python scripts/pfpp_pair_chamfer.py control \
    --pairs-data "$PF/data/pc_data/t3_control_pairs" \
    --inference-dirs $INF/t3_control_s41 $INF/t3_control_s42 $INF/t3_control_s43 \
    --out logs/diagnostics/pfpp_t3_control_${TS}

echo "=== DONE (tags ${TS}) ==="
