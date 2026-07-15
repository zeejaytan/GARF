#!/bin/bash
#SBATCH --job-name=garf-exp9-pseudogt
#SBATCH --partition=sapphire
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=exp9_pseudo_gt_bound_%j.out

# Exp 9 — PF++ PSEUDO-GT LABEL-ERROR BOUND (Juglet pairwise oracle).
#
# The headline Juglet failure number (true-mate chamfer/diag 0.070 vs control
# 0.024, Exp 6) scores GARF against the PF++ pseudo-assembly. If PF++'s
# relative poses are loose, part of that 0.070 is label error, not GARF error.
# This probe refines each true-mate reference pose with band-constrained ICP,
# measures the label drift, and re-scores the EXISTING GARF predictions
# (Exp 6 / 6b GLBs — no GPU needed) against the refined reference. The same
# procedure runs on the control ceramics (real GT) as its own validation:
# there the refinement must barely move and re-scores must match.
#
#   If the re-scored Juglet true-mate median stays > 0.050 -> labels are fine,
#   the perceptual-failure conclusion stands at full strength.
#   If it collapses toward the control's 0.024 -> the oracle overstated GARF's
#   failure and the pseudo-GT must be replaced before further remedy work.

set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="logs/diagnostics/exp9_pseudogt_${STAMP}"

PFPP_DIR=/data/gpfs/projects/punim2657/Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0
MESH_DIR=/data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000

echo "=== Exp 9: PF++ pseudo-GT label-error bound ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

python scripts/juglet_pseudo_gt_bound.py \
    --control-hdf5 input/control_ceramics_pairs.hdf5 \
    --control-adjacency logs/diagnostics/control_ceramics_adjacency.json \
    --control-run-dirs \
        logs/deploy/exp6b_ctrl_20260709_144515_s41 \
        logs/deploy/exp6b_ctrl_20260709_144515_s42 \
        logs/deploy/exp6b_ctrl_20260709_144515_s43 \
    --control-orig logs/diagnostics/pair_chamfer_control/pairs.json \
    --juglet-adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
    --juglet-run-dirs \
        logs/deploy/exp6_pairs_20260610_162659_s41 \
        logs/deploy/exp6_pairs_20260610_162659_s42 \
        logs/deploy/exp6_pairs_20260610_162659_s43 \
    --juglet-orig logs/diagnostics/pair_chamfer_juglet/pairs.json \
    --pfpp-dir "$PFPP_DIR" \
    --mesh-dir "$MESH_DIR" \
    --out "$OUT"

echo "Summary: ${OUT}/summary.md"
echo "=== DONE ==="
