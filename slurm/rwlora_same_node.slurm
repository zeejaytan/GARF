#!/bin/bash
#SBATCH --job-name=garf_t05
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/rwlora/rwlora_%x_%j.out

# .scratch/rough-worn-dose ticket 05: every arm on ONE GPU node, so node-to-node
# differences (up to 0.15 sherd per attempt, ticket 03) cannot decide the comparison.
# Runs slurm/rwlora_arm.slurm once per arm inside this allocation (redraw only, no training).
#   1-3  seed-7 adapters: clean-break, rough 1x, worn 4x, presentations 1-10, Juglet + Fractura
#   4-5  seed-42 clean-break and worn 4x, presentations 7 9 10, Juglet only (worn's two
#        narrowest ticket-02 wins, plus its one loss)

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
echo "=== same-node run on $(hostname) $(date) ==="
run() { env "$@" bash slurm/rwlora_arm.slurm || echo "ARM FAILED: $*"; }
P10="1 2 3 4 5 6 7 8 9 10"
S2="juglet_gt fractura_fresh"
run ARM=fresh_s7 ADAPTER=output/rwlora_fresh_s7_31482030/last.ckpt     DRAW_SEEDS="$P10" SETS="$S2"
run ARM=noise_d100_s7 ADAPTER=output/rwlora_noise_d100_s7_31526350/last.ckpt DRAW_SEEDS="$P10" SETS="$S2"
run ARM=worn_d400_s7 ADAPTER=output/rwlora_worn_d400_s7_31526349/last.ckpt  DRAW_SEEDS="$P10" SETS="$S2"
run ARM=fresh ADAPTER=output/rwlora_fresh_31449318/last.ckpt                DRAW_SEEDS="7 9 10" SETS=juglet_gt
run ARM=worn_d400 ADAPTER=output/rwlora_worn_d400_s42_31482373/last.ckpt    DRAW_SEEDS="7 9 10" SETS=juglet_gt
echo "=== all arms done on $(hostname) $(date) ==="
