#!/bin/bash
#SBATCH --job-name=garf_render
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/data/gpfs/projects/punim2657/GARF/logs/rwlora/render_%j.out

# Draw a GARF attempt against the truth and the input it was given (scripts/render_garf_attempt.py),
# for the eval runs whose clouds/ were saved by rwlora_arm.slurm (pr-workspace wear-beyond-juglet 05).
#   sbatch --export=ALL,RUNS="<run dir> ...",COPIES=<copies.csv>,CARDS="p149_d00 p148_d00" slurm/render_garf_attempt.slurm
# Pictures and the per-sherd table go to <run dir>/render/; CARDS=all prints numbers for every pot.

set -euo pipefail
RUNS="${RUNS:?set RUNS}"
COPIES="${COPIES:?set COPIES}"
CARDS="${CARDS:-all}"
ATTEMPT="${ATTEMPT:-0}"
GARF_WORK=/data/gpfs/projects/punim2657/GARF
GARF_REPO=$GARF_WORK/repo
cd "$GARF_WORK"
source /data/gpfs/projects/punim2657/CRAG/.venv/bin/activate
echo "=== render GARF $(git -C "$GARF_REPO" rev-parse --short HEAD) cards=${CARDS} attempt=${ATTEMPT} $(date) ==="
for RUN in $RUNS; do
    echo "--- $RUN"
    python $GARF_REPO/scripts/render_garf_attempt.py "$RUN" --copies "$COPIES" --cards $CARDS \
        --attempt "$ATTEMPT" --out "$RUN/render" | tee "$RUN/render_${CARDS// /_}_a${ATTEMPT}.txt"
done
echo "=== done $(date) ==="
