#!/bin/bash
#SBATCH --job-name=garf_rwscore
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rwlora/rwscore_%j.out

# Score GARF rough/worn arms on CPU with TORA's own_place (pot-size ruler), for runs whose
# in-job scoring did not complete. STAMPS = space-separated job IDs of rwlora_arm runs.
#   sbatch --export=ALL,STAMPS="31449317 31449318" slurm/rwlora_score.slurm

set -euo pipefail
STAMPS="${STAMPS:?set STAMPS}"
cd /data/gpfs/projects/punim2657/GARF
source /data/gpfs/projects/punim2657/CRAG/.venv/bin/activate
TORA=/data/gpfs/projects/punim2657/TORA/repo
for J in $STAMPS; do
    for RUN in logs/rwlora/rwlora_eval_*_${J}/version_0; do
        echo "--- $RUN"
        POTMM=(); case "$RUN" in *juglet_gt*) POTMM=(--pot-mm 65);; esac
        python $TORA/scripts/own_place.py --run "$RUN" "${POTMM[@]}" --json "$RUN/own_place.json" \
            || echo "own_place failed on $RUN"
    done
done
echo "=== done $(date) ==="
