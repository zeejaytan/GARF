#!/usr/bin/env bash
# Pull latest code into the Spartan checkout GARF/repo (ff-only), then sbatch a job from there.
# Job scripts cd to the working area GARF/ themselves and write Slurm output to GARF/logs/slurm/.
# Usage: ./scripts/remote/pull_and_sbatch.sh <job.slurm> [extra sbatch args...]
set -euo pipefail

SPARTAN_HOST="${SPARTAN_HOST:-spartan}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/gpfs/projects/punim2657/GARF/repo}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job.slurm> [extra sbatch args...]" >&2
  exit 1
fi

JOB_SCRIPT="$1"
shift

ssh "$SPARTAN_HOST" "cd $(printf '%q' "$REMOTE_ROOT") && git pull --ff-only && sbatch $(printf '%q' "$JOB_SCRIPT") $(printf '%q ' "$@")"
