#!/bin/bash
#SBATCH --job-name=pfpp-t0-panel-cpu
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=t0_panel_cpu_%j.out

# T0 panel rescore (CPU only) — reuses an existing GARF deploy GLB.
# Args: $1 = GARF GLB path, $2 = out dir tag suffix (default rerun),
#       $3 = PF++ inference sample dir (default juglet_deploy/0),
#       $4 = PF++ pc npz (default juglet_deploy val 00000.npz)

set -euo pipefail
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

PF=/data/gpfs/projects/punim2657/Puzzlefusion
GLB=${1:?need GARF GLB path}
TAG=${2:-rerun}
PFDIR=${3:-$PF/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0}
PCNPZ=${4:-$PF/data/pc_data/juglet_deploy/val/00000.npz}
TS=$(date +%Y%m%d_%H%M%S)

python scripts/pfpp_layout_probes.py \
    --pfpp-dir "$PFDIR" \
    --pc-npz   "$PCNPZ" \
    --mesh-hdf5 /data/gpfs/projects/punim2657/GARF/input/juglet_deploy.hdf5 --mesh-key artifact/Juglet-000 \
    --garf-glb "$GLB" \
    --out      "logs/diagnostics/pfpp_t0_${TAG}_${TS}" \
    --tag t0 \
    --n-null 5

echo "=== DONE: logs/diagnostics/pfpp_t0_${TAG}_${TS} ==="
