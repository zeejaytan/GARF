#!/usr/bin/env bash
# Orchestrate archaeological *deploy* data prep (CPU): Step 1 + HDF5 + TORA copy + PF++ .npz.
# See ARCHAEOLOGICAL_DEPLOYMENT.md for requirements (anchor-center before inference).
set -euo pipefail

ROOT="/data/gpfs/projects/punim2657"
GARF="${ROOT}/GARF"
RAW="${ROOT}/Dataset/Juglet"
CENTERED="${ROOT}/Dataset/Juglet_anchor_centered"
HDF5_GARF="${GARF}/input/juglet_deploy.hdf5"
HDF5_TORA="${ROOT}/TORA/dataset/juglet_deploy.hdf5"
PF_VAL="${ROOT}/Puzzlefusion/data/pc_data/juglet_deploy/val"

prepare() {
  cd "${GARF}"
  # shellcheck source=/dev/null
  source .venv/bin/activate

  echo "=== Step 1: anchor-center meshes (mandatory for deploy) ==="
  python preprocess_scan_to_anchor_frame.py \
    --input-dir "${RAW}" \
    --output-dir "${CENTERED}"

  echo "=== Step 2: HDF5 (GARF + TORA split keys) ==="
  python create_juglet_hdf5.py \
    --input-dir "${CENTERED}" \
    --output "${HDF5_GARF}" \
    --sample-name Juglet-000 \
    --category artifact \
    --split-keys artifact,juglet_deploy

  mkdir -p "$(dirname "${HDF5_TORA}")"
  cp -f "${HDF5_GARF}" "${HDF5_TORA}"
  echo "Copied -> ${HDF5_TORA}"

  echo "=== Step 2b: PuzzleFusion++ .npz ==="
  python "${ROOT}/Puzzlefusion/convert_hdf5_to_npz.py" \
    --hdf5 "${HDF5_GARF}" \
    --category artifact \
    --output-dir "${PF_VAL}" \
    --min-parts 2 \
    --max-parts 20 \
    --split val

  echo "=== prepare done ==="
  echo "  GARF/TORA HDF5: ${HDF5_GARF}"
  echo "  PF++ val dir:  ${PF_VAL}"
}

usage() {
  echo "Usage: $0 prepare"
  echo "  prepare  Run Step 1 (anchor-center) + build juglet_deploy artifacts."
}

case "${1:-}" in
  prepare) prepare ;;
  -h|--help|help) usage ;;
  *) usage; exit 1 ;;
esac
