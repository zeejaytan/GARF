# Archaeological deployment inference (no assembly ground truth)

Use this workflow when you have **scanned fragments only** and want the model to propose an assembly — without knowing whether the pieces truly belong together.

This is **not** the benchmark `eval_complex` path (which compares predictions to dataloader “GT” poses and reports part_acc / RMSE).

## Requirement (read first)

| Requirement | Why |
|-------------|-----|
| **No true assembly GT** | Metrics and “Ground truth” viz panels are invalid. Judge **predicted** outputs only. |
| **Anchor-centered meshes (step 1)** | Raw Metashape/table coordinates put fragments metres apart. Models were trained on fragments in a **local** layout. Centering on the largest sherd fixes the dominant scale/offset bug. |
| **`anchor_free` inference** | Do not pin the reference sherd to dataloader scatter poses (eval-only trick). |
| **`deploy_mode` (GARF)** | Exports `predicted_assembly.glb` from predicted poses only, not `T_pred @ T_gt⁻¹`. |
| **Human / geometric QA** | Models do not output “these sherds match.” Use GLB/PNG review, fracture rim contact, expert judgement. |

## Pipeline overview

```text
Dataset/<name>/*.obj                    # raw scan export
    │
    ▼  Step 1 — preprocess_scan_to_anchor_frame.py
Dataset/<name>_anchor_centered/*.obj
    │
    ▼  Step 2 — create_juglet_hdf5.py (or equivalent)
GARF/input/<name>_deploy.hdf5
TORA/dataset/<name>_deploy.hdf5
    │
    ▼  Step 3 — inference (GPU)
    ├─ GARF  infer_*_deploy.slurm   → assembly_results/.../predicted_assembly.glb
    ├─ TORA  eval_*_deploy.slurm     → visualizations/*_generation*.png
    └─ PF++  eval_*_deploy.slurm     → inference/ + Blender render
```

One-command orchestration (Step 1 + Step 2 data files):

```bash
cd /data/gpfs/projects/punim2657/GARF
bash scripts/run_juglet_deploy.sh prepare   # CPU: center + HDF5 + TORA copy + PF++ npz
sbatch slurm/infer_juglet_deploy.slurm            # GPU: GARF deploy inference
sbatch /data/gpfs/projects/punim2657/TORA/eval_juglet_deploy.slurm
sbatch /data/gpfs/projects/punim2657/Puzzlefusion/scripts/eval_juglet_deploy.slurm
# optional: PF++ Blender stills/video
sbatch /data/gpfs/projects/punim2657/Puzzlefusion/scripts/render_juglet_deploy.slurm
```

## Step 1 — Anchor-center meshes (mandatory)

**Requirement:** Every deploy run must start from anchor-centered OBJs. Raw exports
often have pairwise centroid distances of **metres**; models expect a **local**
broken-object scale. This step only removes the global table/scanner offset (largest
sherd centroid); it does **not** assert that fragments belong to one vessel.

```bash
cd /data/gpfs/projects/punim2657/GARF
source .venv/bin/activate

python preprocess_scan_to_anchor_frame.py \
  --input-dir /data/gpfs/projects/punim2657/Dataset/Juglet \
  --output-dir /data/gpfs/projects/punim2657/Dataset/Juglet_anchor_centered
```

Outputs `deploy_manifest.json` (anchor piece, centroids, pairwise distances).

## Step 2 — Build HDF5 / NPZ

```bash
# GARF (+ copy for TORA)
python create_juglet_hdf5.py \
  --input-dir /data/gpfs/projects/punim2657/Dataset/Juglet_anchor_centered \
  --output /data/gpfs/projects/punim2657/GARF/input/juglet_deploy.hdf5 \
  --sample-name Juglet-000 --category artifact \
  --split-keys artifact,juglet_deploy

cp /data/gpfs/projects/punim2657/GARF/input/juglet_deploy.hdf5 \
   /data/gpfs/projects/punim2657/TORA/dataset/juglet_deploy.hdf5
```

`--split-keys artifact,juglet_deploy` writes the same sample under both
`data_split/artifact` (GARF `data.categories`) and `data_split/juglet_deploy`
(TORA resolves the split group from the HDF5 filename stem `juglet_deploy`).

```bash
# PuzzleFusion++ point clouds (from GARF venv)
python /data/gpfs/projects/punim2657/Puzzlefusion/convert_hdf5_to_npz.py \
  --hdf5 /data/gpfs/projects/punim2657/GARF/input/juglet_deploy.hdf5 \
  --category artifact \
  --output-dir /data/gpfs/projects/punim2657/Puzzlefusion/data/pc_data/juglet_deploy/val \
  --min-parts 2 --max-parts 20 --split val
```

(`convert_hdf5_to_npz.py` supports `--hdf5` / `--output-dir` for single-dataset conversion.)

## Step 3 — GARF deploy inference

```bash
sbatch slurm/infer_juglet_deploy.slurm
```

Key Hydra overrides:

- `data.data_root=input/juglet_deploy.hdf5`
- `++model.inference_config.anchor_free=true`
- `++model.inference_config.deploy_mode=true`
- `++model.inference_config.save_assembly=true`

**Deliverables**

| Path | Meaning |
|------|---------|
| `logs/deploy/juglet_deploy_*/assembly_results/.../predicted_assembly.glb` | **Primary output** — proposed assembly |
| `.../scan_layout.glb` | Anchor-centered scan positions (reference only) |
| `.../json_results/*.json` | Predicted poses (`deploy_mode: true`; ignore part_acc) |

Regenerate HTML viz (pred-only framing):

```bash
python generate_viz.py \
  --results_dir logs/deploy/<run>/version_0/json_results \
  --hdf5 input/juglet_deploy.hdf5 \
  --all --output_dir viz_output/juglet_deploy \
  --scan_layout
```

Use the **Predicted (model pose frame)** view; ignore benchmark metrics.

## Step 3 — TORA deploy inference

```bash
sbatch /data/gpfs/projects/punim2657/TORA/eval_juglet_deploy.slurm
```

Config: `data=zeroshot/juglet_deploy`, `anchor_free=true`, `center_points=true`.

Judge `*_input.png` and `*_generation*.png` only (no `*_gt`).

## Step 3 — PuzzleFusion++ deploy

```bash
sbatch /data/gpfs/projects/punim2657/Puzzlefusion/scripts/eval_juglet_deploy.slurm
sbatch /data/gpfs/projects/punim2657/Puzzlefusion/scripts/render_juglet_deploy.slurm
```

## New scan / tray batch

1. Put OBJs in `Dataset/<MyScan>/`.
2. Run step 1 with `--input-dir` / `--output-dir`.
3. Run step 2 with new `--output` / `--sample-name`.
4. Point deploy SLURM jobs at the new HDF5 paths.

## What not to do

- Do not interpret **part_acc**, **rmse_r**, **rmse_t**, or **shape_cd** on deploy data.
- Do not treat **gt.png**, **scan_ref.png**, or **Ground truth** HTML modes as true assembly.
- Do not skip step 1 on table-separated Metashape exports.

## Related docs

- `JUGLET_DEPLOY_INFERENCE_ANALYSIS.md` — why PF++ looked good vs GARF/TORA on deploy (2026-05-20 run)
- `GARF_ARCHITECTURE.md` — anchor / anchor_free behaviour
- `GARF_vs_PuzzleFusion_comparison.md` — benchmark eval (with GT), not deploy
