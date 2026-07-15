# Juglet deploy findings — 2026-05-28

Day's investigation across GARF / TORA / PF++ for archaeological deploy on the
9-piece Juglet scan, focused on understanding why GARF and TORA visualizations
don't look assembled while PF++ does, and what would change that.

## TL;DR

- TORA's "proposed_assembly" PNGs **are** the correct paper-style proposal; they
  look scattered on Juglet because the flow target (`pointclouds_gt`) is the
  scan/table layout, not a broken-object layout.
- TORA benefits from compressing mesh layout (local02 experiment) because its
  flow target depends on mesh positions.
- GARF does **not** benefit from the same compression, because GARF's
  dataloader strips global layout (per-part recenter + random rotate + unit
  scale) before the model sees anything.
- GARF on Juglet is a **scale/distribution mismatch in the prediction frame**,
  not a denoising failure. Trajectory length and rotation drift are normal;
  predicted translation magnitudes are simply out of the model's trained range.
- Manual pre-assembly of Juglet would defeat the purpose. Realistic next steps
  involve either calibration at inference, fine-tuning, or an architecture
  change that doesn't strip layout.

---

## 1. TORA: what the PNGs actually mean

Default eval PNGs:

| File | What it is |
|---|---|
| `*_input.png` | Conditioning clouds after anchor-free aug (per-part center + random rotate). |
| `*_scan_ref.png` | `pointclouds_gt` — globally centered scan layout (≠ true assembly). |
| `*_generation*.png` | Flow endpoint `x̂` toward `pointclouds_gt`. |
| `*_proposed_assembly*.png` | Scattered input `P_k` after Procrustes `T̂_k` (paper Eq. 2 proposal). |

On Juglet, `generation` and `proposed_assembly` both look like scattered
clusters along a diagonal — similar in spirit to `scan_ref`. That is because
the flow target is the spread scan layout, not a tight broken-object cloud.

Sample run: `TORA/eval_runs/juglet_deploy_proposed_25279003/visualizations/`

## 2. Why thinwalled / BB look assembled and Juglet doesn't (same model)

Same TORA code, same checkpoint, same anchor-free settings.

| Dataset | Mesh part-centroid max distance | Visual outcome |
|---|---:|---|
| thinwalled bowl (sample 129) | ~0.92 | One coherent bowl |
| Juglet deploy (raw) | ~4.67 | Scattered diagonal clusters |

The `input.png` always looks like a tight pile (per-part recenter). The
`generation` output looks however the **flow target** looks. BB / thinwalled
meshes are stored as a broken object → tight target → tight generation.
Juglet meshes are stored at scan/table spacing → spread target → spread
generation.

## 3. Local compression experiment (method 2)

Built a Juglet variant with per-part centroids compressed toward the global
center (alpha = 0.2). No manual assembly; preserves relative structure.

- Source: `/data/gpfs/projects/punim2657/Dataset/Juglet_anchor_centered`
- Output: `/data/gpfs/projects/punim2657/Dataset/Juglet_anchor_centered_local02`
- Max centroid spread: 4.6740 → 0.9348
- HDF5: `GARF/input/juglet_deploy_local02.hdf5`,
  `TORA/dataset/juglet_deploy_local02.hdf5`
- Configs/launchers:
  - `TORA/repo/config/data/zeroshot/juglet_deploy_local02.yaml`
  - `TORA/eval_juglet_deploy_local02.slurm` → job `25528198` (completed)
  - `GARF/infer_juglet_deploy_local02.slurm` → job `25528466` (completed)

### TORA result on local02 (job 25528198)

- `generation01.png` shows a much tighter bowl-like cluster.
- Object chamfer / translation error dropped sharply vs raw Juglet.
- Confirms: TORA's behavior is driven by `pointclouds_gt` layout.

### GARF result on local02 (job 25528466)

- Mean rotation error: ~113°, comparable to raw Juglet run.
- Visual change: minimal — predicted spread basically unchanged.
- This was a key piece of evidence used in the next step.

## 4. GARF matched-piece diagnostic (the decisive run)

Tool: `GARF/scripts/garf_matched_diagnostic.py`
Launcher: `GARF/diagnose_garf_matched_9pc.slurm` (job `25531535`)
Outputs:

- `GARF/logs/diagnostics/matched_runs/diag_20260528_132826/`
  - `selected_samples.json`
  - `diagnostic_summary.csv`
  - `diagnostic_summary.md`

Same seed, checkpoint, and inference settings across 9-piece samples from:

- juglet_raw (anchor-centered Metashape scan)
- juglet_local02 (compressed variant)
- bb_everyday (Breaking Bad val sample with 9 parts)
- fractura_pig (bone_synthetic pig sample with 9 parts)

Indicators (from JSON trajectory + HDF5 geometry):

| case | mesh_maxd | target_maxd | pred_maxd | pred/target | t_start | t_final | rot_drift_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| juglet_raw | 4.6740 | 4.6972 | 0.6864 | 0.146 | 0.0950 | 0.2988 | 50.94 |
| juglet_local02 | 0.9348 | 0.9569 | 0.6864 | 0.717 | 0.0950 | 0.2988 | 50.94 |
| bb_everyday | 0.5529 | 0.5208 | 0.5123 | 0.984 | 0.1491 | 0.2609 | 45.49 |
| fractura_pig | 1.0791 | 1.0571 | 1.0326 | 0.977 | 0.1477 | 0.3273 | 84.27 |

## 5. Smoking gun: juglet_raw and juglet_local02 give IDENTICAL model output

All of `traj_t_mean_start`, `traj_t_mean_final`, `pred_t_maxdist`,
`traj_rot_change_mean`, `traj_rot_change_max` are bit-for-bit equal between
raw and local02.

Reason: GARF's dataloader (`assembly/data/breaking_bad/weighted.py::transform`
and `assembly/data/inference/mesh.py`) does per-part:

1. recenter (subtract centroid)
2. random rotation
3. scale to unit cube

After these three steps the model receives no information about where pieces
sat in the tray, so changing the global mesh layout from 4.7m to 0.93m has
zero effect on the model's input or output.

This invalidates "mesh compression to help GARF" as a strategy.

## 6. Why Juglet still looks bad while BB / pig look fine

The model output magnitude is similar across all 4 cases (`pred_t_maxdist`
0.5–1.0). The difference is the **target frame** the prediction must compare
against:

- BB / pig training distribution: mesh-frame translations span ~0.5–1.0 → in
  range, ratio ≈ 1.
- Juglet raw: target = 4.7 (table spread) while model output capped near 0.7
  → spread ratio 0.146, 6–7× short.

GARF's denoiser is engaging on Juglet (rotation drift 50.9° vs BB 45.5°). It's
not stuck. It just predicts within its trained scale, which is far smaller than
a scan-tray scale.

## 7. Why local02 helped TORA but not GARF

- TORA: flow target = global `pointclouds_gt` cloud → directly shaped by
  mesh layout. Compressing meshes shrinks the target. Model output follows.
- GARF: per-part SE(3) prediction in a normalized frame; layout is stripped
  before the model sees anything. Compressing meshes only changes the stored
  GT translations, not the model's behavior.

## 8. Implications for the Juglet project

- For tray-level visual QA today, **PF++** remains the right pipeline. It
  reapplies poses through `init_pose` / mesh chain in Blender, producing
  sherd-on-tray visuals.
- **GARF deploy GLB / viz** has been corrected to use `T_pred @ T_aug⁻¹` in
  mesh space (matches benchmark formula) and to skip the post-`anchor_free` GT
  snap during deploy. But these fixes can't compensate for the
  scale-distribution mismatch above.
- **TORA proposed_assembly PNGs** are now generated automatically alongside
  generation PNGs (`save_procrustes_assembly=true`).

## 9. What could actually help GARF on Juglet (no manual assembly)

Ordered by effort:

1. **Post-hoc translation scale calibration at inference.** Multiply predicted
   per-part translations by an archaeological scale factor before applying
   them to meshes. Cheap, no retraining. Spreads pieces correctly but does
   not guarantee good relative fit.
2. **Fine-tune GARF** on broken-object datasets re-positioned at
   archaeological scales so the output translation distribution widens.
3. **Architectural change** — replace per-part recenter + scale with a
   global-aware encoder so the model can actually condition on layout. Major
   work; would let mesh compression and similar layout-side tweaks matter.

## 10. Artifacts produced today

- Compressed-local Juglet dataset: `Dataset/Juglet_anchor_centered_local02/`
- HDF5 variants: `juglet_deploy_local02.hdf5` (GARF and TORA copies)
- TORA config + Slurm: `juglet_deploy_local02.yaml`, `eval_juglet_deploy_local02.slurm`
- TORA Procrustes assembly export wiring:
  - `tora/procrustes.apply_rigid_transformations`
  - `FlowVisualizationCallback.save_procrustes_assembly`
  - `scripts/export_procrustes_assembly_viz.py`,
    `scripts/export_procrustes_assembly_viz.sh`
- GARF deploy fixes:
  - `denoiser_base.py` deploy GLB uses `T_pred @ T_aug⁻¹`
  - Deploy JSON now saves `gt_transform` for mesh-relative viz
  - `generate_viz.py` deploy / `--proposed_only` uses mesh-relative formula and
    recomputes scatter aug from `--aug_seed`
- GARF Slurm launchers:
  - `infer_juglet_deploy.slurm` (raw)
  - `infer_juglet_deploy_local02.slurm` (compressed)
  - `diagnose_garf_matched_9pc.slurm` (matched diagnostic)
- Diagnostic tooling: `scripts/garf_matched_diagnostic.py`
- Submitted jobs:
  - TORA local02: `25528198` (completed)
  - GARF local02 h100: `25528466` (completed)
  - GARF local02 a100 duplicate: `25529713` (cancelled after primary completed)
  - GARF matched diagnostic v1: `25531298` (failed — missing 9-piece in
    fractura/ceramics)
  - GARF matched diagnostic v2: `25531535` (completed)

## 11. Open / suggested next steps

- Add a per-step trajectory dump and small plot to the diagnostic so we can
  see if Juglet converges, oscillates, or collapses to the same fixed cluster
  every time.
- Optional 3-seed repeat of the matched diagnostic for mean/std per case.
- Decide whether to invest in fine-tuning GARF for archaeological deploy or to
  treat PF++ as the deploy renderer and document the GARF/TORA limitations.
