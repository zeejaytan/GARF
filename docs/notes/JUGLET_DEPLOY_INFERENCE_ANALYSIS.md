# Juglet deploy inference analysis (GARF vs TORA vs PF++)

Analysis date: 2026-05-20  
Deploy run IDs: GARF `25192221`, TORA `25192222`, PF++ eval `25192223`, PF++ render `25192539`  
Data: anchor-centered Juglet (`Dataset/Juglet_anchor_centered`, 9 pieces, anchor = Piece03)

## Executive summary

Visual inspection showed **PuzzleFusion++ (PF++) produced a plausible assembly** on the Juglet deploy run, while **GARF and TORA did not** look assembled in their default visualizations.

That gap is largely explained by **inference and visualization pipelines**, not by PF++ alone “understanding” the scan better:

| Framework | Primary issue |
|-----------|----------------|
| **PF++** | Maps denoised poses back onto **real OBJ geometry** in mesh space via `init_pose` / `gt` / `predict` chaining in Blender. |
| **GARF** | Post-inference **`anchor_free` alignment to dataloader GT** warps poses before GLB export; GLB frame ≠ last denoising step in JSON. |
| **TORA** | Renders **point clouds in an augmented per-part frame**; metrics and PNGs do not show mesh-level assembly on the table. |

All three still use **synthetic per-part GT poses** at load time (recenter + random rotation). That is appropriate for Breaking Bad–style training data but is **not** true assembly ground truth for archaeological scans.

---

## Short answer

PF++ is the only pipeline that **maps denoised poses back onto real OBJ geometry in mesh space**. GARF and TORA mostly operate in a **synthetic per-part pose frame** (recenter + random rotation per fragment). On Juglet deploy that frame is **not** “where the sherds sit on the table,” so their outputs stay scattered or get warped by post-processing. PF++’s Blender path explicitly undoes that preprocessing and reapplies the prediction on anchor-centered meshes—which matches visual inspection.

---

## 1. What each framework optimizes at inference

| | **Input geometry** | **“GT” during inference** | **What you view** |
|---|-------------------|---------------------------|-------------------|
| **GARF** | Point clouds from HDF5 meshes: per-part `recenter_pc` + `rotate_pc` | Same synthetic poses (`translations` / `quaternions` from dataloader) | HTML / GLB from predicted SE(3) |
| **TORA** | Same idea: `anchor_free` centers every part + random rot on non-anchor | `pointclouds_gt` = scan layout in normalized frame | PNG point-cloud renders |
| **PF++** | 1000 pts/part from meshes; global ref + per-part scatter | `gt.npy` = scatter poses; saved with `init_pose` | Blender: **raw OBJs** + `compute_final_transformation` |

Deploy TORA metrics (vs fake GT): **~11% part accuracy, ~102° mean rotation error** — same order as the earlier benchmark Juglet run (`juglet_deploy_sample00000_generation00.json`), not a deploy-prep regression.

**Key paths**

- GARF: `logs/deploy/juglet_deploy_20260520_170639/`
- TORA: `TORA/eval_runs/juglet_deploy_25192222/visualizations/`
- PF++: `Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0/`
- PF++ render: `Puzzlefusion/renderer/juglet_deploy_output/0/`

---

## 2. GARF: two separate issues in the deploy path

### A. Post-denoising `anchor_free` alignment (main inference bug)

In `assembly/models/denoiser/denoiser_base.py`, when `anchor_free=true` the model denoises with **no part pinned**, but **after** the loop it realigns the whole solution to the **largest part’s dataloader GT pose**:

```python
if self.inference_config.get("anchor_free", False):
    ref_part = data_dict["ref_part"][part_valids]
    ref_part_gt_trans = gt_trans_and_rots[ref_part, :3]
    ...
    pred_trans = (
        transforms.quaternion_apply(rot_alignment, pred_trans) + trans_alignment
    )
    pred_rots = transforms.quaternion_multiply(rot_alignment, pred_rots)
```

That GT is **not** scan layout—it comes from `recenter_pc` / `rotate_pc` on each fragment (`assembly/data/inference/mesh.py`). For deploy you want poses in a free assembly frame; snapping to a **random** ref-part GT rotates/translates all nine pieces into an arbitrary frame.

**Measured on deploy run `juglet_deploy_20260520_170639`:**

| Source | Typical ‖t‖ per part |
|--------|---------------------|
| Last step in `json_results/0.json` (trajectory) | ~0.07–0.43 |
| `predicted_assembly.json` / GLB export | ~2.7–3.0 |

The denoiser’s last step is modest in pose space; **alignment inflates** translations before GLB export. HTML viz uses the JSON trajectory (pre-alignment); **`predicted_assembly.glb` does not match the last denoising step**.

**Recommended fix:** when `deploy_mode=true`, skip this `anchor_free` GT alignment block (or align only to identity / scan anchor, not dataloader GT).

### B. GLB export frame

Deploy GLB builds each piece as: mesh centroid removed, then predicted SE(3) applied (`predicted_assembly.glb`). Predictions live in the **normalized, per-part-augmented** frame, not in anchor-centered mesh coordinates. Even with a correct relative assembly, applying that directly on table-centered OBJs without the PF++ transform chain will look wrong.

---

## 3. TORA: paper intent vs default PNGs vs Juglet data

Deploy config is appropriate (`anchor_free: true`, `data=zeroshot/juglet_deploy`). The [TORA paper](https://arxiv.org/html/2604.04050v1) describes the **same task family as GARF/RPF**:

| Paper step | Repo implementation |
|------------|---------------------|
| Input `{P_k}` unposed (anchor-free: per-part center + random rot) | `batch["pointclouds"]` after `_proc_part` |
| Flow target `x_k(0)` = **assembled** point cloud | `x_0 = data_dict["pointclouds_gt"]` in `sample_rectified_flow` |
| Denoise noise → `x̂_k(0)` | `trajs[-1]` → `*_generation*.png` |
| Output poses: Procrustes `T̂_k` aligning `P_k` → `x̂_k` | `fit_transformations(pointclouds, pointclouds_pred, …)` in `test_step` |
| Metrics (anchor-free): ICP on anchor for scoring only | `evaluator.align_anchor` — **not** applied to default PNGs |

**Default eval PNGs are not the paper’s “assembly proposal.”** They show the **flow endpoint** in the same frame as `pointclouds_gt`. On Breaking Bad / Fractura, `pointclouds_gt` ≈ broken-object layout, so `*_generation*.png` often looks assembled. On Juglet, `pointclouds_gt` ≈ **scan/table layout** (metres apart even after anchor-centering), so generations can look like `*_scan_ref.png` even when Procrustes poses are meaningful.

**Procrustes proposal PNGs** (added in this repo): `*_proposed_assembly*.png` = scattered **input** parts after `T̂_k` (Eq. 2 in the paper). Use these for “what does TORA propose?” — not `*_generation*.png` alone.

TORA is not worse than benchmark Juglet because of deploy prep; it is the same **domain + target-frame** issue, shown as point clouds instead of sherds on a table.

---

## 4. PF++: why visual inspection can succeed while metrics stay poor

### Inference artifacts

`puzzlefusion_plusplus/auto_aggl.py` writes:

- `predict_<part_acc>.npy` — full denoise trajectory
- `gt.npy` — per-part scatter poses
- `init_pose.npy` — global reference pose
- `mesh_file_path.txt` → `artifact/Juglet-000` (anchor-centered OBJs)

### Renderer maps poses → Blender meshes

`Puzzlefusion/renderer/myrenderer.py` — `compute_final_transformation`:

- Undoes `init_pose` and per-part `gt`
- Applies step `transformation[i]`
- Restores `init_pose`

Meshes load at **real scan-relative positions**; `render_results.py` animates all denoise steps into `imgs/` and `video.mp4`. That matches how humans judge “do these sherds fit?”

Part accuracy in the filename is still **~11%** vs fake GT—the same misleading metric as GARF/TORA—but the **display pipeline** is aligned with archaeological review.

### Other PF++ factors on this sample

- Checkpoint: **everyday** denoiser (pottery-like category overlap).
- No extra post-hoc “align entire solution to ref-part random GT” step like GARF’s `anchor_free` block after denoising.

---

## 5. Data / domain factors (all three)

After Step 1 anchor-centering (`preprocess_scan_to_anchor_frame.py`):

- **Max pairwise centroid distance ≈ 4.67 m** (`Dataset/Juglet_anchor_centered/deploy_manifest.json`) — fragments still widely separated on the table.
- Training (GARF/TORA): Breaking Bad / synthetic fractures — pieces **already near** a broken-object configuration.
- **9 pieces**, irregular Metashape meshes, weak connectivity vs synthetic data.
- Models were **not** trained on “tray of unrelated sherds metres apart.”

PF++ can still look good because the **renderer** puts predictions back on separated table geometry; GARF/TORA viz stays in the **augmented point-cloud frame**.

---

## 6. Visual QA only (no assembly GT)

For archaeological deploy, **part_acc / rotation error / shape_cd are not meaningful** — there is no true assembly to score against. The only question that matters is:

> **Do the fragments look plausibly assembled?**

| Framework | Use for visual QA |
|-----------|-------------------|
| **PF++** | `renderer/juglet_deploy_output/0/video.mp4` and last `imgs/*.png` (meshes on table) |
| **GARF** | Deploy HTML with `--scan_layout`: **Predicted (model pose frame)**; ignore metrics and GLB until deploy export is fixed |
| **TORA** | **Primary:** `*_proposed_assembly*.png` (Procrustes on scattered input). **Auxiliary:** `*_generation*.png` (flow endpoint), `*_input.png`, `*_scan_ref.png` |

---

## 7. Why BB / bones / Fractura viz looked fine (Juglet never had assembly GT either)

Earlier tests on **Breaking Bad, bone_syn, Fractura ceramics/egg/bones** looked reasonable in GARF/TORA viz. That was **not** because those HDF5s contained true assembly GT and Juglet does not.

**All datasets use the same dataloader recipe:** per-part recenter, random rotation, normalization. The stored `translations` / `quaternions` (or TORA rotations/translations) are **augmentation poses**, not scanner-verified assembly.

What differs is **mesh layout and what the viz shows**:

| | **BB / bones / Fractura** | **Juglet (scan)** |
|---|---------------------------|-------------------|
| Meshes in HDF5 | Single **broken object**, pieces **close** in local coords | **Table/scanner** layout, **metres apart** even after anchor-center |
| Dataloader GT | Random aug on a **coherent** broken layout | Same code, aug on **incoherent** scan layout |
| GARF benchmark viz | `T_pred @ T_gt⁻¹` on co-located meshes → looks like **snap-together** | `--scan_layout`: **table ref** vs **pose-frame pred** side by side |
| TORA PNGs | Final cloud often **one tight cluster** in display frame | Spread scan ref vs **partial** clustering in aug frame |

So: GARF/TORA “worked” on BB/bones because **geometry + viz tell one story**. Juglet splits **table meshes** and **pose-space assembly** — not because deploy removed GT that BB had.

---

## 8. Why prediction can be fine in pose space but not read as sherds fitting

### What “pose space” is

Each sherd is sampled, **recentered**, **randomly rotated**, and **scaled** to a unit box. The model predicts SE(3) that means: *move these normalized clouds so they belong together.*

**Good in pose space** ≈ nine clouds form one tight cluster with plausible relative rotations (deploy JSON last step had ‖t‖ ~0.07–0.43 per part).

**Sherds fitting on the tray** ≈ OBJ vertices move so fracture rims meet in **anchor-centered mesh / table** coordinates.

Those are linked only through a long transform chain. **PF++ implements it in Blender; default GARF/TORA viz usually does not.**

### Why viz disagrees with pose-space quality

1. **Mixed frames (Juglet GARF `--scan_layout`)**  
   - Scan ref = mesh centroids in **table** layout.  
   - Predicted = model poses in **pose / CoM** frame.  
   Side-by-side they cannot look like the same sherds moving together even when pose prediction is consistent.

2. **Small ‖t‖ in JSON ≠ on the table**  
   Last denoising step can be modest in pose space while **`predicted_assembly.glb`** uses larger poses after post-`anchor_free` snap to fake ref-part GT (~2.7–3.0 ‖t‖ on the deploy run) — an export bug on top of the frame issue.

3. **Relative assembly ≠ scan layout**  
   Perfect pose-space assembly only guarantees fit among **normalized clouds**, not rim contact on the tray without inverse aug + mesh placement.

4. **TORA shows points, not rims**  
   Clustering in pose space may look like a blob in PNGs, not mesh contact the way PF++ video shows.

### Pipeline diagram

```text
Table meshes (anchor-centered, far apart)
        │
        ▼  sample + recenter + random rot + scale   ← dataloader
Pose space: model predicts SE(3) here
        │                    └── "fine" = cluster here (~0.3 |t|)
        │
        ▼  full inverse chain + mesh placement     ← PF++ Blender only
Table meshes moved → reads as assembly to human
```

GARF/TORA default viz stops in the **middle** (or shows middle vs top together). PF++ completes the **bottom** step.

---

## 9. Do GARF/TORA need mesh transform export? (BB vs Juglet)

**They did not gain a new requirement on Juglet.** They never wrote PF++-style “place OBJs on the table” outputs on BB/bones either.

| Question | BB / bones / Fractura | Juglet |
|----------|----------------------|--------|
| What did viz show? | Pose-relative assembly on **co-located** broken meshes (GARF) or **clustered** output cloud (TORA) | **Table scan ref** vs **pose-frame** pred (GARF) or spread vs generation PNGs (TORA) |
| Did that match “looks assembled”? | Yes — **one local broken world** | Often no — **two visual stories** |
| Need mesh chain for same QA as PF++ video? | No — pose/cloud viz ≈ object layout | **Yes**, if the question is *sherds on the tray* |

- **Pose-space / clustering QA** on Juglet: GARF JSON last step, TORA `generation` vs `input`.  
- **Table-level assembly QA**: PF++ render (today), or future GARF/TORA export that mirrors `compute_final_transformation`.

Mesh export for Juglet is not “fixing missing GT”; it is **aligning human judgment (tray geometry) with what the visualization displays.**

---

## 10. Recommended fixes (priority)

1. **GARF:** In `deploy_mode`, **disable** post-inference `anchor_free` GT alignment; export GLB using the **same** poses as the last denoising step (or a PF++-style mesh transform chain).
2. **GARF viz / GLB:** Map poses to anchor-centered mesh frame (like PF++ `compute_final_transformation`) or document that GLB is pose-frame only until that exists.
3. **TORA deploy:** Procrustes assembly PNGs enabled (`save_procrustes_assembly`); optional future mesh export from poses + OBJ path.
4. **All:** Treat `part_acc` / rotation error on deploy as **diagnostic only**; visual mesh assembly (PF++ path) is the right QA for archaeology.

---

## 11. Bottom line

| Framework | Why visual expectation was not met (GARF/TORA) or was met (PF++) |
|-----------|---------------------------------------------------------------------|
| **GARF** | Deploy GLB uses **GT-snapped** poses after `anchor_free` alignment + centroid-local export; JSON trajectory ≠ GLB; scan_layout viz mixes table vs pose frame. |
| **TORA** | Default `generation` PNGs target **scan-layout** `pointclouds_gt` on Juglet; use **`proposed_assembly`** PNGs for paper-faithful SE(3) proposal. |
| **PF++** | **Mesh-space render pipeline** + trajectory on real OBJ positions. |

**Cross-dataset:** BB/bones viz looked assembled because **broken-object mesh layout ≈ pose-space story**. Juglet does not; PF++ (or a future mesh export) bridges pose space to the tray.

---

## 12. TORA paper vs Juglet (reconciliation)

### What the paper claims

From [arXiv:2604.04050v1](https://arxiv.org/html/2604.04050v1) §3.1–3.2 and Appendix §0.F.1:

1. Rectified flow transports part clouds toward **assembled** configurations `x_k(0)` sampled from the ground-truth object.
2. At inference, integration yields `x̂_k(0)`; **Procrustes** recovers per-part SE(3) from **unposed** `P_k` to `x̂_k`.
3. Anchor-free evaluation matches the repo: no privileged pose; ICP in metrics is an eval convention (RPF), not qualitative cheating.

TORA adds **representation alignment** during training only; inference remains RPF-style flow + Procrustes.

### Why Juglet visuals disagreed with “TORA should assemble like GARF”

| | Paper / BB / Fractura | Juglet deploy HDF5 |
|--|----------------------|-------------------|
| Meaning of `pointclouds_gt` / `x_0` | **Assembled** broken-object layout | **Scan** layout on table |
| `*_generation*.png` | Often one tight cluster | Often ≈ spread `scan_ref` |
| Paper-faithful proposal | Procrustes(`P_k` → `x̂_k`) | Same math; still in **normalized point-cloud frame**, not tray OBJs |

So the **algorithm matches the paper**; the **Juglet packaging** does not give an assembled `x_0` to flow toward. That is a dataset/coordinate assumption, not a missing TORA feature.

### PNG glossary (TORA eval)

| File | What it shows |
|------|----------------|
| `*_input.png` | Scattered conditioning clouds `P_k` (anchor-free aug) |
| `*_scan_ref.png` | `pointclouds_gt` — scan layout, **not** true assembly |
| `*_generation*.png` | Flow endpoint `x̂` toward `pointclouds_gt` |
| `*_proposed_assembly*.png` | **`P_k` after Procrustes `T̂_k`** — paper assembly proposal in point-cloud space |

On Juglet, `proposed_assembly` can still differ from PF++ tray video (no mesh/`init_pose` chain). It is the correct object for “what rigid assembly does TORA propose?”

### How to generate Procrustes proposal PNGs

**Option A — Juglet deploy Slurm** (`TORA/eval_juglet_deploy.slurm`): includes `+visualizer.save_procrustes_assembly=true`.

**Option B — Standalone script** (re-runs inference):

```bash
cd /data/gpfs/projects/punim2657/TORA/repo
python scripts/export_procrustes_assembly_viz.py \
  --ckpt ../checkpoints/bbad_everyday_cka.ckpt \
  --data-root ../dataset \
  --log-dir ../eval_runs/juglet_procrustes_viz
```

Or:

```bash
bash /data/gpfs/projects/punim2657/TORA/scripts/export_procrustes_assembly_viz.sh
```

Outputs: `<log_dir>/visualizations/*_proposed_assembly01.png`.

**Implementation:** `tora/procrustes.apply_rigid_transformations`, `FlowVisualizationCallback.save_procrustes_assembly` in `tora/visualizer.py`.

---

## Related documentation

- `ARCHAEOLOGICAL_DEPLOYMENT.md` — deploy requirements and pipeline steps
- `GARF_vs_PuzzleFusion_comparison.md` — benchmark eval (with GT), not deploy
- `GARF_ARCHITECTURE.md` — anchor / `anchor_free` behaviour
