# GARF Architecture & Inference Deep Dive

## What GARF Is

GARF (Generalizable 3D Reassembly for Real-World Fractures) is a **generalizing reassembly-by-pose-inference model** trained on 1.9 million fractures. It uses flow matching to jointly denoise fragment poses, learning geometric priors about how fragments fit together. It is **not** just pose regression — it provides real zero-shot capability on unseen shapes and real-world fractures. It is also not a classical puzzle solver that uses explicit search, constraint solving, or symbolic assembly planning.

**Paper:** ICCV 2025, arXiv:2504.05400

---

## Pipeline Overview

```
HDF5 meshes (assembled) 
  → Data pipeline: recenter + random rotate each piece (scatter)
  → Store inverse transforms as GT (the "answer")
  → Feature extraction: PointTransformerV3 encodes each fragment's geometry
  → Denoising: flow matching iteratively refines random poses → predicted assembly
  → Metrics: compare predicted poses against GT
```

---

## Two-Stage Architecture

### Stage 1: Feature Extractor (Frozen)

- **Model:** PointTransformerV3 (pretrained via fracture-aware pretraining)
- **Input:** Per-fragment point clouds + normals in scattered coordinates
- **Output:** Per-fragment latent features
- **Checkpoint:** `output/feature_extractor.ckpt` (extracted from GARF.ckpt, 402 params)
- **Code:** Loaded at `denoiser_base.py:60-67`, called at `denoiser_base.py:345`
- **Key point:** Operates purely on geometry — no GT information

### Stage 2: Denoiser (Flow Matching)

- **Model:** Transformer-based SE(3) denoiser
- **Input at each step:**
  - Current noisy poses `x` (7D: translation + quaternion)
  - Point cloud latent features
  - Per-part scale factor
  - Boolean `ref_part` flag (NOT the GT pose values)
  - Timestep
- **Output:** Predicted vector field for one reverse step
- **Code:** `assembly/models/denoiser/modules/denoiser_transformer.py`
- **Key point:** GT values never enter the network — only a boolean flag for the anchor

---

## Inference Pipeline (test_step)

**File:** `assembly/models/denoiser/denoiser_base.py:304-625`

### Step-by-step trace:

1. **Load GT transforms** (line 318-320) — used for anchor pinning and metrics only
2. **Initialize random poses** (line 322-328) — pure random Gaussian translations + uniform SO(3) rotations
3. **Pin anchor to GT** (line 331-338) — overwrites one fragment's pose with its GT position
4. **Extract features** (line 345) — PointTransformerV3 on scattered point clouds, no GT
5. **One-step init** (optional, line 347-371) — coarse denoising from pure noise, then re-pin anchor
6. **Main denoising loop** (line 373-396):
   - Denoiser predicts vector field (no GT input)
   - Scheduler takes reverse step (GT-free, `scheduler.py:320-369`)
   - Re-pin anchor to GT after each step
7. **Metric computation** (line 466-518) — compare predictions vs GT
8. **Visualization** (line 593) — `T_final = T_pred @ T_gt_inv` for mesh rendering

### Anchor mechanism explained:

The assembly problem has **gauge freedom** — any rigid motion applied to the entire assembly is equally valid. Pinning one part (the anchor) to a known position removes this ambiguity. The anchor's "GT" is really just a coordinate frame definition, not supervision.

---

## Exhaustive GT Usage Map

| Location | Lines | What GT Does | Required for Reassembly? |
|----------|-------|-------------|------------------------|
| Anchor pinning | 331, 338, 368, 393 | Fixes reference part position each step | **Replaceable** — any chosen pose works |
| Anchor-free alignment | 401-424 | Post-hoc rigid alignment for metrics | No — evaluation only |
| Metric computation | 466-518 | part_acc, rmse_r, rmse_t, shape_cd | No — evaluation only |
| Visualization transform | 593 | `T_pred @ T_gt_inv` for mesh display | No — rendering only |
| Training forward pass | flow_matching.py:82-91 | Creating noisy samples from GT | No — training only |

**Conclusion:** The denoiser network never receives GT values. All reassembly capability comes from learned geometric priors. GT only defines the coordinate frame (anchor) and enables evaluation.

---

## Data Pipeline

**File:** `assembly/data/breaking_bad/weighted.py:71-186`

For each fragment:
1. Load vertices/faces from HDF5 (stored in assembled position)
2. Compute centroid → becomes GT translation
3. Apply random rotation → inverse becomes GT quaternion
4. Sample surface points (area-weighted)
5. Normalize scale

**Output:**
- `pointclouds`: scattered point clouds (model input)
- `translations`: GT translations (centroid offsets)
- `quaternions`: GT rotations (inverse of random rotation, scalar-first `[w,x,y,z]`)

### Quaternion conventions:
- **PyTorch3D / internal:** scalar-first `[w, x, y, z]`
- **Scipy:** `[x, y, z, w]`
- **JSON results:** `[tx, ty, tz, qx, qy, qz, qw]` — scipy convention
- **Three.js:** constructor takes `(x, y, z, w)`, but our viz format is `[tx, ty, tz, qw, qx, qy, qz]`

---

## Key Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `anchor_free` | `false` | If true, no part is pinned during denoising. Post-hoc alignment used for metrics. |
| `one_step_init` | `false` | If true, do one coarse denoising step from pure noise before main loop |
| `random_anchor` | `true` (train) / `false` (eval) | If false, anchor is the largest fragment |
| `write_to_json` | `false` | Save per-sample results to JSON |
| `save_assembly` | `false` | Save predicted mesh assemblies |
| `num_inference_steps` | 20 | Number of denoising steps |
| `max_iters` | 1 | Number of full denoising iterations |

---

## HDF5 Dataset Formats

### bone_real.hdf5 (flat)
```
/data_split/pig_bone/val: ["pig_bone_20", "pig_bone_22", ...]
/pig_bone/pig_bone_20/pieces/0/vertices, faces
```

### bone_synthetic.hdf5 (nested)
```
/data_split/pig/val: ["pig/synthetic_fracture/20/fractured_15", ...]
/pig/synthetic_fracture/20/fractured_15/pieces/0/vertices, faces
```

### fractura_real.hdf5
```
/data_split/ceramics/val: ["ceramics/blue_pot", ...]
/ceramics/blue_pot/pieces/0/vertices, faces
Categories: bones (16), ceramics (8), egg (3)
```

### tray_archaeological.hdf5
```
/data_split/artifact/val: ["artifact/Tray-000"]
/artifact/Tray-000/pieces/0-39/vertices, faces
Note: 40 pieces stored in SCANNED positions (not assembled)
```

---

## Inference Without GT (for unknown assemblies)

For datasets where the true assembly is unknown (e.g., tray_archaeological):

1. **The model CAN predict assembly** — all reassembly capability comes from learned geometric priors
2. **Anchor handling:** Pick one large fragment, place at origin with identity rotation. This is just defining a coordinate frame, not providing supervision.
3. **Or use `anchor_free=True`:** All parts denoise freely, output in arbitrary coordinate frame
4. **Data pipeline adaptation needed:** Skip random-scatter step since pieces are already in their scanned positions. Feed point clouds directly.
5. **Metrics are meaningless** without GT — evaluate by visual inspection of predicted assembly

---

## Key Code Files

| File | Purpose |
|------|---------|
| `assembly/models/denoiser/denoiser_base.py` | test_step(), anchor pinning, metrics |
| `assembly/models/denoiser/denoiser_flow_matching.py` | Flow matching forward/reverse |
| `assembly/models/denoiser/modules/denoiser_transformer.py` | Denoiser network architecture |
| `assembly/models/denoiser/modules/scheduler.py` | SE3FlowMatchEulerDiscreteScheduler |
| `assembly/data/breaking_bad/base.py` | Dataset loading, HDF5 reading, filtering |
| `assembly/data/breaking_bad/weighted.py` | Area-weighted point sampling, transform() |
| `assembly/data/breaking_bad/module.py` | Lightning DataModule |
| `configs/model/denoiser_flow_matching.yaml` | Model + inference config |
| `configs/data/breaking_bad.yaml` | Dataset config |
| `eval.py` | Evaluation entry point |
| `train.py` | Training entry point |
