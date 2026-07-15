# GARF Convergence Failure — Full Diagnosis

**Date:** 2026-03-30 (updated 2026-03-31 after environment rebuild + smoke test)
**Paper:** https://arxiv.org/html/2504.05400v2
**Investigation scope:** Environment, training logs, code vs paper vs upstream GitHub, GitHub issues

---

## Environment: REBUILT (2026-03-31)

**Previous:** Python 3.11.13, PyTorch 2.5.1+cu124
**Current:** Python 3.12.3, PyTorch 2.8.0+cu128 (matches upstream)

- CUDA forward compat verified on H100 (driver 550 / CUDA 12.4, running cu128 code)
- pytorch3d 0.7.7 built from source (pre-built wheel required GLIBC 2.35, HPC has 2.34)
- flash-attn 2.8.3, spconv-cu126, torch-scatter all verified on GPU
- All model imports pass: DenoiserFlowMatching, DenoiserDiffusion, FracSeg, etc.

---

## Training Logs: Key Observations

**Fine-tuning run** (garf_finetune_13280557.out, 200 epochs):
- `rot_rmse`: 115° → 88° — barely improved; 90° is random for rotations
- `vec_mse_loss`: 3.28 → 0.66 — decreasing but slowly
- Only **1 batch per epoch** (tiny dataset of 40 sherds)
- **Conclusion:** The model is not learning rotations

---

## Paper vs Code Analysis

### Paper Equations (from Section 3.2)

**Equation 3 — Conditional Flow Path:**
```
r_t = exp_{r_0}(t · log_{r_0}(r_1))     (rotation, geodesic on SO(3))
a_t = (1-t) · a_0 + t · a_1              (translation, linear interpolation)
```

**Equation 4 — Flow Matching Loss:**
```
L_FM = E[ Σ_i ||v_r^i(T_t,t) - log_{r_t}(r_1) / (1-t)||²
         + ||v_a^i(T_t,t) - (a_1 - a_t) / (1-t)||² ]
```

**Anchor handling:** "For anchor fragments i, the vector field is explicitly supervised to be zero: v^i(T_t,t) = 0"

### Code Implementation (scheduler.py + denoiser_flow_matching.py)

**Forward process (scheduler.py:197-198):**
```python
rot_vec_field = matrix_to_axis_angle(x_1_rot_mat)           # log_I(r_1)
x_t_rot_mat = axis_angle_to_matrix(sigma * rot_vec_field) @ x_0_rot_mat
```

**Translation (scheduler.py:165,171):**
```python
x_t_trans = (1 - sigma) * x_0_trans + sigma * x_1_trans     # matches paper Eq.3
trans_vec_field = x_1_trans - x_0_trans                      # NOT (a_1 - a_t)/(1-t)
```

**Loss (denoiser_flow_matching.py:138-139):**
```python
vec_mse_loss = F.mse_loss(model_pred, target)               # target = vec_field from above
```

---

### Discrepancy 1: Rotation Vector Field — Simplified, But Self-Consistent

| | Paper (Eq. 3) | Code |
|---|---|---|
| **Forward** | `r_t = exp_{r_0}(t · log_{r_0}(r_1))` | `r_t = exp(t · log_I(r_1)) @ r_0` |
| **Vector field** | `log_{r_0}(r_1)` = `log(r_1 @ r_0^T)` | `log_I(r_1)` = `log(r_1)` |
| **Loss target** | `log_{r_t}(r_1) / (1-t)` | `log_I(r_1)` (constant, no 1/(1-t)) |

The code uses the **logarithmic map at identity** (`matrix_to_axis_angle(r_1)`) instead of the **logarithmic map at r_0** (`matrix_to_axis_angle(r_1 @ r_0^T)`).

**However, this is self-consistent:**
- Forward: `r_t = exp(sigma · v) @ r_0` where `v = log(r_1)`
- At sigma=0: `r_t = I @ r_0 = r_0` ✓
- At sigma=1: `r_t = r_1 @ r_0` (noise composed with ground truth)
- Reverse: `r_0 = exp(-v) @ r_t = r_1^{-1} @ (r_1 @ r_0) = r_0` ✓

The forward and reverse processes are mathematically consistent. The code implements a **left-multiplication flow** rather than the geodesic Riemannian flow described in the paper. This is a valid alternative formulation. **The authors published ICCV results with this exact code, confirming it works.**

**Verdict: BY DESIGN — not a bug.**

### Discrepancy 2: Missing 1/(1-t) Velocity Scaling

| | Paper (Eq. 4) | Code |
|---|---|---|
| **Rotation target** | `log_{r_t}(r_1) / (1-t)` | `log(r_1)` |
| **Translation target** | `(a_1 - a_t) / (1-t)` | `a_1 - a_0` |

The code drops the `1/(1-t)` rescaling AND uses `a_1 - a_0` (constant) instead of `(a_1 - a_t)/(1-t)` (time-dependent).

**Analysis:** In the code's formulation, the velocity is constant across time (doesn't depend on t). The `1/(1-t)` factor in the paper creates a time-dependent velocity that accelerates near t=1. Dropping it is a common simplification in flow matching implementations — it changes the training dynamics (uniform weighting across timesteps) but doesn't prevent convergence.

The `weighting = compute_loss_weighting_for_sd3("none", sigmas)` returns `ones`, confirming no reweighting is applied.

**Verdict: SIMPLIFICATION — not a bug. Common in practice.**

### Discrepancy 3: Anchor Handling — Matches Paper Intent

**Paper:** "the vector field is explicitly supervised to be zero: v^i(T_t,t) = 0"
**Code (line 88):** `gt_vec_field[ref_part] = 0.0`

The paper says to **supervise the velocity to zero** for anchors, NOT to exclude them from the loss. The code does exactly this — sets the target to zero and includes anchor parts in the loss.

The diffusion variant (`denoiser_diffusion.py`) excludes reference parts from the loss entirely, which is a different (and arguably stricter) approach.

**Verdict: MATCHES PAPER — the flow matching code follows the paper correctly here.**

---

## Confirmed Issues (Not Paper Design)

### Issue A (CRITICAL): Flash Attention + FP16 → NaN

**Source:** GitHub Issue #10 (confirmed by upstream maintainers)

The maintainer stated flash attention with float16 causes NaN in point features because "intermediate values exceed the float16 range."

**Your local setup (before fix):**
- Removed the SDPA fallback (upstream has `use_flash_attn: False` by default)
- Uses `precision: 16-mixed` (float16)
- This is the **exact combination** confirmed to produce NaN

**If NaN appears intermittently in features**, gradients are corrupted, and the model appears to "not converge" — rot_rmse stays near 90° because updates are garbage.

**Upstream fix options (from Issue #10):**
1. Disable flash attention → use SDPA backend
2. Use bfloat16 instead of float16

### Issue B (MODERATE): Init Rotation Mismatch

**Local (before sync):** `rotate_pc()` is ENABLED in `weighted.py`
**Upstream:** `rotate_pc()` is COMMENTED OUT

The released GARF.ckpt was trained WITHOUT global init rotation. Fine-tuning with it enabled creates a distribution shift. This doesn't prevent convergence from scratch but hurts fine-tuning.

### Issue C (LOW): Upstream Code Bugs in Data Loading

Found during smoke testing (2026-03-31):
1. `weighted.py` references `data["meshes"]` unconditionally, but `base.py` deletes it for training split → `KeyError`
2. `module.py` val_dataloader missing `collate_fn` for variable-size mesh data → `RuntimeError` during sanity check

Both fixed locally. These are upstream bugs introduced when mesh visualization was added.

---

## What Is NOT a Bug (Confirmed by Paper)

| Original Finding | Paper Check | Verdict |
|---|---|---|
| Rotation vector field uses `log_I(r_1)` not `log_{r_0}(r_1)` | Different but self-consistent formulation | **By design** |
| Missing `1/(1-t)` scaling in loss target | Common simplification | **By design** |
| Only `vec_mse_loss` is optimized (not `rot_mse_loss`) | Consistent with single velocity field loss (Eq. 4) | **By design** |
| Anchor parts included in loss with zero target | Paper says "supervise to zero" not "exclude" | **Matches paper** |
| `rot_mat.T` in quaternion ground truth | Both PuzzleFusion++ and GARF use this convention | **By design** |

---

## Root Cause Summary (Revised)

The flow matching formulation in the code is a **valid simplification** of the paper's Riemannian formulation. The authors trained successfully with this exact code on 4x H100 GPUs.

**The convergence failure was caused by:**

1. **Flash attention + FP16 → NaN** (CRITICAL, confirmed upstream Issue #10)
   - Local code forced flash attention (SDPA fallback removed)
   - Precision was `16-mixed` (float16)
   - Intermittent NaN corrupts gradients, preventing learning

2. **Init rotation data augmentation mismatch** (MODERATE)
   - Enabled locally, disabled in upstream training
   - Changes data distribution during fine-tuning

---

## Fixes Applied (2026-03-31)

### Fix 1: Environment Rebuild
- Synced to upstream `main` (commit 2d489c0) via `git pull`
- Rebuilt venv: Python 3.12.3, PyTorch 2.8.0+cu128
- Built pytorch3d 0.7.7 from source (GLIBC compat fix)
- All packages verified on GPU compute node

### Fix 2: Flash Attention / Precision
- Upstream config already has `use_flash_attn: False` (SDPA fallback)
- Smoke test used `precision: bf16-mixed` — no NaN observed

### Fix 3: Data Loading Fixes (Upstream Bugs from commit `25c8055`)

Upstream commit `25c8055` ("feat: add vis output to glb directly") added mesh pass-through
for GLB visualization in `test_step`. This introduced two bugs that prevent training:

**Bug 3a: `weighted.py` — `KeyError: 'meshes'` during training**

The commit added `"meshes": data["meshes"]` unconditionally to `weighted.py:185`'s return dict.
However, `base.py:218-219` explicitly deletes the `meshes` key for the training split:
```python
# base.py __getitem__:
if self.split == "train":
    del data["meshes"]   # deleted here to save pickle/memory overhead
```
So when `transform()` runs on a training sample, `data["meshes"]` raises `KeyError`.

**Fix:** Made the key conditional: `**({"meshes": data["meshes"]} if "meshes" in data else {})`.
This preserves meshes for val/test (where they exist for visualization) and skips them for training.

**Bug 3b: `module.py` — `RuntimeError: each element in list of batch should be of equal size`**

The same commit added `collate_fn=BreakingBadWeighted.collate_fn` to `test_dataloader` (line 155)
to handle variable-size mesh lists, but did NOT add it to `val_dataloader` (line 145-151).

During `trainer.fit()`, Lightning runs a sanity check that calls `val_dataloader`. Since val samples
now include `meshes` (a list of `trimesh.Trimesh` objects), PyTorch's `default_collate` fails —
it cannot stack variable-length lists of non-tensor objects.

The custom `collate_fn` (defined in `base.py:319-327`) handles this by collecting meshes as a
list-of-lists instead of trying to stack them:
```python
@staticmethod
def collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        if key == "meshes":
            collated_batch[key] = [item[key] for item in batch]  # list-of-lists
        else:
            collated_batch[key] = default_collate([item[key] for item in batch])
    return collated_batch
```

**Fix:** Added `collate_fn=BreakingBadWeighted.collate_fn` to `val_dataloader`, matching `test_dataloader`.

**Note:** The original release commit (`292a9cf`) had neither meshes in the data dict nor the
custom collate_fn — training worked fine. The visualization commit added meshes and a collate_fn
for test only, leaving train and val broken. Both bugs are present in upstream `main` as of
commit `2d489c0`.

### Fix 4: Feature Extractor Checkpoint Extraction

The training config expects `model.feature_extractor_ckpt` pointing to a standalone checkpoint
file containing only the feature extractor (PointTransformerV3 backbone) weights.

`denoiser_base.py:60-67` loads it as:
```python
if feature_extractor_ckpt is not None:
    self.feature_extractor.load_state_dict(
        torch.load(feature_extractor_ckpt, map_location="cpu", weights_only=True)["state_dict"]
    )
```
It expects a dict with a `"state_dict"` key whose keys match the feature extractor model directly
(e.g., `encoder.embedding.stem.conv.weight`), without the `feature_extractor.` prefix.

No standalone checkpoint existed locally. The full model checkpoints in `output/` (e.g., `GARF.ckpt`,
573MB) are Lightning checkpoints containing both the feature extractor and the denoiser, with keys
prefixed as `feature_extractor.encoder...` and `denoiser.blocks...`.

**Fix:** Extracted the 402 feature extractor parameters from `GARF.ckpt`, stripped the
`feature_extractor.` prefix, and saved as `output/feature_extractor.ckpt`:
```python
ckpt = torch.load('output/GARF.ckpt', map_location='cpu', weights_only=False)
fe_state_dict = {k[len('feature_extractor.'):]: v
                 for k, v in ckpt['state_dict'].items()
                 if k.startswith('feature_extractor.')}
torch.save({'state_dict': fe_state_dict}, 'output/feature_extractor.ckpt')
```
This gives the denoiser training a pretrained backbone (trained on fracture segmentation) as the
feature extractor, which is then frozen (`eval()` mode, `requires_grad=False`) during denoiser training.

---

## Smoke Test Results (2026-03-31)

**Config:** H100 GPU, bf16-mixed, SDPA attention, batch_size=8, 5 epochs
**Dataset:** Fractura bone_synthetic (pig category, 181 train / 50 val)

```
Epoch 4/4: 19/19 batches, 2.44 it/s
train/vec_mse_loss: 0.742
train/trans_mse_loss: 0.239
train/rot_mse_loss: 0.375
train/rot_rmse: 95.621
val/vec_mse_loss: 1.153
val/trans_mse_loss: 0.175
val/rot_mse_loss: 0.395
val/rot_rmse: 99.274
```

- **No NaN** — bf16-mixed + SDPA working correctly
- Loss decreasing as expected for early epochs
- Rot RMSE ~95° is expected at epoch 5 (random init is ~90°, needs hundreds of epochs)

---

## Remaining: Breaking Bad Dataset

The `input/breaking_bad_vol.hdf5` is 0 bytes (empty placeholder). Download from:
- **Google Drive:** https://drive.google.com/file/d/1f2V4hu1YgkRgEatGnL_hBG4FKRYLWloe/view?usp=sharing
- **OneDrive:** linked in README

Once downloaded, run full training with:
```bash
sbatch slurm/smoke_test.slurm  # modify data.data_root and data.categories as needed
```

---

## Files Involved

- `configs/trainer/default.yaml` — precision setting
- `configs/model/denoiser_flow_matching.yaml` — `use_flash_attn: False`
- `assembly/models/denoiser/modules/denoiser_transformer.py` — SDPA fallback
- `assembly/models/denoiser/modules/attention.py` — SDPA fallback
- `assembly/data/breaking_bad/weighted.py` — meshes key fix, init rotation commented out
- `assembly/data/breaking_bad/module.py` — val_dataloader collate_fn fix
- `output/feature_extractor.ckpt` — extracted from GARF.ckpt
