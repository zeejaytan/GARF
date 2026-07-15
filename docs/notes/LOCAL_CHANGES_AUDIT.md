# GARF: Local Changes vs Upstream GitHub (ai4ce/GARF)

**Date:** 2026-03-30
**Local HEAD:** fe5814e (July 28, 2025 — "update news")
**Upstream HEAD:** 2d489c0 (Feb 23, 2026 — "Merge pull request #24")
**Upstream is 4 commits ahead of local.**

---

## Upstream Commits After Local

| Date | SHA | Message |
|------|-----|---------|
| Oct 4, 2025 | bfd0796 | Fix deps |
| Feb 23, 2026 | 25c8055 | Add vis output to glb directly |
| Feb 23, 2026 | c4cfa7f | Add SDPA attention backend with pad-once optimization |
| Feb 23, 2026 | b566bda | Fix cusolver init error |
| Feb 23, 2026 | 2d489c0 | Merge pull request #24 |

---

## Verified Against Live GitHub (not just local git cache)

### Critical Files — Identical Between Local and Current Upstream

| File | Local vs Upstream | Status |
|------|-------------------|--------|
| `assembly/models/denoiser/denoiser_flow_matching.py` | **IDENTICAL** | Bugs #1-#4 exist upstream |
| `assembly/models/denoiser/modules/scheduler.py` | **IDENTICAL** | Bug #1 (vector field) exists upstream |
| `assembly/data/transform.py` | **IDENTICAL** | `rot_mat.T` convention same in both |
| `configs/trainer/default.yaml` | **IDENTICAL** | fp16-mixed, 4 nodes, 500 epochs |
| `assembly/models/denoiser/denoiser_diffusion.py` | **IDENTICAL** | Reference part masking present here |
| `assembly/models/denoiser/denoiser_base.py` | Upstream has +90 lines | Mesh viz, `se3_to_matrix()` — no training impact |

### Files That Differ

| File | What Upstream Added | Training Impact |
|------|---------------------|-----------------|
| `pyproject.toml` | PyTorch 2.8, Python 3.12, CUDA 12.8, new deps | Environment only |
| `weighted.py` | Init rotation COMMENTED OUT, `meshes` key added | **See below** |
| `denoiser_transformer.py` | `use_flash_attn` flag, `forward_sdpa()` method | Attention path only |
| `attention.py` | `forward_sdpa()` method | Attention path only |
| `denoiser_flow_matching.yaml` | `use_flash_attn: False` | Attention path only |
| `eval.py` | `weights_only=False`, extra imports | Eval only |
| `base.py` | `meshes` in data dict, custom `collate_fn` | Test viz only |
| `module.py` | Custom `collate_fn` for test loader | Test viz only |
| `denoiser_base.py` | `se3_to_matrix()`, mesh assembly viz in test_step | Test viz only |

---

## Key Finding: Init Rotation

**Local:** `rotate_pc()` is **ENABLED** (uncommented)
**Upstream (current GitHub):** `rotate_pc()` is **COMMENTED OUT**

This means locally you're applying a random global rotation to each sample as data augmentation. The upstream-trained checkpoints (GARF.ckpt) were trained WITHOUT this augmentation. When fine-tuning with it enabled, the model sees a different distribution than what it learned.

**This is a local deviation but NOT the root cause of convergence failure.**

---

## GitHub Issues — Relevant Findings

### Issue #10: "points_features has nan"
- **Cause:** flash attention + float16 produces NaN from overflow
- **Fix:** Either disable flash attention OR use bfloat16 instead of float16
- **Relevance:** Your local setup forces flash attention (removed SDPA fallback) AND uses `16-mixed` precision. This is a **known instability** confirmed by the upstream maintainers.

### Issue #18: "Questions about Stage 2 Training"
- Authors trained on **4x H100 GPUs**, batch_size=32, took **3 days** for flow matching
- On a single consumer GPU, Stage 2 is much slower — expected behavior
- Advised to monitor **loss curves** not epoch counts

### Issue #16: "Commenting out checkpoint during first training"
- Feature extractor checkpoint must be from Stage 1 pretraining
- Loading wrong checkpoint causes "Missing key(s) in state_dict" — matches your slurm log errors

### No Upstream Issues About Convergence or Rotation Bugs
- 19 closed issues + 6 open issues
- **None** report convergence failure, rotation RMSE problems, or flow matching bugs
- This could mean: (a) the bugs only manifest in certain conditions, or (b) the authors' training setup somehow works despite the issues, or (c) we need to re-examine our diagnosis

---

## Re-examination of Diagnosis in Light of Upstream

Since the upstream code is identical for the critical files, and the authors claim successful training (ICCV 2025 paper, published checkpoints), we need to reconsider:

### Bug #1 (scheduler.py:197 — rotation vector field)

```python
rot_vec_field = p3dt.matrix_to_axis_angle(x_1_rot_mat)
x_t_rot_mat = p3dt.axis_angle_to_matrix(sigma * rot_vec_field) @ x_0_rot_mat
```

**Re-analysis:** This implements the forward process as:
- `x_t = exp(sigma * log(x_1)) @ x_0`
- At sigma=0: `x_t = I @ x_0 = x_0` (correct — starts at ground truth)
- At sigma=1: `x_t = x_1 @ x_0` (NOT `x_1` — but this may be intentional)

The vector field `v = log(x_1)` is the axis-angle of the noise rotation. The model learns to predict this vector field. During inference, the scheduler reverses it with `delta_sigma * v`. This is a valid formulation IF the training and inference are consistent — the model learns to map from any interpolated state back along the same path.

**Revised verdict:** This may actually be a **deliberate design choice**, not a bug. The flow isn't between `x_0` and `x_1`, but rather: starting from `x_0`, progressively apply `exp(sigma * log(x_1))` as a perturbation. The vector field target `log(x_1)` is constant regardless of sigma, which simplifies learning. This is similar to how some flow matching papers define the velocity as a function of the noise endpoint only.

**However**, the fact that the authors published successful results with this code strongly suggests it IS correct for their formulation. The convergence failure may lie elsewhere.

### Bug #2 (ref parts not excluded from flow matching loss)

The diffusion variant excludes reference parts:
```python
ref_part_mask = ~data_dict["ref_part"][part_valids]
mse_loss = F.mse_loss(model_pred[ref_part_mask], gt_noise[ref_part_mask])
```

The flow matching variant does NOT:
```python
vec_mse_loss = F.mse_loss(model_pred, target, reduction="none")
```

But the target IS zeroed for ref parts (line 88). So the model is trained to predict zero velocity for reference parts. This is a weaker form of masking — it still works, just wastes some gradient capacity. **Likely not a convergence blocker** since the authors trained successfully.

### Issue #10 — NaN from Flash Attention + FP16

**This is the most actionable finding.** The upstream maintainers confirmed:
- flash_attn + float16 → NaN in point features
- Fix: use bfloat16 or disable flash attention

Your local setup:
- Removed SDPA fallback → forced flash attention
- Precision: `16-mixed` (float16)
- This is the **exact combination** reported to produce NaN

If NaN appears intermittently in the features, the loss would appear to "not converge" because gradients are corrupted. The rot_rmse staying near 90° could be explained by NaN-corrupted updates that prevent any learning.

---

## Revised Verdict

| Finding | Severity | Source | Action |
|---------|----------|--------|--------|
| Flash attention + fp16 → NaN (Issue #10) | **CRITICAL** | Upstream confirmed bug | Switch to `bf16-mixed` or add back SDPA fallback |
| Init rotation enabled (upstream has it off) | MODERATE | Local change | Comment it out to match upstream |
| Ref parts included in flow matching loss | LOW | Upstream design | Not a blocker — authors trained with this |
| Rotation vector field formulation | UNCLEAR | Need deeper analysis | May be correct — authors published results with it |
| Missing `weights_only=False` in eval.py | LOW | Local divergence | Add back for eval compatibility |

### Primary Recommendation

**The #1 suspect is now flash attention + fp16 causing NaN** — this is a confirmed upstream issue (#10), and your local code forces this exact combination by removing the SDPA fallback. Fix options:

1. Change `precision: 16-mixed` → `precision: bf16-mixed` in `configs/trainer/default.yaml`
2. OR: Restore the SDPA fallback from upstream and set `use_flash_attn: False`
3. AND: Comment out init rotation in `weighted.py` to match upstream training conditions
