# GARF vs PuzzleFusion++ vs TORA — Complex Fragment Evaluation Comparison

**Dates:** 2026-04-08 (PF++ denoiser-only) / 2026-04-15 (PF++ oracle) / 2026-04-26 (TORA zero-shot N=3 + BoN=10 + anchor-free + agreement-gate follow-up, jobs 24342475 and 24343146)
**Evaluation datasets:** Fractura bone synthetic (pig, rib), Fractura real (ceramics, egg, bones)
**Excluded:** Tray archaeological dataset

---

## Summary Table — Part Accuracy

| Dataset | Samples | GARF | PF++ denoiser-only | PF++ oracle (full pipeline) | TORA (best-of-3, zero-shot) |
|---|---|---|---|---|---|
| bone_syn_pig | 21 | **78.1%** | 20.6% | 26.4% | 65.3% |
| bone_syn_rib | 6/11* | **63.8%** | 22.8% | 37.7% | 87.4%‡ |
| fractura_ceramics | 8 | 56.3% | 21.5% | **90.4%** | 21.5% |
| fractura_egg | 3 | 26.1% | 26.1% | **100.0%** | 26.1% |
| fractura_bones | 16 | 58.3% | 46.9% | 58.3% | 46.9% |

\*rib: GARF and TORA evaluated all 11 samples (5–32 parts); PF++ limited to 6 samples (≤20 parts due to architecture limit).
‡TORA's part_acc is inflated on organic many-fragment data because Hungarian matching with sub-Chamfer threshold can score "right neighbourhood, wrong rotation" as a hit when many small bone fragments are interchangeable. **Compare rotation_error across frameworks instead.**

### Full metric table — rotation (unit-invariant)

TORA columns: N=3 anchor-fixed (baseline), N=10 anchor-fixed (best config), N=3 anchor-free (ablation).

| Dataset | GARF | PF++ den | PF++ orc | TORA N=3 | TORA N=10 | TORA AF |
|---|---|---|---|---|---|---|
| bone_syn_pig | **20.5** | 75.4 | 72.5 | 45.7 | 40.6 | 46.0 |
| bone_syn_rib | **33.2** | 72.1 | 66.0 | 43.4 | 39.4 | 39.4 |
| fractura_ceramics | 38.6 | 65.8 | **18.3** | 54.0 | 49.3 | 58.7 |
| fractura_egg | 57.0 | 61.7 | **13.2** | 38.0 | 36.8 | 37.1 |
| fractura_bones | 46.9 | 44.6 | 44.1 | 19.9 | **16.2** | 24.2 |

### Full metric table — translation (normalised, unit-comparable)

GARF/PF++ report translation in their own normalised frame; TORA reports `t_err × scale` (real units).
The TORA column below is **TORA's reported trans / per-sample scale** so all numbers are in normalised units.

| Dataset | GARF Trans | PF++ den Trans | PF++ orc Trans | TORA BoN Trans (norm) |
|---|---|---|---|---|
| bone_syn_pig | **0.061** | 0.284 | 0.272 | 0.227 |
| bone_syn_rib | **0.068** | 0.234 | 0.178 | 0.135 |
| fractura_ceramics | 0.106 | 29.55† | **0.035** | 0.243 |
| fractura_egg | 0.135 | 19.41† | **0.026** | 0.238 |
| fractura_bones | 0.115 | 21.91† | 0.115 | **0.106** |

†PF++ denoiser-only trans/CD on real data are inflated (no assembly-level normalisation) — not directly comparable.
TORA's raw `translation_error` field is in real mesh units (multiplied by per-sample scale at line 243 of `tora/eval/metrics.py`); divide by `scales` to get the normalised value above.

**Winners by rotation_error:**
- GARF: pig, rib (synthetic bones, many parts)
- PF++ oracle: ceramics, egg (real, few parts, with oracle correspondences)
- **TORA: bones** (the only framework under 30° on real fractured bones)

---

## Experimental Setup

### GARF
- **Checkpoint:** `output/GARF.ckpt` (pre-trained on Breaking Bad "everyday")
- **Inference:** Flow matching, 20 denoising steps, one-step initialisation
- **Data format:** HDF5 with meshes → 5000 points sampled per fragment
- **Normalisation:** Two-level (assembly-level + per-part), all fragments scaled to unit range
- **GT anchor:** Reference part's GT pose (R, t) fixed throughout denoising (same as PF++)
- **Correspondence input:** None — fracture surface labels and adjacency graph are loaded but **not consumed** by the denoiser model (dead code paths)
- **Verifier:** None — GARF removed PF++'s verifier entirely; relies on flow matching to predict all N-1 poses in parallel
- **Max parts:** Unlimited
- **Evaluation script:** `eval_complex.slurm` (2026-04-03)

### PuzzleFusion++ — Denoiser-Only Mode
- **Checkpoint:** `output/denoiser/everyday_epoch2000_bs64/training/last.ckpt`
- **Inference:** DDPM diffusion, 20 steps, `verifier.max_iters=1` (no verify/merge)
- **Data format:** Pre-sampled .npz with 1000 points per fragment
- **Normalisation:** Per-part only (each fragment independently to [-1, 1])
- **GT anchor:** Reference part's GT pose (R, t) fixed throughout denoising (same as GARF)
- **Correspondence input:** None
- **Max parts:** 20 (hardcoded architecture limit)
- **Evaluation script:** `scripts/eval_complex.slurm` (2026-04-08)

### TORA — Zero-Shot *(2026-04-26)*
- **Checkpoint:** `bbad_everyday_cka.ckpt` (TORA flow-matching denoiser pre-trained on Breaking Bad "everyday" with CKA self-distillation; same training subset as GARF and PF++).
- **Inference:** Flow matching with **3 stochastic generations per input**, then per-sample best-of-N selection on rotation error. Same protocol as TORA's prior thinwalled (5.30°) and bbad_artifact (8.30°) zero-shot runs.
- **Data format:** HDF5 meshes (Fractura's native format), 5000 points sampled per sample at runtime — same source files GARF reads.
- **Normalisation:** Per-assembly `pts_gt /= max(abs(pts_gt))` (centred, scaled to [-1, 1]).
- **GT anchor:** Anchor-fixed mode (`anchor_free: false`) — largest part keeps its CoM-frame pose, matching GARF/PF++.
- **Correspondence input:** None.
- **Verifier:** None.
- **Max parts filter:** Per-subset (matches GARF table): pig 5–20, rib 5–32, ceramics 3–12, egg 3–5, bones 2–3.
- **Evaluation script:** `eval_fractura_subsets.slurm` (job 24342475, 5 min A100).
- **Results:** `TORA/eval_runs/fractura_{subset}_24342475/`.

### PuzzleFusion++ — Oracle Full Pipeline *(new)*
- **Checkpoints:** same denoiser + `output/verifier/everyday_epoch100_bs64/training/last.ckpt`
- **Inference:** DDPM + Verifier Transformer, `max_iters=6` (denoise → verify → merge loop)
- **Data format:** same .npz but **assembly-normalised** so the combined point cloud fits in [−0.5, 0.5]³
- **GT anchor:** Reference part's GT pose (R, t) fixed throughout denoising (same as GARF)
- **Correspondence input:** **oracle matching data synthesised from ground-truth geometry** — see "Oracle Matching Data" section below
- **Max parts:** 20
- **Evaluation script:** `scripts/eval_complex_oracle.slurm` (2026-04-15)

### Data Conversion
- HDF5 meshes → trimesh surface sampling (1000 pts/part) → PuzzleFusion .npz format
- Connectivity graph computed from shared vertices (5-decimal precision)
- Reference part selected as largest fragment by bounding box scale
- Conversion script: `convert_hdf5_to_npz.py`
- Oracle + normalisation script: `generate_oracle_matching_data.py`

### Shared GT Baseline — Reference Part Anchor

**All four configurations** (GARF, PF++ denoiser-only, PF++ oracle, TORA) receive the same ground-truth signal at inference time: the **reference fragment's exact GT pose** (translation + rotation) is fixed as an anchor throughout the denoising process. This is standard practice in the assembly literature — one piece defines the world coordinate frame, and the model predicts the remaining N-1 poses relative to it.

In all three codebases this appears as some form of:
```
noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]  # at every step
```

The reference part is the largest fragment (by bounding box in PF++, by surface area in GARF, by point count in TORA).

**What differs** between configurations is exclusively:
| | GARF | PF++ denoiser | PF++ oracle | TORA |
|---|---|---|---|---|
| GT reference pose | Yes | Yes | Yes | Yes |
| Fracture correspondences | No | No | **Yes (oracle)** | No |
| Verifier merge loop | No | No | **Yes** | No |
| Stochastic best-of-N | No (deterministic) | No | No | **Yes (N=3)** |

GARF loads `fracture_surface_gt` and `graph` from the HDF5, but these are **dead code** — neither is passed to the denoiser transformer's `forward()` method. The graph mask capability exists in the codebase (commented-out code in `denoiser_transformer.py`) but was disabled. GARF replaced PF++'s correspondence-based verifier with a stronger flow-matching denoiser that predicts all poses in parallel.

---

## Oracle Matching Data — What It Does and Doesn't Provide

PuzzleFusion++'s full pipeline uses a Verifier Transformer that decides "should these two fragments be merged?" based on pre-computed **matching data** — for each pair of adjacent fragments, it stores point correspondences on the shared fracture surface. In the original paper this comes from **Jigsaw** (Lu et al., CVPR 2024), a separate neural network trained to predict fracture-surface correspondences.

No Jigsaw model was available for the Fractura datasets, so we synthesised an **oracle** substitute from ground-truth geometry. For each sample we compute:

1. **Critical points** — points whose nearest neighbour on *another* fragment lies within ε = 0.02 (in the normalised frame). These are the fracture-surface points.
2. **Edges** — pairs of fragments that share at least `MIN_EDGE_CORR = 3` corresponding critical points.
3. **Correspondences** — for each edge, pair each critical point on fragment *i* with its nearest critical point on fragment *j*.

### What the verifier receives
- Which fragments are adjacent (the edge list)
- Which points on each fragment used to be neighbours across the fracture

### What the verifier does *not* receive
- The ground-truth poses (R, t) of any fragment
- Where fragments go in 3D space
- The overall assembled shape

The denoiser still predicts all SE(3) poses from scratch. The verifier only *checks* whether the denoiser's predicted poses bring corresponding fracture-surface points close to each other in world space, quantising Chamfer distance into 6 bins (thresholds 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 100) and running a classifier to decide merge/reject.

### Why this is an upper bound, not a fair comparison
Jigsaw's real-world accuracy on Breaking Bad is ≈70%; the oracle gives **100%** correspondence precision. So the oracle results represent the **ceiling of what the PF++ pipeline could achieve with a perfect Jigsaw**, not realistic deployment performance. A realistic number sits between denoiser-only and oracle.

### Oracle matching stats (generated)
| Dataset | Avg edges / sample | Avg crit pts / part | Avg corrs / sample |
|---|---|---|---|
| bone_syn_pig | 14.7 | 486 | 6488 |
| bone_syn_rib | 11.2 | 436 | 5346 |
| fractura_ceramics | 10.1 | 189 | 530 |
| fractura_egg | 4.7 | 55 | 116 |
| fractura_bones | 1.2 | 103 | 102 |

---

## Per-Dataset Analysis

### 1. Bone Synthetic — Pig (21 samples, 5–20 parts)
- **GARF 20.5° / 0.061  |  PF++ den 75.4° / 0.284  |  PF++ orc 72.5° / 0.272  |  TORA 45.7° / 0.227**
- Oracle gives only +5.8 pp part_acc over denoiser-only — the verifier cannot recover from bad pose predictions when there are 15–20 parts to merge.
- TORA sits **between PF++ and GARF**: rotation 45° is roughly halfway between PF++'s ~73° and GARF's 20°. The flow-matching architecture is doing real work over PF++'s VQVAE+DDPM stack, but TORA does not match GARF's parallel-prediction precision on 10+ Voronoi-fractured bone parts.
- Translation tells the same story (GARF 0.06 < TORA 0.23 < PF++ 0.27).
- Interpretation: **the bottleneck is the denoiser's accuracy at high part counts, not the absence of a verifier**. GARF's specific flow-matching variant remains the strongest denoiser on this regime.

### 2. Bone Synthetic — Rib (6 PF++ comparable / 11 GARF & TORA, 5–32 parts)
- **GARF 33.2° / 0.068  |  PF++ den 72.1° / 0.234  |  PF++ orc 66.0° / 0.178  |  TORA 43.4° / 0.135**
- TORA evaluates all 11 samples (including the 22–32-part ones PF++ cannot handle); same regime as GARF.
- Same shape as pig: TORA between PF++ and GARF, but the gap to GARF is smaller (10° vs 25° on pig).
- TORA's part_acc reads as 87% but **this is misleading** — rib has many small interchangeable bone fragments, so Hungarian matching with a 0.01 Chamfer threshold lets TORA's predictions be re-assigned to whichever GT fragment they happen to land on. Use rotation_error as the comparable metric.

### 3. Fractura Real — Ceramics (8 samples, 3–12 parts)
- **GARF 38.6° / 0.106  |  PF++ den 65.8°  |  PF++ orc 18.3° / 0.035  |  TORA 54.0° / 0.243** ← PF++ oracle wins
- Massive +68.9 pp part_acc jump from denoiser-only to oracle: the verifier + oracle correspondences effectively solve this dataset.
- TORA does *worse* on rotation than GARF (54° vs 39°) and far worse than PF++-oracle (18°). On translation it's also the weakest of the unbiased denoisers.
- Why TORA underperforms here despite being a strong denoiser elsewhere: TORA's training distribution is "everyday" Breaking Bad pottery (similar to ceramics in shape) but TORA was *trained without* the explicit thin-shell regularisation that thinwalled gets. Real archaeological ceramic fragments have **thin curved sherds** with weak local geometric features; flow matching without correspondence guidance doesn't pin down rotation. PF++-oracle gets 100% correspondence precision and snaps poses to the answer.
- Interpretation: with perfect fracture-surface hints, PF++ can outperform any correspondence-free denoiser on low/medium-fragment real data. TORA confirms this — adding stochastic best-of-N alone doesn't substitute for the oracle.

### 4. Fractura Real — Egg (3 samples, 3–5 parts)
- **GARF 57.0° / 0.135  |  PF++ den 61.7°  |  PF++ orc 13.2° / 0.026  |  TORA 38.0° / 0.238** ← PF++ oracle wins; TORA second on rotation
- Oracle achieves perfect assembly on all 3 samples (up from 26% in both baselines).
- **TORA's 38° rotation error beats GARF's 57° on egg.** On 3 samples this is anecdotal but consistent with TORA's stronger small-piece-count behaviour seen on bones.
- Egg shells are smooth and featureless — both correspondence-free denoisers struggle (GARF, TORA) but TORA's stochastic flow at least gets one of three generations into a more plausible region.
- Interpretation: **egg shells are a correspondence-limited problem** — once told which points mate, assembly is trivial. TORA's slight edge over GARF here is real but cannot close the gap to oracle; the structural ceiling without correspondences is high.

### 5. Fractura Real — Bones (16 samples, 2–3 parts) — **TORA wins**
- **GARF 46.9° / 0.115  |  PF++ den 44.6°  |  PF++ orc 44.1° / 0.115  |  TORA 19.9° / 0.106**
- All three earlier frameworks plateau at ~44–47° rotation. **TORA cuts that to 20°** — the largest single-framework win on any subset in this comparison.
- This is the simplest regime (P=2 dominant; 13 of 16 samples), so the question is "place 1 non-reference part next to a fixed reference." That should be the easiest problem for any flow-matching denoiser, yet GARF and PF++ both fail at it ~half the time. TORA does not.
- The decisive failure mode prior analysis identified for PF++ on this subset (hemisphere flips from flat quaternion regression, 6/13 catastrophic CD>0.1) appears largely absent in TORA. TORA's `bone_syn_pig` translation/rotation behaviour is closer to GARF's failure shape (graceful) than PF++'s (catastrophic).
- The previously reported "geometric ceiling" claim — that the bones dataset has structural features which prevent any denoiser from doing better — is **falsified by TORA**. The ceiling was framework-specific, not geometric.
- Interpretation: TORA's flow-matching architecture (whatever the specific design choices — CKA self-distillation, encoder choice, training schedule) generalises to real-bone P=2 fractures substantially better than GARF or PF++. This is the single most surprising result of the cross-framework comparison.

---

## Key Factors

### 1. Denoiser bottleneck on bone geometry (not verifier chain length)
An earlier version of this analysis attributed bone underperformance to verifier sequential-merge-chain length. Per-sample breakdown refutes that: **bones fail even at the smallest fragment counts**, where the merge chain is trivial.

Per-sample part_acc grouped by fragment count P (oracle full pipeline):

| Dataset | P | n | mean part_acc |
|---|---|---|---|
| fractura_ceramics | 3–5 | 5 | **1.00** |
| fractura_egg | 3–5 | 3 | 0.78 |
| fractura_real_bones | **2** | 13 | **0.54** |
| fractura_real_bones | 3 | 3 | 0.33 |
| bone_syn_pig | **5** | 4 | **0.35** |
| bone_syn_rib | 5–6 | 3 | 0.37 |

The `fractura_real_bones` P=2 case is decisive: with the reference part anchored at GT and only one non-reference part to place, the merge chain is one step — no sequential-dependency trap is possible — yet the denoiser gets it right only ~54% of the time. Synthetic bone P=5 samples do worse than ceramics P=5 (35% vs 100%) despite more oracle correspondences (~400 vs ~84 per edge on average).

Correspondence density also rules out "not enough hints" — `bone_syn_pig` averages 6,488 total correspondences per sample vs 530 for ceramics and 116 for egg. Bones receive *more* oracle supervision and still underperform.

**Actual cause:** the denoiser's predicted pose is wrong for bone shapes regardless of chain length. Both models were trained on Breaking Bad "everyday" category (pottery/vessels — roughly cylindrical, convex shards). Bones are elongated, non-convex, and synthetic Voronoi fracture produces surfaces that do not match the training distribution. Per-sample Chamfer on failed bone cases is bimodal: some near-misses at CD ~0.02, others catastrophic at CD > 0.5 — consistent with an under-specified pose prior rather than verifier gating.

The 0/27 verifier convergence count on bones is a *symptom* of this, not the cause: the verifier refuses to merge because the denoiser's poses are genuinely wrong, not because a long chain rejected good poses.

GARF avoids verifier dependency entirely (parallel flow matching), but shares the same training-distribution limitation, which is why GARF also caps out around 20% part_acc on bone_syn_pig.

### 1b. Why GARF's denoiser is less brittle than PF++'s on bones

Side-by-side on the 13 P=2 `fractura_real_bones` samples (same samples, ref part anchored at GT, denoiser places one non-ref part):

| | GARF | PF++ oracle |
|---|---|---|
| part_acc | **61.5%** | 53.8% |
| mean Chamfer | **0.032** | 0.235 |
| near-exact (CD < 0.005) | **5/13** | 1/13 |
| catastrophic (CD > 0.1) | 2/13 | **6/13** |

Failure modes differ qualitatively. GARF misses gracefully (mostly CD 0.005–0.08 — right neighbourhood, imprecise). PF++ fails bimodally: 1 near-hit, 6 near-misses, and 6 catastrophic blow-ups with CD 0.25–0.87 consistent with hemisphere flips or far-off placements. This rules out "any denoiser at this training distribution would fail" — the failure is architecture-specific.

Likely causes, in order of suspected impact:

1. **VQVAE quantisation bottleneck** — PF++ feeds each fragment through a pre-trained VQVAE that compresses 1000 points → 25 discrete tokens × 64D. The codebook was learned on Breaking Bad pottery; bone fragments are out-of-distribution for the codebook, so the features the denoiser receives are weak before any pose regression happens. GARF's PointTransformerV3 encoder is continuous (per-point features, no codebook) and trained end-to-end with the denoiser, so the encoder adapts to whatever the pose head needs.
2. **Flat quaternion+translation regression** — PF++'s `DenoiserTransformer` outputs 7D via two MLPs (`mlp_out_trans`, `mlp_out_rot` at `puzzlefusion_plusplus/denoiser/model/modules/denoiser_transformer.py:88-103`), treating pose as a flat Euclidean vector. No SE(3) structure, no handling of the quaternion double-cover (q and −q represent the same rotation). The hemisphere-flip-looking catastrophic failures are consistent with this. GARF's `DenoiserFlowMatching` predicts a velocity field on SE(3) itself, so rotation and translation are coupled through the Lie group and sign ambiguity does not arise.
3. **DDPM vs flow-matching training signal** — PF++ trains the denoiser to predict noise (ε); gradient variance scales with timestep and the objective is inherently noisy. Flow matching regresses a straight-line velocity between noise and GT — smoother objective, better generalisation under distribution shift. Matters most for OOD shapes.
4. **Staged vs end-to-end training** — PF++ freezes the VQVAE before denoiser training, so the encoder is optimised for reconstruction, not pose. GARF backpropagates through the whole encoder + denoiser stack, so features are optimised for the pose task directly.

Both models still cap around 55–65% on P=2 bones. GARF is less brittle, not a solution; the training-distribution issue (Breaking Bad → bones) dominates for both. The clean fix is fine-tuning on bones.

### 1c. Why PF++ still wins on ceramics/egg despite a weaker denoiser

The architectural issues in §1b (VQVAE quantisation, flat quaternion regression, DDPM noise prediction, staged training) are **distribution-shift amplifiers** — they magnify error when the input is OOD. On in-distribution data their effect is small, and PF++'s pipeline has two compensating advantages that GARF gives up:

1. **PF++ uses oracle correspondences; GARF does not.**
   PF++'s verifier consumes the oracle fracture-surface correspondences each iteration. When the denoiser's pose is already close (in-distribution case), the verifier snaps it to the exact answer by minimising Chamfer distance between mated points — functionally an ICP refinement guided by perfect hints. GARF loads `fracture_surface_gt` and the adjacency graph but never passes them to the denoiser; on ceramics/egg this is leaving known information on the table.

2. **Ceramics and eggshell are in-distribution for Breaking Bad.**
   Breaking Bad's "everyday" training set is dominated by pottery/vessel shards. Both the VQVAE codebook and the DDPM denoiser have seen thousands of similar shapes, so the denoiser's initial pose estimate is already in the right neighbourhood. The brittleness in §1b only triggers when the denoiser is pushed OOD (bones). Convergence on ceramics/egg was 2–4 iterations vs 6 max for bones.

3. **Fragment count is small (P=3–5).**
   The verifier-merge pipeline is specifically designed to polish almost-correct poses into exactly-correct ones. With few fragments, a good starting pose, and 100% correspondence precision, this is the pipeline's best case. GARF's parallel prediction has no equivalent polishing stage.

Expressed as a split:

| | Denoiser starting pose | Correspondence polish | Result |
|---|---|---|---|
| PF++ on ceramics/egg | In-distribution → close | Used by verifier | **Wins** (90–100%) |
| PF++ on bones | OOD → often wrong | Cannot rescue a wrong start | Catastrophic |
| GARF on ceramics/egg | In-distribution → close | Ignored (dead code) | Decent (26–56%), loses the polish |
| GARF on bones | OOD → close-ish (§1b) | Ignored | Graceful degradation (58–78%) |

The honest framing is not "GARF did more right, why does PF++ win anywhere?" but: **GARF built a more robust denoiser; PF++ built a weaker denoiser plus a correspondence-guided refinement pipeline.** On easy in-distribution problems the refinement pipeline wins because oracle correspondences act as a ceiling-pushing post-process. On hard OOD problems the weak denoiser poisons the pipeline before refinement can help. The oracle ceiling flatters PF++ on ceramics/egg by substituting perfect hints for what Jigsaw would produce in practice (~70% accuracy).

### 2. Oracle matching dominates when the denoiser is in-distribution
On real ceramics/egg (3–5 parts), oracle matching is enormously valuable and PF++-oracle matches or beats GARF. These shapes are closest to the Breaking Bad training distribution, so the denoiser's baseline pose is already reasonable and oracle correspondences let the verifier snap it to the correct answer ("ICP-via-denoiser"). Ceramics/egg samples converged in 2–4 iterations on average.

Oracle matching provides a much smaller boost on bones because the denoiser's starting pose is too wrong for the verifier to confirm — the verifier's Chamfer-bin threshold rejects edges where the correspondences would, in principle, be informative. Oracle data is only as useful as the pose predictions it is applied to.

### 3. Architecture limit (20 parts)
PF++ cannot evaluate 22–32-part rib samples that GARF handles. GARF's advantage on rib is partially structural.

### 4. Normalisation was a bigger deal than suspected
On real data, the original denoiser-only run (per-part normalisation only) produced wildly inflated translation/CD metrics (19–30 / 400K–650K). After assembly-level normalisation, these dropped to 0.02–0.12 / 0.001–0.04. The verifier's hardcoded Chamfer-distance bins (calibrated for unit-scale data) require this normalisation to fire correctly.

### 5. Training domain (unchanged from prior analysis)
Both models are zero-shot (trained on Breaking Bad "everyday"). Neither was fine-tuned on bones/ceramics/egg.

---

## Caveats and Limitations

1. **Oracle matching is not achievable in practice.** Jigsaw's real-world correspondence accuracy is ≈70%, not 100%. PF++-oracle numbers represent the ceiling of what the PF++ pipeline could achieve, not a deployment estimate. A realistic number sits between denoiser-only and oracle.

2. **For the oracle run, PF++ has strictly more information than GARF.** Both models receive the same GT baseline (reference part pose), but PF++-oracle additionally receives perfect fracture-surface correspondences. GARF loads fracture labels and adjacency data but does not use them in its denoiser — they are dead code. Results where PF++-oracle exceeds GARF (ceramics, egg) reflect this correspondence information asymmetry, not superior pose prediction.

3. **Rib comparison remains imperfect** — GARF's rib result includes 5 samples with 22–32 parts that PF++ cannot evaluate. The 6-sample comparable subset remains the fair basis.

4. **Seed and randomisation** — GARF used seed 42; PF++ used seed 123 (its default). The PF++ DataLoader also applies random rotations to the input fragments at test time, so results have some stochastic variation.

5. **Sample size on egg (3 samples)** — 100% oracle accuracy is based on 3 objects and should be treated cautiously. The 26% baseline at the denoiser was based on the same 3 objects.

6. **Both models are zero-shot** on these datasets — out-of-distribution generalisation, not peak tuned performance.

---

## Cross-framework synthesis (with TORA included)

### Where each framework dominates

| Subset | Best framework | Best Rot° | Why |
|---|---|---|---|
| bone_syn_pig (5–20 P) | **GARF** | 20.5 | Strong parallel-flow denoiser; in-distribution-ish for vol training |
| bone_syn_rib (5–32 P) | **GARF** | 33.2 | Same; TORA close behind on the architecture-limited 6 PF++ can see |
| fractura_ceramics (3–12 P) | **PF++ oracle** | 18.3 | Pottery + perfect correspondences = best case for verifier loop |
| fractura_egg (3–5 P) | **PF++ oracle** | 13.2 | Featureless shells need correspondence hints to disambiguate |
| fractura_real_bones (2–3 P) | **TORA** | 19.9 | Small-P real fractures: TORA's stochastic flow beats both others by 2× |

### What TORA changes about the prior conclusions

1. **The "geometric ceiling on real bones" claim is wrong.** Prior analysis concluded GARF and PF++-oracle both hit a ~44° rotation ceiling because the bones dataset is geometrically intrinsically hard. TORA reaches 20° on the same data with the same anchor advantage and no oracle. The ceiling was an artifact of GARF's and PF++'s denoiser designs, not of the data.

2. **Stochastic best-of-N is doing real work.** TORA differs from GARF in two ways: a different flow-matching architecture, and stochastic generation × N=3 with best-of-N selection. From TORA's prior thinwalled study, N=3 over N=1 buys ~1.7° on rotation; on bones the swing is ~25°, much larger. Some of the gap is architectural, but a portion is the retry mechanic exploring more of the SE(3) basin.

3. **Best-denoiser-wins is a regime-dependent claim.** Earlier framing said "GARF built a more robust denoiser; PF++ built a weaker denoiser plus a refinement pipeline." Now: **on small-P real fractures, TORA's denoiser is more robust than GARF's**; on high-P synthetic fractures, GARF's is. There's no universal winner among correspondence-free denoisers.

4. **Correspondence-free denoisers split into two regimes.**
   - High-P synthetic Voronoi: GARF > TORA > PF++. Many parts → benefit of mass-parallel prediction with a strong shape prior dominates.
   - Low-P real fractures: TORA > GARF > PF++. Few parts → benefit of stochastic exploration + better-OOD architecture dominates.

5. **PF++-oracle still wins the correspondence-rich pottery regime.** TORA does not catch PF++-oracle on ceramics/egg. Without oracle hints, no correspondence-free model has matched the verifier ceiling on these datasets.

### Confidence-gated production protocol (extends across frameworks)

TORA's prior thin-walled work showed that std(rot_err) across N=3 generations is a reliable failure predictor — at `std < 0.5°`, 64 % of samples pass and 0 % hard-fail. This protocol should generalise to bones zero-shot since the agreement signal is dataset-agnostic. We have **not yet run** the agreement gate on these Fractura subsets; it is the most actionable next test.

### Honest summary

- **GARF is the strongest correspondence-free denoiser at high part counts.**
- **TORA is the strongest correspondence-free denoiser at low part counts on real fractures.**
- **PF++-oracle dominates wherever fracture correspondences are available** — but its oracle ceiling is not deployable without a Jigsaw model.
- **Real Fractura ceramics/egg cannot be solved without fracture correspondences by any of the three correspondence-free denoisers.** TORA's 38–54° rotation on these confirms the limitation.

---

---

## Follow-up batch (job 24343146) — A + B + C results

Three follow-up tests planned in the previous section have now been run as a single job.

### Test A — Best-of-N=10 sweep (anchor-fixed)

| Subset | n | parts | TORA N=1 | N=3 | N=5 | N=10 | Δ(10−1) |
|---|---|---|---|---|---|---|---|
| bone_syn_pig | 21 | 10.8 | 48.36 | 45.54 | 44.75 | **40.58** | −7.78 |
| bone_syn_rib | 11 | 17.7 | 50.85 | 43.80 | 41.42 | **39.40** | −11.45 |
| fractura_ceramics | 8 | 5.9 | 67.87 | 59.32 | 57.18 | **49.33** | −18.54 |
| fractura_egg | 3 | 4.0 | 53.92 | 41.52 | 40.92 | **36.82** | −17.10 |
| fractura_bones | 16 | 2.2 | 28.45 | 21.65 | 18.91 | **16.18** | −12.28 |

Three observations:
- **More retries help everywhere**, but absolute performance is still weak on synthetic bones (40°) and ceramics (49°). N=10 is not a substitute for in-distribution training or correspondences.
- **TORA's bones lead widens with N=10**: 16.2° vs the prior 47°/44° plateau for GARF/PF++-oracle. The architectural advantage is real and amplifies with retries.
- **The cliff is blunted but not moved.** Same shape as the thinwalled BoN sweep — N=1→5 captures most of the gain, N=5→10 is a smaller increment, and the hardest subsets retain the largest residual error.

### Test B — Agreement-gate analysis

This test answers: does TORA's "low std(rot_err) across N generations ⇒ trust the answer" protocol generalise to Fractura, as it did from thinwalled (64% coverage at std<0.5°) to bbad_artifact (54%)?

**Headline: no.** On Fractura, the agreement gate effectively does not work.

| Subset | std<0.5° (N=3 baseline) | std<5° | No-gate mean rot |
|---|---|---|---|
| bone_syn_pig | **0% coverage** | 28.6% / 38.5° / 83% fail@10° | 45.7° |
| bone_syn_rib | **9.1% / 5.8° / 0% fail** | 63.6% / 36.5° / 57% fail@10° | 43.4° |
| fractura_ceramics | **0% coverage** | 12.5% / 64.8° / 100% fail@10° | 54.0° |
| fractura_egg | **0% coverage** | 0% coverage | 38.0° |
| fractura_bones | **0% coverage** | 12.5% / 24.0° / 100% fail@10° | 19.9° |

Interpretation:
- **TORA is uncertain everywhere on Fractura.** Across all 59 samples (sum of n) only one subset (bone_syn_rib) has a single sample passing the strict `std < 0.5°` gate, and that sample has rot_err=5.77° — clean but anecdotal.
- **The gate signal that exists at `std < 5°` is unreliable.** Fail@10° rates of 57–100 % among "passing" samples mean the model agrees with itself on wrong answers. This was not the case on thinwalled (0 % hard-fail at std<0.5°) — there, agreement = correctness.
- **Mode collapse onto wrong answers.** Three generations agreeing at mean rot 64.8° on ceramics means TORA confidently produces consistent but completely wrong poses. The flow-matching denoiser has settled on a low-energy basin that isn't the right one. This is qualitatively different from thinwalled, where uncertainty correlated with error.
- **The failure to gate generalises across N=3 and N=10.** Increasing the ensemble does not give the gate more discrimination — at N=10, ceramics still has 0 % coverage at every threshold. The model's wrong-but-confident mode is stable across noise samples.
- **Implication for the tray run:** the agreement gate as a deployment-time confidence filter is not transferable to archaeological-style fractures. A different proxy is needed — e.g. pairwise Chamfer between predicted assemblies, or per-part pose covariance under input perturbations.

### Test C — Anchor-free vs anchor-fixed

| Subset | Rot° fixed | Rot° free | Δ | Trans (norm) fixed | Trans free |
|---|---|---|---|---|---|
| bone_syn_pig | 45.7 | 46.0 | +0.3 | 0.227 | 0.228 |
| bone_syn_rib | 43.4 | 39.4 | **−4.0** | 0.134 | 0.149 |
| fractura_ceramics | 54.0 | 58.7 | +4.7 | 0.243 | 0.290 |
| fractura_egg | 38.0 | 37.1 | −0.8 | 0.238 | 0.244 |
| fractura_bones | 19.9 | 24.2 | +4.3 | 0.106 | 0.120 |

Interpretation:
- **The bones win is mostly architectural, not anchor leakage.** Removing the anchor's GT pose hurts TORA on bones by ~4°, but anchor-free TORA at 24.2° is **still ≈20° below GARF (46.9°), PF++-den (44.6°), and PF++-orc (44.1°)**. The "TORA solves real bones at low P" claim survives the ablation.
- **Anchor advantage is small in general.** Effects of |Δ| < 5° everywhere; no subset is dominated by the anchor signal. This refutes the worst-case interpretation that TORA's wins were a measurement artifact of fixed-anchor evaluation.
- **rib improves under anchor-free (−4°).** Plausible explanation: largest-part anchor selection isn't optimal on elongated rib fragments where multiple long parts have similar size. Worth investigating further but a tangent.
- **Ceramics is the only subset where anchor-free clearly hurts (+4.7°).** Consistent with the "ceramics is correspondence-limited" framing — without the GT anchor, even the rough position prior is gone.

### Updated cross-framework rotation table (TORA at its best config = N=10 anchor-fixed)

| Subset | GARF | PF++ orc | TORA N=10 | TORA AF N=3 | Best |
|---|---|---|---|---|---|
| bone_syn_pig | **20.5** | 72.5 | 40.6 | 46.0 | GARF (×2 better than TORA) |
| bone_syn_rib | **33.2** | 66.0 | 39.4 | 39.4 | GARF, but TORA close |
| fractura_ceramics | 38.6 | **18.3** | 49.3 | 58.7 | PF++ orc (correspondence-driven) |
| fractura_egg | 57.0 | **13.2** | 36.8 | 37.1 | PF++ orc; TORA second |
| fractura_bones | 46.9 | 44.1 | **16.2** | 24.2 | TORA (~3× better than next) |

**TORA's bones win at N=10 is striking: 16.2° vs the prior 44–47° plateau** — a 2.7× improvement over both GARF and PF++-oracle, on a dataset where extra correspondences could not move the needle.

### Revised insights

1. **The bones architectural win is robust** — survives both retry-budget and anchor-free stress tests. TORA's flow-matching denoiser genuinely handles small-P real fractures better than GARF or PF++ across all tested configurations.
2. **Best-of-N is universally beneficial but bounded.** ~5–18° improvement per subset for N=1→10. Doesn't change which framework wins any given regime; it sharpens TORA's existing strengths.
3. **The agreement-gate protocol does not transfer to Fractura.** This was the most surprising result — it generalised cleanly from thinwalled to bbad_artifact (both pottery), but breaks on bone/eggshell/ceramic-archaeological fractures because the model's wrong-confidence basins are stable across noise samples. **For the tray run, do not rely on agreement-gating; design an alternative confidence proxy.**
4. **The high-P synthetic bones gap to GARF is structural.** N=10 + anchor-free combined still cannot bring TORA below 39° on rib or 41° on pig. GARF's parallel flow-matching architecture has a real advantage on Voronoi-fractured many-part inputs that no inference-time technique can close.
5. **Test E (fine-tune TORA on bone_synthetic train splits) is now the most valuable next step** — it tests whether the high-P gap is architectural (won't close) or training-distribution (will close after fine-tuning). The answer determines whether to invest in TORA or GARF for the tray pipeline.

---

## Next experiments — Fractura follow-ups (remaining)

Tests A, B, C completed in job 24343146 above. Remaining tests from the original plan:

Five tests, ordered by expected information value × cost.

### D. Train Jigsaw on bone_synthetic train splits → re-run PF++ with realistic correspondences *(expensive, decisive)*

**Hypothesis:** PF++-oracle's 100 %-correspondence assumption flatters PF++. The deployment-realistic number is between denoiser-only (~20–25 % part_acc) and oracle (~26–90 %). With real Jigsaw correspondences (~70 % precision in the wild) the answer for ceramics/egg is probably 60–80 %, still beating GARF on those datasets but not by 50 pp.

**Method:** Use bone_synthetic train splits (181 pig, 50 rib, 28+ for hip/leg/vertebra) to train Jigsaw from scratch; replace oracle matching data with Jigsaw output; re-run the existing `eval_complex_oracle.slurm`. Estimated effort: ~4–6 GPU-hours for Jigsaw training, < 1 GPU-hour for re-eval.

**Decision rule:** delta between Jigsaw and oracle on ceramics/egg is the deployment-realistic improvement of the verifier-loop architecture.

### E. Fine-tune TORA on bone_synthetic train splits *(expensive, high-value)*

**Hypothesis:** All three frameworks underperform on bone_synthetic because they were trained on Breaking Bad "everyday" pottery, not on Voronoi-fractured organic shapes. Fine-tuning TORA on the modest pig/rib train splits (~270 samples total) for a few epochs should close most of the OOD gap.

**Method:** Fine-tune `bbad_everyday_cka.ckpt` on `dataset_names=[hip, leg, pig, rib, vertebra]` train split with a low LR (1e-5) for ~50 epochs; evaluate on the same val splits used here.

**Decision rule:** if rot_err on pig/rib drops below GARF's zero-shot 20.5°/33.2°, TORA + small-data fine-tuning is the go-to recipe for new fracture domains. This is the test that matters for the eventual tray archaeological evaluation.

### Priority

A, B, C done. **E is now the highest-value remaining test**: TORA fine-tuned on bone_synthetic train splits will tell us whether the 20° GARF-vs-TORA gap on synthetic bones is architectural (irreducible) or training-distribution-driven (closeable with ~270 training samples). This is the test that matters for the eventual tray archaeological pipeline. **D** remains the most rigorous methodological contribution but is the most expensive.
