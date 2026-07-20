# Why PF++ forms a good Juglet shape — findings

Results for `PFPP_JUGLET_SUCCESS_EXPERIMENT_PLAN.md` (T0–T4). Executed
2026-07-20. All PF++ runs are denoiser-only (the configuration of the
original "semi-good" deploy result), checkpoint `everyday_epoch2000_bs64`,
baseline GARF is `GARF.ckpt` with the fixed deploy export.

Jobs: T0 27650662/27651352/27651601 · T4 27650827 + scores 27651278/27651602
· T1/T3 27651277 + scores 27651279 · T2 27651483 · fix pass 27651794.

---

## TL;DR

**PF++ succeeds on the Juglet by *curvature-coherent form composition*, not
by perceiving how the sherds mate.** Its VQVAE tokens encode each sherd's
coarse wall curvature — a signal abrasion cannot erase — and its
everyday-trained DDPM arranges any mutually-compatible curved sherds into a
category-typical vessel. The composition needs no fracture information at
all: nine *copies* of one sherd assemble into a *perfect* vessel (profile
1.000), *mirrored* sherds (mating geometrically impossible) assemble almost
as well as the real ones (0.928 vs 0.961), and the sherd-to-slot arrangement
is re-drawn ~at random every seed (127° cross-seed dispersion ≈ the 126.9°
random-rotation expectation) while the *form* stays excellent (0.89–0.98).

The difference from GARF is therefore a **channel difference, not a quality
difference**: GARF reads fracture-surface micro-texture (destroyed by wear —
the closed Exp 1–15 arc) and has no compositional form prior, so its output
on the Juglet is a compact *pile* (vessel-profile 0.719 ≈ random-pile 0.650).
PF++ reads coarse macro-curvature (wear-invariant by construction) through a
category form prior, so its output is a compact *vessel* (0.961). Neither
model recovers actual rim-to-rim mating (fine contact ≈ absent in both; the
Exp 9 zero-contact finding stands).

**Main factor:** the 25-token coarse representation + everyday vessel prior
— i.e. PF++'s "weakness" (the quantisation bottleneck that makes it lose on
fresh-break benchmarks, 21.5% PA vs GARF 56.3% on control ceramics) is
exactly what makes it degrade gracefully on worn archaeology: it never
depended on the signal that wear destroyed.

---

## T0 — the explanandum, quantified (and F5 bounded)

Layout-quality panel on the same 9 sherds, one shared frame
(`scripts/pfpp_layout_probes.py`; `logs/diagnostics/pfpp_t0_*`):

| arm | compactness | coarse pairs | fine pairs | SoR resid | profile frac |
|---|---|---|---|---|---|
| **PF++ deploy** | 0.885 | **18/36** | 8 | **0.0271** | **0.961** |
| GARF deploy (fixed export) | 0.944 | 17/36 | 10 | 0.0399 | 0.719 |
| random compact pile (n=5) | 1.396 | 7.8 | 4.8 | 0.0471 | 0.650 |
| identity-permuted PF++ slots (n=5) | 0.884 | 11.2 | 3.0 | 0.0458 | 0.652 |

- The visual gap is **not** a viz artifact (F5 bounded): GARF's corrected
  layout is just as *compact* as PF++'s and touches almost as many pairs —
  what separates them is **vessel-ness**: PF++ fits a surface of revolution
  over 96% of its points; GARF (0.719) is statistically a pile.
- "Semi-good result, at least a good shape" is exactly the right reading:
  form-level success without contact-level success.
- The permuted null already shows the form is not assignment-independent:
  putting the *wrong* sherd in each slot destroys vessel-ness (0.652).

## T4 — the form is driven by the tokens, not hallucinated by the prior

Corrupting the shape conditioning at inference (scale + anchor intact;
`ablation_mode` in `auto_aggl.py`; `logs/diagnostics/pfpp_t4_*`):

| arm | compactness | coarse pairs | SoR resid | profile frac |
|---|---|---|---|---|
| none (control, seed 123) | 0.885 | 18/36 | 0.0271 | **0.961** |
| shuffle tokens between sherds | 0.839 | 17/36 | 0.0489 | 0.655 |
| random codebook tokens | 0.796 | 12/36 | 0.0504 | 0.597 |
| zero latents | 0.798 | 11/36 | 0.0516 | 0.577 |

Anchor + scale + prior alone produce only a compact blob at pile-level
vessel-ness. The vessel form **requires the true per-sherd token content**.
So PF++ is not "dreaming a pot regardless of input" — it genuinely reads
sherd geometry. T1 pins down *what* it reads.

## T1a — what it reads is coarse shape compatibility, not identity or mating

Sherd-identity scrambles (`build_t1a_scrambles.py`;
`logs/diagnostics/pfpp_t1a_*`):

| arm | compactness | coarse pairs | SoR resid | profile frac |
|---|---|---|---|---|
| original | 0.885 | 18/36 | 0.0271 | 0.961 |
| foreign (4/9 sherds from other vessels, size-matched) | 1.009 | 13/36 | 0.0440 | 0.759 |
| **dup (9 copies of one body sherd)** | 0.966 | 15/36 | **0.0162** | **1.000** |
| **mirror (all sherds mirrored)** | 0.844 | 22/36 | 0.0302 | **0.928** |

Three arms, one story:

- **dup:** with *zero* mating structure available (nine identical copies),
  PF++ builds its most perfect vessel of the whole study. Real
  complementarity is not just unused — its absence *helps*, because copies
  of one curvature tile a surface of revolution perfectly.
- **mirror:** chirality-breaking makes true mating geometrically impossible
  while preserving curvature magnitudes — and costs almost nothing (0.961 →
  0.928). Fracture-level and orientation-sensitive cues contribute ~nothing.
- **foreign:** mixing curvature radii from different vessels is the one
  scramble that hurts (0.759) — the sherds must be *mutually compatible in
  coarse curvature* to merge into one clean form.

Together: the model perceives **per-sherd macro-curvature** and requires
**cross-sherd curvature coherence** — nothing sherd-pair-specific.

## T1b — the arrangement is re-drawn every run; only the form is stable

5-seed rerun of the unmodified deploy (`scripts/pfpp_stability.py`;
`logs/diagnostics/pfpp_t1b_*`):

- Per-seed form: compactness 0.81–0.89, vessel profile 0.89–0.98 — a good
  pot every time.
- Cross-seed relative-pose dispersion: **mean 127.3°** (median 131°) —
  statistically indistinguishable from the 126.9° expectation for *random*
  relative rotations. The sherd-to-slot assignment is not preserved between
  seeds at all.

This is the signature of interchangeable slots: PF++ does not know (or
need to know) *where each sherd goes*; it only needs each seed's arrangement
to be curvature-coherent. (Caveat: per-sherd rotational symmetry inflates
raw dispersion — Exp 6b measured ~72° for *correct* symmetric placements —
but 127° ≈ random is far above even that confound.)

## T3 — pairwise oracle: strong self-consistency; control calibration

Juglet pairs, 3 seeds, scored with the exact Exp 6 instrument
(`scripts/pfpp_pair_chamfer.py`; reference = PF++'s own 9-pc layout;
`logs/diagnostics/pfpp_t3_juglet_*`):

| model on juglet pairs | true-mate (mean/med) | non-mate (mean/med) | separation |
|---|---|---|---|
| GARF (Exp 6, PF++ pseudo-GT ref) | 0.070 / — | 0.073 / — | none |
| **PF++ (same ref = its own layout)** | **0.0268 / 0.0264** | 0.0464 / 0.0454 | **1.72×** |

Both pre-registered gates pass (median ≤ 0.045; separation ≥ 1.25×). Two
sherds alone reproduce the 9-piece relative placement for the pairs the
9-piece layout treats as adjacent — the pair behaviour is pair-specific,
stable across seeds, and consistent between the 2-piece and 9-piece
settings.

**Circularity caveat (important):** the juglet "true mates" *are* PF++'s own
touching pairs, so this arm measures self-consistency, not independent
mating perception. Combined with T1b (arrangement random across seeds) the
parsimonious reading is: adjacent-pair *shape* (two stacked curvature-
compatible sherds) is reproducible even though slot assignment is not.
The control arm — real GT mates on fresh ceramics (fix job 27651794) —
calibrates whether PF++ can separate *true* fracture adjacency at all.

**Control calibration (job 27651794) — PF++ has no independent mating
perception.** Same instrument, real GT reference, fresh control ceramics
(22 pairs: 16 true mates / 6 non-mates; small non-mate n because the pairs
HDF5 is mate-heavy — noted):

| model on control pairs (real GT) | true-mate (mean/med) | non-mate (mean/med) | separation |
|---|---|---|---|
| GARF (Exp 6b) | 0.024 / 0.024 | 0.039 / 0.039 | **1.61×** |
| **PF++** | 0.0362 / 0.0354 | 0.0429 / 0.0414 | **1.17× — gate FAIL** |

On fresh, in-distribution ceramics — where the fracture signal exists and
GARF separates cleanly — PF++ still cannot distinguish true mates from
non-mates at the pre-registered 1.25× bar, and its true-mate alignment
(0.0354) is looser than GARF's (0.024). **Verdict:** the juglet arm's 1.72×
"separation" is self-consistency of PF++'s composition habit (2-piece runs
re-produce the stacked-adjacent shapes its 9-piece layout also produces),
not perception of real fracture adjacency. This closes the last F2 escape
route: PF++'s channel contains no pair-mating signal on fresh breaks either.

## T2 — wear-invariance of the channel

The first T2 pass (job 27651483) was invalid for pose/latent comparison: the
May npz and the current HDF5 builds differ by a global frame/scale (the
juglet HDF5 was rebuilt during the watertight-remesh arc), which the
per-part metrics read as huge deltas. Two results survived it:

- **Layout-level invariance (valid, frame-internal):** the de-weathered
  juglet still assembles into a vessel — profile 0.904, 17/36 coarse pairs,
  SoR 0.0311 (vs 0.961/18/0.0271 fresh). The manipulation that moved GARF's
  fracture response 3.8× (Exp 14) barely moves PF++'s output form.

The fix pass (job 27651794) redoes the comparison on same-build geometry
with a resample-null control and a center-matched latent probe:

Fix-pass results (job 27651794, same-build geometry):

- **The strict per-part pose gates are unusable — instructively so.** The
  *resample null* (same geometry, independently resampled, same seed)
  already moves poses by mean 48° / 0.27 diag: per-part placement is not
  reproducible even for identical geometry — the pose-level face of T1b's
  interchangeable-slots result. (The wear-arm pose delta additionally shows
  a ~4.2-diag uniform offset consistent with a residual frame difference
  between the dewear and fresh HDF5 builds; superseded by the two clean
  readouts below.)
- **Output form is wear-invariant within resampling noise:** fresh rerun on
  the current build: profile 0.939, 19/36 coarse, SoR 0.0368; de-weathered:
  0.904 / 17/36 / 0.0311; original May build: 0.961 / 18/36 / 0.0271. The
  wear manipulation changes the layout no more than resampling does.
- **Representation is blind to the wear axis (T2b, center-matched probe):**
  same-part latent cosine under wear **0.811** vs resample-null **0.823**,
  both far above the different-sherd baseline 0.284. The de-weathering that
  moved GARF's fracture response **3.8×** (Exp 14) adds *nothing* beyond
  sampling noise to PF++'s VQVAE latents. (Discrete token-ID overlap is
  unusable as a gate — 7.6% even under pure resampling — the codebook
  assignment is unstable; the continuous latent is the valid readout.)
- **Interpenetration (finally mesh-valid): 0.0000** with 9/9 watertight
  pieces — the PF++ vessel is a physically plausible, non-overlapping shell
  arrangement, not a merged blob.

**F3 confirmed at both output and feature level.**

## T1c — seam continuity

Wall-thickness and local-curvature continuity across the 17 seams of the
PF++ layout vs 187 permuted-null seams (`pfpp_seam_continuity.py`, job
27652278):

| statistic | true seams (median) | permuted null (median) | one-sided p |
|---|---|---|---|
| thickness mismatch | 0.242 | 0.266 | 0.151 — FAIL |
| curvature mismatch | 0.400 | 0.591 | 0.177 — FAIL |

Directionally better than chance, not significantly so. PF++'s seams do not
carry statistically demonstrable macro-profile *matching* beyond what
curvature-coherent composition produces incidentally — consistent with T1b
(slots interchangeable) and T3-control (no mating perception). The
wall-profile channel, if it is to be used for pairing (Exp 16), must be
built explicitly; PF++ has not learned it.

---

## Synthesis — the main factor, and what differs from GARF

| question | answer | evidence |
|---|---|---|
| Is the good shape a viz artifact? | No — but the *compactness* part is shared; only vessel-ness separates the models | T0 |
| Is it unconditional prior hallucination? | No — destroying tokens collapses the form to a pile | T4 |
| Does it need *these* sherds? | Only their coarse curvature compatibility — copies are perfect, mirrors fine, foreign curvature breaks it | T1a |
| Does it know where each sherd goes? | No — arrangement ≈ random re-draw each seed; only the form is stable | T1b |
| Does it perceive pairwise mating? | No — control calibration fails (1.17×, vs GARF 1.61×); the juglet 1.72× is self-consistency | T3 |
| Is the channel wear-invariant? | Yes, at output level (form ≈ unchanged within resample noise) and feature level (wear ≈ resample in latent space; GARF shifted 3.8×) | T2 |

**Main factor.** PF++'s success is carried by **(i)** a representation — 25
FPS-centred VQ tokens — that keeps only coarse macro-curvature, which worn
archaeological abrasion does not touch, and **(ii)** an everyday-pottery
DDPM prior that composes curvature-compatible pieces into a category-typical
vessel form. The success is *form completion over wear-invariant coarse
features*: hypothesis F1 in its content-gated variant ("prior composes,
tokens gate plausibility" — exactly the middle row of the plan's decision
matrix), with the everyday/jug category match as the enabling condition.

**Why GARF differs.** Same training data, same anchor/scale/steps — but
GARF's channel is dense fracture micro-texture (pretrained by fracture
segmentation) feeding an input-driven flow matcher with no compositional
category prior. On worn sherds that channel carries nothing (Exp 6/10/15),
and with nothing to read, GARF's output is an amorphous compact pile. PF++'s
channel never contained the destroyed information in the first place — so
its behaviour on the Juglet is the same as its behaviour everywhere:
plausible vessel, loose contact. On *fresh* benchmarks the same trade
reverses the ranking (PF++ denoiser-only 21.5% PA vs GARF 56.3% on control
ceramics): micro-texture wins when it exists; the form prior wins when it
doesn't.

**What PF++'s layout is, operationally.** A category-prior readout with
per-sherd curvature slotting — trustworthy at the level of "these sherds
form a vessel of this size/shape", untrustworthy at the level of "sherd A
mates sherd B along this rim". Consistent with how it has been used since
Exp 6: as a *pseudo*-GT for coarse adjacency, never for contact (Exp 9
found zero contact bands under its poses).

## Implications for the reconstruction project

1. **Neither model perceives worn-rim mating.** The juglet remains unsolved
   at contact level by both frameworks; PF++'s output should be treated as
   a coarse initialisation, not an assembly.
2. **The hybrid path (proposed Exp 16) is now doubly motivated:** PF++
   (or its principle — coarse curvature + form prior) supplies global slot
   structure; break-boundary curve registration supplies pairing and
   contact; fracture features refine where signal exists.
3. **The T0 panel generalises** into a standing no-GT QA metric for
   archaeological deploys (vessel-ness + contact + interpenetration),
   replacing eyeball judgement of render videos.
4. **A "PF++-style" robustness lesson for GARF:** a coarse-curvature token
   stream (or downsampled PTv3 features) added to GARF's conditioning could
   give it the same graceful worn-domain fallback without giving up its
   fresh-break precision.

## Artifacts

- Plan: `PFPP_JUGLET_SUCCESS_EXPERIMENT_PLAN.md`
- Panel/nulls: `GARF/scripts/pfpp_layout_probes.py` →
  `logs/diagnostics/pfpp_t0_*`, `pfpp_t4_*`, `pfpp_t1a_*`
- Ablations: `puzzlefusion-plusplus/puzzlefusion_plusplus/auto_aggl.py`
  (`ablation_mode`), `scripts/eval_t4_ablations.slurm`
- Scrambles: `puzzlefusion-plusplus/scripts/build_t1a_scrambles.py`
- Stability: `GARF/scripts/pfpp_stability.py` → `pfpp_t1b_*`
- Pair oracle: `puzzlefusion-plusplus/scripts/build_pair_npz.py`,
  `GARF/scripts/pfpp_pair_chamfer.py` → `pfpp_t3_juglet_*`, `pfpp_t3_control_*`
- Wear: `puzzlefusion-plusplus/scripts/eval_t2_wear.slurm`,
  `eval_t2t3_fix.slurm`, `scripts/vqvae_latent_probe.py`,
  `GARF/scripts/pfpp_pose_delta.py` → `pfpp_t2a_*`, `pfpp_t2fix_*`
- Seams: `GARF/scripts/pfpp_seam_continuity.py` → `pfpp_t1c_*`
