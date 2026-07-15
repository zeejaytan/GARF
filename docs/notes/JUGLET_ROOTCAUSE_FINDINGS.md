# Why GARF fails to assemble Juglet — root-cause findings

Final writeup for the root-cause investigation defined in
`JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md`. Question: why does GARF produce a wrong
assembly on the 9-piece Juglet archaeological scan, while it assembles synthetic
Breaking Bad and real Fractura ceramics well?

Date: 2026-07-09. Updated 2026-07-10 with Exp 7/8 results (see addendum at the
end): worn-rim erosion is contributing but NOT sufficient to explain the
failure, and rim-oversampling does NOT remedy it. Updated 2026-07-13 with
Exp 7b/9 results (second addendum): relief amplitude is ruled OUT as the
mechanism, and PF++ pseudo-GT label error is ruled out as an explanation of
the deficit — the pairwise perception failure stands, but its Juglet-specific
carrier is still unidentified.

---

## TL;DR

**GARF fails on Juglet because of a pairwise perception failure: its encoder
extracts no usable mating signal from Juglet's worn archaeological fracture
rims.** Given two sherds that genuinely fit together, GARF places them no better
than two that do not belong together — and ~3× worse than fresh-ceramic mates.
Because the pairwise signal is already absent in the easiest two-piece case, the
nine-piece joint assembly cannot succeed. The failure is **perceptual**, not a
9-piece joint-inference / global-search problem.

This is consistent with the earlier Exp 5 result (inference-time levers —
steps, iterations, anchor, SDE, schedule — barely move accuracy) and with the
worn-rim relief handicap (Juglet fracture relief ~37% below fresh ceramics).

**Remedy direction:** target the per-instance fracture signal — rim-focused
resampling and/or fine-tuning on worn/eroded breaks — not joint-inference knobs.

---

## What was ruled out first

Earlier phases of the investigation refuted the initial hypotheses (details and
job IDs in `JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md`):

- **Scale / global layout** — GARF's dataloader recenters, randomly rotates, and
  unit-scales every part, so global tray layout is invisible to the model
  (raw vs local02 give bit-identical output). Placement/viz effect, not a fit cause.
- **Scan noise (H2), mesh topology / resolution (H3)** — Juglet is the *cleanest,
  highest-resolution, fully watertight* set measured, better than the known-good
  data on every one of those axes.
- **Thin walls (H1)** — refuted by Exp 4c: real Fractura `ceramics` at Juglet-like
  wall thickness assemble at part_acc 0.92. Exp 4b's collapse was a marching-cubes
  remesh artifact.
- **Part count, global axisymmetry, contact-area fraction, aspect ratio** — no
  separation between GARF's success and failure cases.

No single aggregate mesh statistic predicts GARF success. That pointed the
investigation at *per-instance fracture-surface matchability*, tested directly
by the pairwise oracle (Exp 6).

---

## The decisive experiment: pairwise oracle (Exp 6 / 6b)

**Idea.** Decompose the 9-piece problem into all C(9,2)=36 two-piece
subproblems. If GARF cannot align even a *true mating pair* in isolation, the
failure is perceptual. If pairs align but the 9-piece run fails, the failure is
in joint inference / search.

### Step 1 — true-mate labels from the PF++ assembly

Juglet has no assembly ground truth, so true-mate labels came from the PF++
plausible assembly (`scripts/derive_pfpp_adjacency.py`): PF++'s renderer
`compute_final_transformation` was reproduced in numpy, applied to the stored
per-part clouds. The PF++ assembly compacts into a tight vessel and yields
**18/36 touching pairs** as true mates → `logs/diagnostics/juglet_adjacency/`.

### Step 2 — the original probe was invalid (why the control mattered)

Exp 6's first metric was cross-seed relative-pose *rotation dispersion*. It
showed Juglet true mates at 71.4° vs non-mates 67.1° — no separation. But a
positive control on known-good Fractura ceramics (Exp 6b, job 27000787:
pink_bowl, narrow_bottle2, narrow_bottle4, blue_pot; all part_acc ≥0.92) showed
those true mates *also* sit at ~71.8° dispersion. **The probe cannot separate
mates even on the success cases** — it is confounded by per-sherd rotational
symmetry (vessel sherds are near surfaces-of-revolution, so a 2-piece relative
pose has a symmetry degree of freedom). Same pitfall flagged in the Exp 1c
post-mortem. Summary: `logs/diagnostics/ctrl_pairs_20260709_144515/summary.md`.

### Step 3 — symmetry-invariant re-scoring (the clean result)

`scripts/pair_reference_chamfer.py` replaces dispersion with a *correctness*
metric: the assembled-shape **chamfer distance vs a reference pose**, normalised
by the pair diagonal, using correspondence-free multi-init ICP so the
unobservable symmetry DOF is not penalised. Reference = real GT for the control,
PF++ pseudo-GT for Juglet. Median over seeds 41/42/43.

| dataset | true-mate chamfer/diag | non-mate chamfer/diag |
|---|---|---|
| control ceramics (GARF works, part_acc ≥0.92) | **0.024** | 0.039 |
| Juglet (GARF fails) | **0.070** | 0.073 |

- **Metric validated on the control:** true mates (0.024, best ~0.008) align
  clearly tighter than non-mates (0.039) — the metric registers a real mate when
  one exists.
- **Juglet, two failures at once:**
  1. **No discrimination** — true mates (0.070) score the same as non-mates
     (0.073). GARF places a genuinely-mating pair no better than an unrelated one.
  2. **~3× worse absolute alignment** — Juglet true mates (0.070) are 3× worse
     than the control's true mates (0.024); even Juglet's single best true mate
     (0.039) is worse than the control's *median* true mate.

Outputs: `logs/diagnostics/pair_chamfer_control/summary.md`,
`logs/diagnostics/pair_chamfer_juglet/summary.md`.

---

## Conclusion

GARF cannot align even genuinely-mating Juglet sherd pairs. The mating signal is
already absent at the pairwise level, so the nine-piece assembly has nothing to
compose. The root cause is **perceptual** — the encoder does not extract a
distinctive, complementary rim signal from Juglet's worn archaeological fracture
surfaces — not a global search / joint-inference limitation.

---

## Recommended next steps

*(Both steps below were executed on 2026-07-10 — see the addendum. Step 1 came
back NOT confirmed, step 2's oversampling arm came back no-effect.)*

1. **Rim-erosion domain bridge (most decisive confirmation).** Add an `erode`
   transform to `scripts/domain_bridge.py` that smooths a working ceramic's
   fracture-surface relief down toward Juglet's worn level (GT pose preserved),
   then confirm its pairwise chamfer collapses from ~0.024 toward Juglet's
   ~0.070. If it does, worn-rim perception is nailed as the mechanism.
2. **Remedy, once confirmed:** rim-focused / fracture-region oversampling for
   thin-walled worn artifacts, and/or fine-tuning GARF on eroded archaeological
   breaks. Inference-lever tuning is not the lever (Exp 5).

---

## Addendum 2026-07-10 — Exp 7/8 results: the worn-rim story is incomplete

The pairwise **perception failure itself stands** (Exp 6/6b is unaffected), but
the two follow-ups narrow — and complicate — the mechanism:

### Exp 7 — rim-erosion bridge: NOT confirmed (job 27038450)

Eroding the true fracture bands of the four known-good ceramics
(`fracture_mesh_ops.erode_fracture_band`, strengths 0→1.0, GT preserved):

| erode strength | band relief_p90 | true-mate chamfer/diag (mean) | mate/non separation |
|---|---|---|---|
| 0.00 | — | 0.0229 | 1.61x |
| 0.50 | 0.201 | 0.0278 | 1.37x |
| 1.00 | 0.202 | 0.0317 | 1.24x |
| Juglet (measured) | 0.171 | **0.0700** | 1.04x |

Degradation is real and monotonic, but plateaus at less than half of Juglet's
failure level. Caveat: achieved relief saturated at ~0.20 and never reached
Juglet's 0.171 (mollifier stops carving once the band is smooth at its fixed
radius), so the bridge undershot calibration. Verdict: **worn-rim relief is a
contributing factor, not the mechanism.**
Output: `logs/diagnostics/exp7_erosion_20260710_141409/summary.md`.

### Exp 8 — rim-oversampling remedy: no effect (job 27038543)

Forcing 35% / 60% of each part's point budget onto the detected rim band:
Juglet true-mate median chamfer 0.0731 → 0.0705 → 0.0694 (best −5%, below the
pre-registered >10% bar) and no mate/non separation emerges (0.92–1.02x).
Control regression at 0.35 passes (no harm to working cases). Verdict: **the
deficit is not point-starvation at the rim — the encoder sees the rim and
still extracts no mating signal.**
Output: `logs/diagnostics/exp8_rim_20260710_142521/summary.md`.

### Revised remedy direction

- **Still true:** failure is pairwise-perceptual; inference levers and sampling
  allocation are not the answer.
- **Now open:** which Juglet-specific property carries the deficit. Untested
  candidates: fracture-band geometry *class* (weathered fillet profile vs sharp
  conchoidal, beyond relief amplitude), break-face texture/curvature
  statistics, and PF++ pseudo-GT label error inflating the measured 0.070.
- **Cheap next probes:** (a) rerun the erosion bridge with a larger mollify
  radius to break the relief plateau; (b) bound pseudo-GT error by re-scoring
  Juglet pairs against a relaxed best-fit reference.
  *(Both probes were executed on 2026-07-10 as Exp 7b and Exp 9 — see the
  second addendum below. (a) came back NOT confirmed even past the plateau;
  (b) ruled label error out.)*
- **Fine-tuning on eroded breaks remains the actionable lever**, with tempered
  expectations: Exp 7 implies eroded-fresh-ceramic is an imperfect proxy for
  the Juglet domain, so real/weathered training data may be needed.

---

## Addendum 2026-07-13 — Exp 7b/9 results: erosion mechanism ruled out, labels ruled out

Both "cheap next probes" from the first addendum ran on 2026-07-10 (jobs
27043960, 27043961). Net effect: the pairwise perception failure is confirmed
at full strength, but the *mechanism* is now narrower and still open —
worn-rim relief amplitude is definitively not it, and the PF++ pseudo-GT
labels are not inflating the measured deficit.

### Exp 7b — erosion bridge, larger mollify radius: NOT confirmed (job 27043960)

Reran the Exp 7 erosion bridge sweeping mollify radius 0 → 0.20 (piece-scale
units) to break the relief plateau that capped Exp 7 at ~0.20 relief:

| mollify radius / piece scale | relief_p90 (band) | true-mate chamfer mean/med | non-mate mean/med | mate/non separation |
|---|---|---|---|---|
| 0.00 | — | 0.0229 / 0.0240 | 0.0388 / 0.0386 | 1.61x |
| 0.05 | 0.202 | 0.0317 / 0.0330 | 0.0421 / 0.0410 | 1.24x |
| 0.10 | 0.336 | 0.0315 / 0.0315 | 0.0421 / 0.0404 | 1.28x |
| 0.15 | 0.407 | 0.0299 / 0.0305 | 0.0417 / 0.0413 | 1.35x |
| 0.20 | 0.422 | 0.0310 / 0.0312 | 0.0406 / 0.0404 | 1.30x |
| Juglet (measured) | 0.171 | **0.0700** | 0.0730 | 1.04x |

The plateau is broken — achieved band relief reaches 0.42, 2.5× past Juglet's
0.171 — yet true-mate chamfer saturates at ~0.031 and mate/non separation
never collapses below 1.24x. Exp 7's open caveat ("bridge undershot
calibration") is closed: even *over*-eroding fresh ceramics far beyond
Juglet's worn level does not reproduce the Juglet failure.
**Verdict: relief amplitude is NOT the mechanism.** The remaining candidates
are the ones the first addendum listed as untested: fracture-band geometry
*class* (weathered fillet profile vs sharp conchoidal) and break-face
texture/curvature statistics.
Output: `logs/diagnostics/exp7b_radius_20260710_164633/summary.md`.

### Exp 9 — PF++ pseudo-GT label-error bound: labels do NOT explain the deficit (job 27043961)

Band-constrained ICP refines each true-mate reference pose; the drift bounds
the label error and the re-scored chamfer is GARF's error against the refined
reference (`scripts/juglet_pseudo_gt_bound.py`):

| dataset | n mates | ref drift | orig true-mate chamfer | rescored | delta |
|---|---|---|---|---|---|
| control (real GT) | 18 | 0.0071 | 0.0242 | 0.0224 | −0.0018 |
| Juglet (PF++ pseudo-GT) | 18 | n/a (see caveat) | 0.0731 | 0.0727 | −0.0004 |

Procedure validated on the control (drift < 0.015, |rescored − orig| < 0.008).
Bias-corrected pseudo-GT label-error contribution: **+0.0021 of the 0.0731
Juglet deficit** — negligible. All decision gates passed.
**Verdict: PF++ pseudo-GT error does not explain the Juglet deficit; the
pairwise perception-failure conclusion stands at full strength.**

**Caveat on the Juglet arm's strength:** every Juglet pair reports drift = nan
with 0/0 band points and contact fraction 0.000 — the band-constrained ICP
found *no contact band at all* under the PF++ poses, so the Juglet references
were effectively re-scored unrefined. The control arm validates the procedure,
but the Juglet bound is correspondingly weaker than the control's. The
zero-contact finding is itself informative, though: even the PF++ "plausible"
assembly leaves Juglet mating faces without a touching band, consistent with
the perception-failure story rather than contradicting it.
Output: `logs/diagnostics/exp9_pseudogt_20260710_164147/summary.md`.

### Where this leaves the investigation

- **Closed:** relief amplitude (Exp 7 + 7b), rim point-starvation (Exp 8),
  pseudo-GT label error (Exp 9), and everything ruled out pre-Exp 6.
- **Open (mechanism):** fracture-band geometry class (weathered fillet vs
  sharp conchoidal profile, beyond amplitude) and break-face
  texture/curvature statistics — these now carry the full weight of the
  unexplained ~0.070 vs ~0.024 gap.
- **Actionable lever unchanged:** fine-tuning GARF on real weathered /
  archaeological breaks. Exp 7b strengthens the Exp 7 implication that
  synthetically eroded fresh ceramics are an inadequate proxy for the Juglet
  domain, so real weathered training data is likely required.

---

## Addendum 2026-07-13 — Exp 10: the mechanism, seen directly inside the encoder

Every prior experiment measured GARF as a black box (output chamfer) and *inferred*
a perception failure. Exp 10 (job 27188479, `scripts/fracseg_introspection.py`)
looks **inside** the frozen feature extractor for the first time. That extractor is
a `FracSeg` module whose `coarse_segmenter` head outputs, per point, P(fracture-
surface); the same backbone feeds the denoiser. So the head is a direct readout of
"does GARF recognize this surface as a mating/fracture surface?".

**Probe validated:** on labeled synthetic pig fractures the frozen segmenter fires
on the correct points — **AUC vs true fracture GT = 0.95**. (The relief-band label
turned out to be a poor proxy — on these objects surface relief anchors the *rough
original* surface, not the smooth fresh break, so P(fracture) anti-correlates with
relief. The trustworthy, label-free readout is the **fracture-response strength**:
what fraction of points the encoder calls fracture.)

**The finding — the encoder is blind to worn breaks:**

| arm | fracture-response (fired %) |
|---|---|
| synthetic fresh breaks (labeled, AUC 0.95) | **9.7%** |
| fresh real ceramics (GARF assembles, part_acc ≥0.92) | **3.4%** |
| **Juglet worn rims (GARF fails)** | **0.57%**  (0.17× control) |

GARF's fracture-aware encoder — pretrained on synthetic **fresh** breaks — assigns
near-zero fracture probability across Juglet's worn, smoothed archaeological rims.
It does not perceive them as fracture surfaces at all, so the identical backbone
hands the denoiser features with no mating cue. This is the **direct, feature-level
mechanism** behind the pairwise perception failure (Exp 6) and behind why the open
"fracture-band geometry class / break-face texture" candidates carried the deficit:
worn break faces lack the fresh-break micro-texture the encoder keys on. It is a
different lever than the denoiser knobs / rim oversampling that Exp 5/8 ruled out.

Output: `logs/diagnostics/exp10_fracseg_20260713_174759/summary.md`.

**Exp 10b (causal bridge, job 27188744):** eroding the true fracture bands of the
*labeled synthetic* breaks toward Juglet-like wear (labels preserved) drives the
encoder's response down monotonically — fired% **9.8% → 2.7%** and AUC vs true
fracture GT **0.91 → 0.66** at full strength. Surface wear *causes* the blindness.
It stops short of Juglet's 0.6% only because the mollifier undershoots Juglet's
wear level (same plateau flagged in Exp 7/7b). Output:
`logs/diagnostics/exp10b_erode_20260713_180057/`.

**Remedy (Exp 11, job 27189221) — direction confirmed, gap not yet closed.**
Fine-tuned the FracSeg feature extractor on labeled synthetic fractures
(bone_synthetic, 5 categories) with a new worn-break augmentation
(`data.frac_erode_prob`: mollify each object's true contact band at random wear
strength, labels preserved — `assembly/data/breaking_bad/fracture_erosion.py`),
60 epochs, lr 2e-5, erode prob 0.6, starting from `feature_extractor.ckpt`.
Re-running the Exp 10 probe on the fine-tuned encoder:

| arm | fired% before | fired% after |
|---|---|---|
| synthetic (probe, AUC vs GT) | 9.7% (0.95) | 9.6% (**0.965**) |
| fresh real ceramics | 3.4% | 6.0% |
| **Juglet worn rims** | **0.57%** | **0.97%** (+70%) |

The augmentation raised Juglet's fracture response by ~70% with no probe
degradation (synthetic AUC preserved) — the remedy direction is real. But the
control response rose proportionally, so the relative Juglet/control gap held
(~0.16×). This is exactly the limitation the earlier addenda predicted:
synthetically eroded *fresh* breaks are an imperfect proxy for real archaeological
wear (relief amplitude was already ruled out as the mechanism in Exp 7b, so
deeper erosion is not expected to close it; the operative factor is worn break-face
*texture/geometry class*, which needs real weathered training data to reproduce).
Artifacts: `scripts/finetune_frac_seg.py`, `exp11_finetune_frac_seg.slurm`,
`output/frac_seg_worn_20260713_181235/`.

**Exp 12 (job 27189756) — encoder-only swap does NOT assemble the Juglet.**
`scripts/merge_worn_encoder.py` swapped the worn encoder into GARF.ckpt (denoiser
untouched); reran the decisive pairwise chamfer oracle (3 seeds, PF++ pseudo-GT):

| encoder | true-mate chamfer/diag (mean/med) | non-mate (mean/med) | separation |
|---|---|---|---|
| frozen (baseline, Exp 6) | 0.070 | 0.073 | none |
| worn fine-tuned (Exp 12) | 0.0705 / 0.0676 | 0.0684 / 0.0658 | **none** |

The +70% perception gain did not translate into pairwise mating separation — true
mates are still indistinguishable from (marginally worse than) non-mates. Two
reasons: (1) Juglet's absolute fracture response is still ~6× below fresh ceramics,
so the added signal is small; (2) **confound** — swapping only the encoder feeds
the denoiser features from a distribution it was never trained on; encoder and
denoiser are coupled. The architecturally-correct remedy is to **co-adapt the
denoiser** to the worn encoder's features (LoRA fine-tune on worn-augmented data
starting from `GARF_worn_encoder.ckpt`), which Exp 13 tests. Output:
`logs/diagnostics/pair_chamfer_juglet_worn_20260713_182732/summary.md`.

**Exp 13 (job 27192118) — denoiser co-adaptation: a real but weak improvement.**
Full denoiser fine-tune (40 ep, lr 1e-5) from `GARF_worn_encoder.ckpt` on
worn-augmented bone_synthetic, so the denoiser learns to use the worn encoder's
features (`exp13_denoiser_coadapt.slurm`; fixed a torch>=2.6 `weights_only`
regression in `train.py`'s finetuning load). Juglet pairwise chamfer:

| model | true-mate (mean/med) | non-mate (mean/med) | best true mate |
|---|---|---|---|
| baseline frozen (Exp 6) | 0.070 / — | 0.073 / — | — |
| worn encoder only (Exp 12) | 0.0705 / 0.0676 | 0.0684 / 0.0658 | — |
| **+ co-adapted denoiser (Exp 13)** | **0.0645 / 0.0624** | 0.0671 / 0.0617 | **p0104 = 0.029** |

For the first time the true-mate **mean sits below** the non-mate mean, absolute
true-mate chamfer improved ~8% (0.070→0.0645), and one true pair (Piece01–Piece04)
now aligns at **control quality** (0.029, vs control true mates ~0.024). But the
effect is weak — medians are tied and most true mates are still interleaved with
non-mates (the 2nd–4th best pairs are non-mates), far from the control's clean
0.024-vs-0.039 separation. Output:
`logs/diagnostics/pair_chamfer_juglet_coadapt_20260713_194759/summary.md`.

**Bottom line.** The investigation is now complete end-to-end:
1. **Mechanism nailed (Exp 10):** GARF's frozen fracture-aware encoder is blind to
   Juglet's worn breaks (fires on 0.57% of points vs 3.4% on fresh ceramics),
   pretrained only on synthetic *fresh* breaks. This is the direct feature-level
   cause of the pairwise perception failure inferred by the black-box Exp 6.
2. **Cause is causal (Exp 10b):** eroding fresh breaks toward worn drives the
   encoder's response down (9.8%→2.7%).
3. **Remedy direction works but is data-limited (Exp 11–13):** worn-break
   augmentation + encoder/denoiser co-adaptation moves every metric the right way
   (Juglet fracture response +70%; true-mate chamfer −8%; first mate-over-nonmate
   preference; one pair now assembles at control quality) — but not enough to
   assemble the 9-piece vessel, because synthetic *bone*-break wear is an imperfect
   proxy for real archaeological *ceramic* wear.

**Remaining lever (the one the earlier addenda kept pointing to):** real weathered
training supervision — either annotated worn archaeological breaks, or
self-supervised adaptation on Juglet's own real worn geometry (pseudo fracture
labels from the validated relief-band detector) to inject the real worn-texture
signal the synthetic proxy lacks. Inference knobs, rim sampling, relief amplitude,
and pseudo-GT labels are all ruled out.

## Artifacts

- Adjacency / true mates: `scripts/derive_pfpp_adjacency.py` →
  `logs/diagnostics/juglet_adjacency/`
- Control pairs + GT labels: `scripts/build_control_pairs_hdf5.py`,
  `scripts/summarize_control_pairs.py` → `logs/diagnostics/ctrl_pairs_20260709_144515/`
- Symmetry-invariant re-scoring: `scripts/pair_reference_chamfer.py` →
  `logs/diagnostics/pair_chamfer_{control,juglet}/`
- Erosion bridge (Exp 7/7b): `scripts/fracture_mesh_ops.py`,
  `scripts/summarize_erosion_bridge.py`, `exp7_rim_erosion_bridge.slurm` →
  `logs/diagnostics/exp7_erosion_20260710_141409/`,
  `logs/diagnostics/exp7b_radius_20260710_164633/`
- Rim-oversampling remedy (Exp 8): `scripts/summarize_rim_remedy.py`,
  `exp8_rim_oversample_remedy.slurm` →
  `logs/diagnostics/exp8_rim_20260710_142521/`
- Pseudo-GT bound (Exp 9): `scripts/juglet_pseudo_gt_bound.py`,
  `exp9_pseudo_gt_bound.slurm` →
  `logs/diagnostics/exp9_pseudogt_20260710_164147/`
- Slurm: `exp6_pairwise_oracle.slurm` (Juglet), `exp6b_control_pairs.slurm` (control)
- Full experiment history + hypothesis ledger: `JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md`
