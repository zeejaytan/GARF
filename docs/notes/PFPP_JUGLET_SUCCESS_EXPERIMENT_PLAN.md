# Why does PF++ form a good Juglet shape? — Success-factor experiment plan

Created 2026-07-19. Mirror-image of `JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md`
(which asked why GARF *fails* on Juglet; closed 2026-07-19 with
`JUGLET_ROOTCAUSE_FINDINGS.md`). This plan asks the complementary question:

> **Why does PuzzleFusion++ produce a semi-good result — at least a coherent
> vessel shape — on the 9-piece Juglet, when GARF cannot? What is the main
> factor behind its success?**

Everything here is inference-only (no training). Each GPU experiment is
minutes on one A100. Per workspace rules, Slurm jobs are submitted only after
user OK.

---

## 0. What probing the existing evidence already establishes

These facts are settled by prior runs and code reading — not to be re-tested:

1. **The PF++ Juglet deploy was denoiser-only.**
   `puzzlefusion-plusplus/scripts/eval_juglet_deploy.slurm` runs `test.py`
   with `verifier.max_iters=1`: no verifier, no merge loop, no Jigsaw/oracle
   correspondences. The entire result comes from the VQVAE encoder + DDPM
   denoiser + anchor. Any explanation invoking PF++'s verify/merge pipeline
   is ruled out from the start.

2. **The shared-inputs list is longer than expected.** Both frameworks:
   trained on Breaking Bad `everyday` only; anchor the largest fragment;
   per-part recenter + unit-normalise; **both condition on per-part scale**
   (PF++: NeRF-embedded `n^f`; GARF: NeRF-embedded scale in
   `denoiser_transformer.py:156-177`); both run 20 denoising steps. So scale
   conditioning, anchoring, step count, and training category are all
   *not* differentiators.

3. **The two real differentiators are representation and objective.**
   - *Representation:* GARF = dense per-point PointTransformerV3 features,
     pretrained with a fracture-**segmentation** objective (micro-texture
     channel). PF++ = 25 FPS-centred VQ tokens per sherd, 64D, codebook
     trained for *reconstruction* on everyday pottery (coarse macro-geometry
     channel; 1000 pts → 25 tokens discards micro-texture by construction).
   - *Objective/dynamics:* GARF = flow matching (velocity field, strongly
     input-driven); PF++ = DDPM ε-prediction with piecewise schedule
     (mode-seeking sampler with a strong learned prior).

4. **PF++'s Juglet layout is form-correct but contact-empty.**
   `derive_pfpp_adjacency.py`: the assembly compacts into a tight vessel,
   18/36 pairs touch at coarse thresholds. But Exp 9's band-constrained ICP
   found **zero contact band** (contact fraction 0.000, 0/0 band points) for
   every mated pair under the PF++ poses. PF++ did *not* recover rim-to-rim
   mating; it recovered a plausible **arrangement**. "Semi-good, at least a
   good shape" is exactly right and now has a mechanical reading: form-level
   success, contact-level failure.

5. **The crossover that reframes the whole question.** On *fresh* Fractura
   ceramics (control set GARF assembles at part_acc ≥0.92 in the pair study;
   56.3% PA / 38.6° in the 2026-04 benchmark), PF++ **denoiser-only** — the
   exact configuration that "succeeded" on Juglet — scores **21.5% PA /
   65.8° rotation**, i.e. far *worse* than GARF. PF++ denoiser-only is not a
   better ceramics solver that finally met its natural prey. The more
   parsimonious reading: **PF++'s denoiser emits a compact, vessel-shaped
   layout more or less regardless of whether it has solved the problem** —
   its failure mode *looks like* a pot — while GARF's failure mode looks
   like noise. On Juglet, where the fracture channel is dead for both
   (Exp 6/10/15), the model whose fallback is "a plausible vessel" appears
   to succeed.

6. **Residual visualization asymmetry is bounded but nonzero.**
   `JUGLET_DEPLOY_INFERENCE_ANALYSIS.md` (2026-05-20) showed part of the
   original visual gap was pipeline: PF++ renders poses back onto real OBJs
   in mesh space; GARF's deploy GLB was then buggy (post-hoc `anchor_free`
   GT snap — since fixed). The adjacency work proves PF++'s compaction is
   real in pose space, but a like-for-like *quantitative* form comparison
   between PF++'s and (fixed) GARF's final layouts has never been done.

7. **Category match.** The Juglet is literally a jug; `everyday` is
   dominated by jugs/bottles/vases/bowls — near-ideal for a category-level
   vessel prior. The Fractura control ceramics are also vessels, so category
   alone doesn't separate the frameworks (see fact 5) — but it is the
   enabling condition for a prior-driven layout to look *right* on Juglet.

---

## Hypotheses (ranked)

What carries PF++'s good Juglet shape?

- **F1 — Prior-driven form composition (front-runner).** The DDPM denoiser,
  sampling under a strong everyday-vessel prior, arranges the sherds into
  the learned category form. Per-sherd input mostly tells it *what kind of
  slot* each sherd fits (size via scale conditioning, shell curvature via
  coarse tokens); pairwise mating information contributes ~nothing. Success
  = prior completion, not perception. Predicts: layout survives sherd
  scrambling; no pairwise mate discrimination; wear-invariant.

- **F2 — Genuine coarse-geometry perception.** PF++ actually reads
  sherd-specific macro cues (wall curvature radius, thickness profile, rim
  arcs) and places sherds in *their* correct relative arrangement — the
  macro channel Exp 15's conclusion says survives wear (boundary curves,
  wall profiles). Predicts: placement is sherd-specific (scramble hurts;
  seeds agree; seam continuity beats chance), possibly weak pairwise
  discrimination.

- **F3 — Wear-invariant representation (robustness explanation, orthogonal
  to F1/F2).** The 25-token bottleneck never encoded fresh-break
  micro-texture, so worn sherds are in-distribution for PF++'s encoder,
  while GARF's fracture-pretrained features collapse (Exp 10: 6× response
  drop). Explains *why PF++'s channel survived*, whichever of F1/F2 it is.

- **F4 — Objective/dynamics (DDPM prior completion vs flow matching).**
  The sampler, not the representation, produces the graceful fallback.
  Partially entangled with F1; testable only by feature-side ablations
  (T4), since retraining swapped architectures is out of scope.

- **F5 — Residual visualization flattery.** Some of the perceived gap is
  still frames/rendering. Must be bounded quantitatively before crediting
  the model (T0).

F1+F3 together form the working "main factor" candidate:
**a category-level vessel-form prior operating on wear-invariant coarse
features — success by form composition, not by fracture perception.**

---

## Fixed assets (all verified present on Spartan, 2026-07-19)

- PF++ deploy output: `Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0/`
  (`gt.npy`, `init_pose.npy`, `predict_*.npy`, `mesh_file_path.txt`)
- PF++ input npz: `Puzzlefusion/data/pc_data/juglet_deploy/val/00000.npz`
- PF++ npz builder: `Puzzlefusion/convert_hdf5_to_npz.py`; launcher template
  `scripts/eval_juglet_deploy.slurm` (add a `seed=` override for T1b)
- Juglet meshes: `Dataset/Juglet_anchor_centered/Piece0[1-9].obj`
- Pose-chain reproduction (numpy `compute_final_transformation`):
  `GARF/scripts/derive_pfpp_adjacency.py` → `logs/diagnostics/juglet_adjacency/`
- Symmetry-invariant pair scorer: `GARF/scripts/pair_reference_chamfer.py`
  (+ control pair sets from Exp 6b, `build_control_pairs_hdf5.py`,
  `build_juglet_pairs_hdf5.py`)
- Wear machinery: `GARF/scripts/fracture_mesh_ops.py`
  (`erode_fracture_band`, `sharpen_fracture_band_solo`) + Exp 14 de-weathered
  Juglet variants
- No-GT layout probes: `GARF/scripts/no_gt_probes.py` (contact, stability)
- GARF fixed-deploy result for comparison: rerun of
  `infer_juglet_deploy.slurm` post-fix (or reuse latest deploy GLB)

---

## Experiments

### T0 — Quantify the explanandum: like-for-like layout quality panel

**Purpose:** turn "PF++ looks semi-good, GARF doesn't" into numbers in one
shared frame; bound F5.

**Method.** New `GARF/scripts/pfpp_layout_probes.py` (CPU-only): apply each
framework's final poses to the same 9 OBJs (PF++ via the
`derive_pfpp_adjacency` chain; GARF via its fixed `T_pred @ T_aug⁻¹` deploy
export), plus two baselines: (i) random compact layout (parts piled at
random orientations in a vessel-sized bbox), (ii) the PF++ layout with part
identities randomly permuted between slots. Report per layout:

1. **Compactness** — max pairwise centroid distance / mean part diagonal.
2. **Coarse adjacency** — touching pairs at the `derive_pfpp_adjacency`
   thresholds (expect PF++ ≈ 18/36).
3. **Fine contact** — Exp 9 band-contact fraction (expect ≈ 0 for all).
4. **Interpenetration** — fraction of part-i points inside part-j meshes
   (winding number); a pile scores high, a shell arrangement low.
5. **Vessel-ness** — fit a surface of revolution (axis by optimisation);
   RMS radial residual / assembly diagonal, plus fraction of surface points
   within 5% of the fitted profile.

**Pre-registered expectations / gates.**
- Explanandum is real iff PF++ beats fixed-GARF *and* the random-compact
  baseline on vessel-ness and interpenetration (not just compactness).
- F5 bounded: if fixed-GARF's layout scores within noise of PF++ on all
  five, the "success" was viz after all and the investigation pivots back
  to pipeline (unlikely given the matched-diagnostic spread ratios, but
  must be on record).
- The identity-permuted baseline (6) previews T1: if it scores ≈ PF++, form
  metrics cannot distinguish arrangement correctness — noted for T1's
  design, which then carries the discrimination load.

**Cost:** CPU only, runnable on existing saved outputs. **Blocks:** nothing;
everything else can run in parallel but T0 frames their readouts.

### T1 — Sherd-identity dependence: is the layout about *these* sherds?

**Purpose:** discriminate F1 (prior hallucination) vs F2 (genuine coarse
perception). The decisive question: does PF++'s output depend on the
specific sherds, or only on "nine vessel-ish fragments of these sizes"?

**Method.** Three arms, all denoiser-only, all scored with the T0 panel plus
per-sherd placement agreement:

- **T1a — scramble.** Build Juglet variants via a new
  `build_juglet_scramble_sets.py`: (i) replace 4 of 9 sherds with sherds
  from a *different* vessel (control ceramics HDF5, matched approximate
  size); (ii) duplicate one sherd 9× (same scale distribution, zero mating
  structure); (iii) mirror every sherd (chirality-breaking: mating becomes
  geometrically impossible, coarse shape statistics preserved). Convert
  with `convert_hdf5_to_npz.py`, run PF++.
  *Gate:* if vessel-ness and compactness drop <20% from baseline in (i) and
  (ii) → the form does not depend on sherd identity → **F1**. If (iii)
  degrades but (i)/(ii) don't, the model reads chirality-level macro cues —
  partial F2.
- **T1b — seed consistency.** ≥5 seeds of the unmodified deploy run
  (`test.py` seed override; PF++ default is 123). Feed final poses to the
  `no_gt_probes.py` stability metric (per-pair relative-pose dispersion).
  *Discrimination:* F2 predicts low per-pair dispersion (same specific
  arrangement each time). F1 predicts high per-pair dispersion with low
  *form-metric* dispersion (a different plausible pot each time —
  interchangeable slots). This probe is symmetry-aware territory: report
  both raw dispersion and dispersion modulo the fitted vessel axis (the
  Exp 6 post-mortem pitfall — sherds of a revolution surface have a free
  azimuth DOF).
- **T1c — seam continuity.** For each of the 18 touching pairs in the PF++
  layout, measure wall-thickness and principal-curvature continuity across
  the seam vs the distribution over random touching arrangements (from the
  T0 permuted baseline). *Gate:* continuity better than the permuted null
  at p<0.05 (rank test over pairs) → the layout encodes real macro
  matching → **F2** evidence, and directly supports the Exp 16
  boundary-curve/wall-profile channel proposed in the GARF findings.

**Cost:** T1a ~3 runs + builders; T1b 5 runs; T1c CPU. ≈1–2 GPU-hours total.

### T2 — Wear-invariance of the PF++ channel (the F3 test)

**Purpose:** verify the robustness half of the story on the PF++ side, with
the same manipulations that indicted GARF's encoder.

**Method.**
- **T2a — output invariance.** Run PF++ denoiser-only on: original Juglet;
  Exp 14 de-weathered Juglet (strengths 1.0/2.0); eroded control ceramics
  (Exp 7 assets, strength 1.0). Compare final layouts (per-part pose deltas
  + T0 panel). *Gate:* mean per-part rotation delta <10° and translation
  delta <0.05 (normalised units) across wear levels → the channel ignores
  the micro-texture GARF depends on.
- **T2b — representation invariance (mirror of Exp 10, PF++ side).** New
  `GARF/scripts/pfpp_latent_probe.py`: encode fresh vs eroded versions of
  the same control sherds with the frozen VQVAE; report token-ID overlap
  (fraction of the 25×4 codebook indices unchanged) and continuous-latent
  cosine distance; contrast with the known 6× PTv3 fracture-response
  collapse. *Gate:* token overlap ≥80% / cosine distance an order of
  magnitude below the PTv3 shift → **F3 confirmed at feature level.**

**Cost:** T2a ~4 runs (minutes each); T2b CPU+GPU encode-only, trivial.

### T3 — Pairwise mating oracle for PF++ (the decisive instrument, reused)

**Purpose:** the single most informative probe. Exp 6 indicted GARF by
showing zero mate/non-mate separation on the 36 Juglet pairs. Running the
*identical* harness on PF++ answers, with the same instrument, whether
PF++'s 9-piece success involves any pairwise perception at all.

**Method.** Build PF++-format pair npz sets for all 36 Juglet pairs and the
Exp 6b control pairs (new `puzzlefusion-plusplus/scripts/build_pair_npz.py`,
mirroring `build_juglet_pairs_hdf5.py`; anchor = larger sherd). Run
denoiser-only, 3 seeds. Score with `pair_reference_chamfer.py` (references:
PF++ pseudo-GT for Juglet — self-consistency is acceptable here since the
reference *is* PF++'s own 9-piece layout; real GT for control).

**Pre-registered gates (same as the GARF arc):** separation ≥1.25× and
true-mate median ≤0.045.

**Discriminating outcomes:**
- **No separation on Juglet (predicted by F1/F3):** PF++ has no pairwise
  mating perception on worn sherds either — its 9-piece success is *joint
  form composition*, something that only exists at the ensemble level. Main
  factor = prior, confirmed from a second angle.
- **Separation on Juglet:** PF++ genuinely extracts worn-sherd mating
  signal from coarse geometry that GARF's micro-texture channel misses —
  a much stronger F2 result, and immediately actionable (the macro channel
  is learnable; hybridise into GARF per Exp 16).
- Control arm calibrates the instrument (does PF++ separate mates on fresh
  ceramics at all? Given the 21.5% PA crossover, possibly only weakly —
  itself informative: it would show even PF++'s *control* behaviour is
  form-driven).

**Cost:** ~36+30 pair runs × 3 seeds, each seconds–minutes; ≈1–2 GPU-hours.

### T4 — Prior-dominance ablation (how much does the input matter at all?)

**Purpose:** directly measure how much of the layout is prior vs input;
cheapest possible F1-vs-F4 evidence without retraining.

**Method.** Patch PF++ inference (flag in `auto_aggl.py`/`test.py` fork) to
corrupt the shape conditioning while keeping scale + anchor intact:
(i) shuffle the 25 VQ tokens *between* sherds; (ii) replace all tokens with
random codebook entries; (iii) zero the shape embedding. Run each on the
Juglet set, score with the T0 panel.

**Discrimination.**
- Compact vessel-like form survives (i)/(ii) → the arrangement is nearly
  unconditional given scales + anchor → **F1 in its strongest form**: the
  everyday prior plus 9 scale tokens is sufficient to produce "a jug".
- Form collapses under (ii) but survives (i) → per-sherd content matters
  but identity assignment doesn't — intermediate: prior composes, tokens
  gate plausibility.
- Form collapses under all → the coarse tokens genuinely drive placement
  (F2 support; combine with T1/T3 to decide whether *correctly*).

**Cost:** 3 runs + small patch. Minutes of GPU.

---

## Decision matrix → "main factor" verdict

| T1a scramble | T1b seeds | T1c seams | T3 pairs | T2 wear | Verdict |
|---|---|---|---|---|---|
| form survives | slot-swapping | ≈ null | no sep | invariant | **F1+F3: category vessel prior over wear-invariant coarse features; success = form completion, not perception.** GARF differs because its channel (fracture micro-texture) was destroyed while PF++'s (macro form) never needed what wear removed. |
| form degrades | consistent | > null | sep | invariant | **F2+F3: PF++ genuinely perceives worn-sherd macro mating cues.** Actionable: port the macro channel (boundary curves / wall profiles / coarse tokens) into GARF (Exp 16 hybrid); PF++ layout becomes a trustworthy pose init. |
| mixed | mixed | > null | no sep | invariant | **F1 composition + weak F2 content-gating:** prior arranges, sherd macro-geometry selects slots. Still implies PF++ layout is a *class*-level prior, usable as coarse init only. |
| — | — | — | — | *not* invariant | Surprise: PF++ also reads micro-texture; re-examine F4 (objective difference) as the differentiator. |
| T0 shows GARF ≈ PF++ | | | | | **F5:** the residual gap was visualization; publish the corrected comparison and close. |

Whatever branch wins, the deliverable claim has the same shape as the GARF
findings: *which channel carried the result, shown by manipulation of that
channel* — not correlation.

## Consequences for the project

- If F1 (expected): PF++'s Juglet layout is a **category prior readout**.
  It remains legitimately useful as (a) the pseudo-GT/adjacency source it
  already is, (b) a coarse pose initialiser for a contact-refinement stage
  (curve-based registration, Exp 16) — but it should *not* be read as
  evidence that the sherds were "recognised". The honest system framing
  becomes: PF++ supplies the form prior, a macro-geometry matcher supplies
  pairing, fracture features refine where signal exists.
- If F2: the macro channel is real and learned — strongest possible
  motivation for Exp 16 and for hybrid GARF+coarse-token architectures.
- Either way T0's panel becomes the standing no-GT QA metric for
  archaeological deploys (replacing eyeballing render videos).

## Execution order & budget

1. **T0** (CPU, no job needed — can run on saved outputs immediately)
2. **T4** (cheapest GPU discriminator, 3 short runs)
3. **T1a/T1b** (scramble + seeds, ~2 GPU-hours)
4. **T3** (pair oracle, ~2 GPU-hours)
5. **T2** (wear invariance, ~1 GPU-hour)
6. T1c (CPU, uses T0 machinery)

Total ≈ 5–6 GPU-hours across ~6 Slurm submissions. All new scripts follow
existing conventions (`GARF/scripts/*` for analysis, PF++ fork for builders
and inference patches; outputs under `GARF/logs/diagnostics/pfpp_*`).
