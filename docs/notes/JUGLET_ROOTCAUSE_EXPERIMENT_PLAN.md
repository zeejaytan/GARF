# Why does GARF fail on Juglet? — Root-cause experiment plan

Created 2026-05-29. Goal: find the *mechanism* by which GARF produces a wrong
assembly on the 9-piece Juglet scan (shape doesn't close, fractured edges don't
align), given that GARF assembles synthetic BB/Fractura (≤6 pieces) well.

This is a **root-cause** investigation, not an accuracy benchmark. Juglet has no
assembly GT, so every metric below is a **no-GT proxy**.

---

## 0. What the code already tells us (established, not to be re-tested)

From reading the GARF HDF5 inference path and denoiser:

- Model input per part = points **recentered → randomly rotated → scaled to unit
  cube** + normals (`assembly/data/breaking_bad/weighted.py:97-113`). Global tray
  layout is invisible to the model.
  → **Scale/spread is a placement/visualization effect, not a relative-fit
  cause.** (Demoted hypothesis H4.)
- Denoiser `forward_sdpa` uses a **full** global-attention mask; `calc_graph_mask`
  is defined but never called, and `fracture_surface_gt` is only a FracSeg/
  training target (`assembly/models/denoiser/modules/denoiser_transformer.py`).
  → Juglet's degenerate `shared_faces` (all `0` → star graph, all points flagged
  fracture) **do not directly corrupt inference**; they are a symptom that the
  break geometry was never validated, not the failure mechanism.
- The only signal the model has for how pieces mate is **the geometry of each
  piece's sampled surface points** (+ normals). So the investigation targets
  *what the encoder sees on Juglet vs BB*.

## Hypotheses (ranked)

- **H1 — Fracture-surface point starvation (thin-wall geometry).** Area-weighted
  sampling (`weighted.py:24-39`) puts almost all points on the broad inner/outer
  sherd walls; the thin break rim (the only matchable surface) gets ~no points.
  Per-part unit-cube scaling shrinks the rim further. **REFUTED (2026-06-05, Exp
  4c):** real thin-walled `ceramics` (pottery, fill 0.08–0.18 ≈ Juglet 0.13) score
  part_acc **0.92** (rmse_r 8.6°) on the same checkpoint — thin walls assemble fine.
  Exp 4b's collapse was a marching-cubes REMESH confound, not thinness. (egg fails
  but is a near-symmetric featureless surface of revolution; bones=chance, cross-
  domain 2-part.) Thin-wall alone is NOT the Juglet cause.
- **H2 — Real-scan domain gap (noise).** **REFUTED for Juglet** (2026-05-29): the
  real Juglet meshes are the *cleanest* set measured (hf/scale 0.004, ~10–17× below
  BB/Fractura). The bridge confirmed GARF is noise-sensitive, but Juglet isn't noisy.
- **H3 — Mesh topology issues** (non-manifold, open shells, inconsistent normals).
  **REFUTED for Juglet**: now fully watertight, highest-res; open-shell bridge had
  zero effect on accuracy.
- **H4 — Scale/translation distribution.** Demoted to placement-only effect.

---

## Fixed assets / commands (reused by all experiments)

- Juglet source meshes: `/data/gpfs/projects/punim2657/Dataset/Juglet_anchor_centered/Piece0[1-9].obj`
- Juglet HDF5: `GARF/input/juglet_deploy.hdf5` (sample `artifact/Juglet-000`)
- HDF5 builder: `GARF/create_juglet_hdf5.py`
- Checkpoint: `output/GARF.ckpt`
- Known-good reference: BB `everyday` 6-piece sample + Fractura `pig` sample
  (same ones used in `scripts/garf_matched_diagnostic.py`).
- Canonical eval call (from `infer_juglet_deploy.slurm`): `eval.py` with
  `experiment=denoiser_flow_matching`, `deploy_mode=true`, `anchor_free=true`,
  `write_to_json=true`, `save_assembly=true`, `data.data_root=<hdf5>`,
  `data.categories=['artifact']`.
- Result JSON of interest: `pred_trans_rots` (T,P,7 trajectory), final pose per
  part; meshes recoverable from the HDF5 + `mesh_scale`.

---

## Experiment 0 — No-GT quality probes (foundation)

**Purpose:** replace the misleading spread-ratio with metrics that reflect what
"assembled correctly" means without GT.

**Build** `scripts/no_gt_probes.py` taking (result JSON, source HDF5):
1. Apply predicted final pose per part to that part's mesh.
2. **Pairwise contact gap:** for each pair, nearest-surface distance between the
   two transformed meshes; report min-gap distribution. Aligned edges ⇒ several
   pairs with ~0 gap over a non-trivial contact area.
3. **Interpenetration:** estimate overlap volume / fraction of one part's
   sampled points inside the other's mesh. Correct assembly ⇒ ~0.
4. **Coverage of contact:** area of near-contact region per mating pair.
5. **Seed-stability hook:** given N result JSONs of the same object, compute
   pairwise relative-pose dispersion (rotation° and translation std across runs).

**Output:** `logs/diagnostics/probes/<obj>/probe_summary.{json,csv,md}`.
**Effort:** ~half day. **Blocks:** all later experiments.

---

## Experiment 1 — Seed/anchor stability on Juglet (cheapest discriminator)

**Purpose:** decide between "no usable signal" vs "systematic bias".

**Method:** run the canonical Juglet eval 5–10× varying `seed` (and a variant with
`++data.random_anchor=true`). Feed all result JSONs to the Exp-0 stability metric.

**Discriminating outcome:**
- High relative-pose dispersion across seeds → model has **no reliable mating
  signal** → supports H1/H2.
- Low dispersion but wrong → **systematic bias** → points at H3/normals/encoder.

**Effort:** ~1–2 GPU-hours (short jobs). **Depends on:** Exp 0.

---

## Experiment 2 — Fracture-rim point-coverage instrumentation

**Purpose:** directly test H1.

**Method:** add instrumentation (a standalone script reusing the dataloader
sampling) that, for Juglet and a known-good BB 6-piece object, computes per part
the fraction of sampled points lying within a thin band of the true mating rim.
Since Juglet lacks fracture labels, approximate the rim geometrically: rim =
points within ε of another part's surface in the (visually/manually) plausible
arrangement, or via boundary-edge detection on each open-shell mesh.

**Discriminating outcome:** Juglet fracture-coverage ≈ 0 while BB is substantial
⇒ H1 confirmed.

**Effort:** ~half day. **Depends on:** Exp 0 (shares geometry utils).

---

## Experiment 3 — Resampling intervention (H1 fix-test)

**Purpose:** if H1 holds, prove it by removing the starvation.

**Method:** rebuild Juglet point sampling to **upweight the break region**
(boundary-edge / rim-band oversampling, or Poisson-even sampling at higher
density), holding mesh geometry, scale, and poses-pipeline fixed. Re-run eval +
Exp-0 probes.

**Discriminating outcome:** assembly contact-gap/interpenetration improve
materially ⇒ H1 is the cause **and** we have a concrete remedy (custom sampler
for thin-walled artifacts).

**Effort:** ~1 day (sampler change + eval). **Depends on:** Exp 1, 2.

---

## Experiment 4 — Domain-bridge ablation (confirmation centerpiece)

**Purpose:** isolate the single dominant factor by morphing between domains.

**Forward (BB → sherd):** take one BB object GARF assembles well; apply, one at a
time and measure after each:
  (i) shell-thinning (hollow it to sherd-like wall thickness),
  (ii) scan-like vertex/normal noise,
  (iii) reduced fracture-area fraction / decimation of break faces.
Whichever single degradation collapses assembly = dominant cause.

**Reverse (Juglet → clean):** on a Juglet pair, clean/repair mesh, thicken, or
re-sample rims; if assembly recovers, confirms the same factor.

**Discriminating outcome:** a single edit that flips good↔bad pins the cause to
H1 vs H2 vs H3.

**Effort:** ~1–2 days. **Depends on:** Exp 0; informed by Exp 1–3.

---

## Experiment 5 — PTv3 feature/attention introspection (optional)

**Purpose:** only if Exp 1–4 stay ambiguous.

**Method:** dump PTv3 point features + attention for Juglet vs BB; check whether
fracture-region points are even separable / attended. 

**Effort:** ~1 day. **Depends on:** ambiguity after Exp 4.

---

## Execution order & decision flow

```
Exp 0 (probes)            ── foundation
   └─> Exp 1 (stability)  ── cheap: signal vs bias?
         ├─ no signal ──> Exp 2 (coverage) ──> Exp 3 (resample fix)
         └─ biased    ──> Exp 4 (domain bridge, focus on normals/topology)
Exp 4 confirms; Exp 5 only if still ambiguous.
Final: writeup + remedy.
```

Run 0 → 1 → 2 → 3 first (cheap and likely decisive); 4 is heavier confirmation.

## Progress log

- **2026-05-29** — Exp 0 built & validated: `scripts/no_gt_probes.py` (contact +
  stability subcommands). Smoke test on existing Juglet glb ran clean.
  - **Caveat found:** contact/min-gap alone rewards *pile-ups* (reported
    `single_object=True`, 16 contact pairs on a visually-wrong assembly), and
    interpenetration is `NaN` for non-watertight scan meshes. So **stability is
    the decisive probe**; contact is descriptive context.
  - **Preliminary hint:** stability over 3 pre-existing raw runs = low
    dispersion (mean ~3°, max ~8.7°, trans <1% of scale). If confirmed with
    controlled seeds → argues *against* "no signal" (H1/H2) and toward a
    **stable-but-wrong attractor** (H3 / systematic bias). Not yet conclusive
    (unknown seeds for those runs).
- **2026-05-29** — Exp 1 submitted: `exp1_seed_stability.slurm` (job
  `25584804`), 6 fixed-anchor seeds + 3 random-anchor, then stability + per-run
  contact probes → `logs/diagnostics/probes/exp1_stab_<stamp>/`.
- **2026-05-29** — Exp 1b submitted: `exp1b_reference_stability.slurm` (job
  `25584810`), known-good BB 6pc / BB 9pc / Fractura pig 9pc across 4 seeds with
  the same probes → `logs/diagnostics/probes/exp1b_ref_<stamp>/`. Provides the
  contrast baseline. Helper: `scripts/build_single_sample_subset.py`.

- **2026-05-29** — Exp 1 + 1b results (jobs 25584804, 25584810): **probes do NOT
  discriminate Juglet from known-good references.**
  - Relative-pose dispersion (gauge-invariant): Juglet 93.0°, BB-6pc 84.0°,
    BB-9pc 85.9°, Fractura-pig 84.1°. Trans disp: Juglet 0.41, BB 0.21–0.23,
    Fractura 0.42.
  - Contact: all report single_object=True, similar contact counts,
    interpenetration 0 (uninformative; scan meshes non-watertight).
  - Code check: `anchor_free` snap is a single GLOBAL SE(3) (denoiser_base.py
    404–427) → does not change relative config → the ~85° dispersion is the
    model's INTRINSIC seed-sensitivity, present on synthetic objects too.
  - **Conclusion:** no-GT self-consistency can't isolate the Juglet cause; the
    references aren't a clean contrast. Pivot to GT-correctness on references.
- **2026-05-29** — `one_step_init` clarified: it is a single warmup step BEFORE
  the full `num_inference_steps` (default 20) schedule, which always runs
  (denoiser_base.py 347–396). So "full-schedule" = `one_step_init=false`.
- **2026-05-29** — Exp 1c submitted: `exp1c_initstep_vs_fullschedule.slurm` (job
  `25587246`). BB-6pc / BB-9pc / Fractura-pig in BENCHMARK mode (real metrics),
  one_step_init ∈ {true,false} × 3 seeds → `logs/diagnostics/initcmp_<stamp>/`.
  Tests whether deploy inference config is a confound before blaming geometry.
  Tools: `scripts/summarize_benchmark_metrics.py`.

- **2026-05-29** — Exp 1c results (job 25587246), benchmark mode (real GT):
  - bb_6pc part_acc 1.00 / shape_cd 0.0002; bb_9pc 0.89–0.93; fractura_pig 0.85.
    **References are genuinely good → Juglet failure is real, model is capable.**
  - one_step_init neutral-to-helpful (bb_9pc 0.89→0.93). **Deploy config is not
    the confound.**
  - **Metric post-mortem:** cross-seed rotation dispersion = 80° on a
    part_acc=1.0 assembly; a single part pair differs 130° across seeds while
    chamfer is perfect → individual fragments (bottle shards = surfaces of
    revolution) have rotational symmetry. **Raw-rotation self-consistency is
    invalid; correctness needs chamfer/GT.** Exp 1/2 dropped as discriminators.
- **2026-05-29** — Exp 4a (domain bridge Tier 1) submitted: job `25591142`.
  `scripts/domain_bridge.py` deforms BB/Fractura fragments in-place (GT pose
  preserved) — transforms: noise (H2), decimate (H3), open-shell (H3); base =
  sanity gate. Sweeps strength × 2 sources (bb6, frpig) × 3 seeds, benchmark
  metrics → `logs/diagnostics/bridge_<stamp>/summary.md` via
  `scripts/summarize_bridge_metrics.py`. Tier-2 voxel thin-shell (H1) gated on
  Tier-1 outcome. (Installed `fast_simplification` into the uv venv with
  `--no-deps`; numpy unchanged at 2.4.0.)

- **2026-05-29** — Exp 4a results (job 25591142), benchmark mode (real GT):
  - **open-shell: no effect** — bb6 part_acc 1.00 at open10/open25, frpig 0.82→0.89.
    Removing faces / single-sided geometry does NOT hurt GARF. (H3 open-shell out.)
  - **noise: collapses fast** — bb6 1.00→0.39 at 1% noise (rmse_r 83°); 0.5% already
    degrades. **decimate: hurts at low res** — bb6 keep-10% → 0.44.
  - Naive reading was "Juglet fails from scan noise (H2)". **This was then refuted —
    see geom-stats below.**
- **2026-05-29** — Geometric-stats comparison (`scripts/geom_stats_compare.py`),
  per-fragment, normalized by frag max-extent so it's comparable to the bridge
  `strength`. Sampled BB everyday / Fractura pig / Juglet:
  - Juglet is the **cleanest** set, not the noisiest: high-freq normal roughness
    `hf/scale` **0.004** vs BB 0.067 / Fractura 0.039 (~10–17× lower); dihedral
    11.5° vs ~21°; **highest resolution** (13.4k faces, edge/scale 5× finer);
    **fully watertight** (1.00 vs 0.73–0.98).
  - **→ H2 (scan noise) and H3 (resolution/open/topology) are REFUTED for Juglet**:
    it is better than the known-good data on every one of those axes. The bridge
    found a real GARF weakness (noise), but **Juglet does not trigger it**.
  - The one axis where Juglet is the clear outlier is **solidity**: `fill_ratio`
    **0.126** vs BB 0.400 / Fractura 0.213; wall `thickness/scale` **0.075** vs
    0.167 / 0.135. Juglet sherds are **thin pottery walls**; BB everyday are solid.
    **→ promotes H1 (thin-wall) as the leading mechanism.**
- **2026-05-29** — Juglet meshes rebuilt watertight + rerun (GARF + TORA on
  `*_local02.hdf5`, local-cluster tightness matched to prior 0.935 via α=0.069).
  Both still fail to assemble. **Important correction:** an earlier "grouped"
  Juglet result was a **geometry artifact** — the previous Piece03 mesh was wrongly
  oversized (extent 0.97), making it the anchor; the corrected watertight Piece03
  is small (0.46) and Piece01 is the true largest/anchor. So the prior "GARF
  groups the pieces" was coincidental on bad geometry, **not** real assembly. The
  honest result on correct geometry: GARF does not reassemble Juglet — consistent
  with H1. (Pipeline: `rebuild_juglet_pipeline.py`.)
- **2026-05-29** — Exp 4b (thin-shell bridge, H1) submitted: job `25599264`.
  Added `t_shell` to `scripts/domain_bridge.py`: voxelize+fill each solid fragment,
  erode interior, keep boundary band, re-mesh via marching cubes in the SAME world
  frame (GT pose preserved; verified centroid Δ≤0.004, extent intact). Sweeps wall
  `thickness/scale` ∈ {0.02,0.04,0.06,0.08} (brackets Juglet 0.075; shell04 gives
  fill≈0.15 ≈ Juglet 0.126) × 2 sources × 3 seeds, benchmark metrics →
  `logs/diagnostics/bridge2_<stamp>/summary.md`. If part_acc collapses at
  Juglet-like wall thickness on BOTH sources, H1 is confirmed. (Installed
  `scikit-image` into the uv venv for marching cubes.)

- **2026-05-29** — Exp 4b results (job 25599264), benchmark mode (real GT) —
  **H1 CONFIRMED, root cause found.**
  - base (solid): bb6 part_acc **1.00** (rmse_r 2.8°, cd 0.0002); frpig **0.85**.
  - thin-shell (hollowed, GT-preserved): bb6 → **0.17–0.28**, frpig → **0.11–0.22**
    across wall thickness {0.02,0.04,0.06,0.08}; rmse_r 72–87°; shape_cd ~100× worse.
    part_acc lands at ~chance (1/6≈0.167, 1/9≈0.111) → only the anchor placed.
  - **Collapse is flat across all thicknesses** (incl. shell08 > Juglet's 0.075) →
    it's the shell *topology* (two thin walls + thin fracture rim), not a fine
    thickness threshold. Juglet (0.075) is firmly inside the collapse regime.
  - Contrast with Exp 4a: open-shell (face removal) had **zero** effect, but
    hollowing collapses hardest of any factor (worse than 1% noise's 0.39). The
    difference is exactly H1: a thin shell starves the matchable fracture rim of
    sampled points while doubling area on the broad inner/outer walls.
  - **Remedy implied:** rim/fracture-region oversampling (Exp 3) and/or training on
    thin-walled data. Next: implement Exp 3 intervention on Juglet + a thin-shell BB.

- **2026-06-05** — Challenge to H1 (user) + Exp 4c submitted (job 25690421).
  Observation: BB/Fractura DO include thin-walled objects. `fractura_real.hdf5` has
  `egg` (fill 0.016–0.022, ultra-thin), `ceramics` (fill 0.08–0.18 ≈ Juglet 0.13,
  real pottery), `bones` (0.06–0.26) — all REAL scans WITH GT (very high-res,
  ~1e5–1e6 faces). Also exposes a confound in Exp 4b: the synthetic `t_shell`
  remeshes via marching cubes, so its collapse may be a remesh artifact, not
  thinness. Exp 4c (`exp4c_real_thinwall.slurm`) evals GARF in benchmark mode on
  egg/ceramics/bones × 3 seeds → `logs/diagnostics/thinwall_<stamp>/summary.md`.
  Decision: egg/ceramics GOOD ⇒ thin-wall NOT the cause (look at real-scan rim
  character / Juglet-specific factors); egg/ceramics BAD ⇒ thin-wall confirmed on
  real data. (Optional follow-up control: voxel-remesh the SOLID with no erosion to
  isolate the marching-cubes artifact.)

- **2026-06-05** — Exp 4c results (job 25690421), benchmark mode (real GT):
  - ceramics part_acc **0.917±0.12**, rmse_r 8.6°, cd 0.0009 (4 parts) — **excellent**.
  - egg 0.333 (=chance/3, ultra-thin featureless surface-of-revolution, symmetry).
  - bones 0.500 (=chance/2, cross-domain). 
  - **H1 thin-wall REFUTED**: real pottery (Juglet's analog) assembles great; Exp 4b
    was a remesh artifact. **New question:** ceramics works but Juglet (same material/
    thickness) fails — differentiator is NOT thinness.
  - **Next hypotheses to test (Juglet vs Fractura ceramics, the matched success case):**
    (a) **part count** — ceramics 4–5 vs Juglet 9; subsample Juglet to 4–5 sherds and
        re-eval contact/visual. (b) **fracture-surface erosion** — archaeological sherd
        rims are worn/low-feature (Juglet dihedral very low) vs Fractura's crisp fresh
        breaks; quantify rim sharpness/curvature on mating bands. (c) **completeness** —
        is the Juglet set missing pieces (no global solution)?

- **2026-06-09** — User refuted completeness (9 sherds DO form a closed object)
  and part-count (GARF works on some high-P, fails on some low-P). Confirmed in
  Exp4c data: galli_pot (10P)=1.00 but narrow_bottle1 (12P)=0.08; narrow_bottle4
  (4P)=1.00 but narrow_bottle3 (4P)=0.50. So part count is NOT the cause.
- **2026-06-09** — Five geometric probes vs Exp4c part_acc (scripts:
  `symmetry_analysis.py`, `fracture_contact_analysis.py`,
  `fracture_sharpness_analysis.py`, `contact_band_relief.py`, + aspect/size):
  - **Global axisymmetry**: r=-0.04 (REFUTED). Most-axisymmetric piece
    (narrow_bottle4) succeeds.
  - **Contact area fraction**: weak/none (narrow_bottle1 FAILS at high contact
    0.39; narrow_bottle2 WORKS at 0.14).
  - **Aspect ratio / part-size disparity**: no separation (plate works at
    disparity 5.6; narrow_bottle4 works at aspect 1.91).
  - **Whole-piece fracture relief (resolution-independent)**: Juglet 0.171 vs
    Fractura-ceramics 0.272 (~37% lower) — consistent with worn archaeological
    edges — but Juglet ≈ BB-everyday 0.183 (which works), so not a sole cause.
  - **Contact-band relief**: r=-0.68 within ceramics+egg but CONFOUNDED — egg's
    high value is fine-scale scan noise on 1e6-face meshes, not interlocking.
  - **Verdict:** No single simple mesh statistic predicts GARF success, not even
    the within-ceramics failures. The failing fresh-pottery cases (egg,
    narrow_bottle1/3) are the smoothest/least-featured vessels; the determinant
    is *instance-level fracture-surface matchability* (is THIS set of breaks
    distinctive & complementary), which aggregate geometry doesn't capture.
  - **Synthesis for Juglet:** it sits in GARF's hard regime — thin-walled + 9
    pieces + near-body-of-revolution vessel — where even fresh thin pottery
    (egg, narrow_bottle1) fails, PLUS an archaeological-erosion handicap (fracture
    relief ~37% below fresh ceramics). Not one cause; a stack of disadvantages at
    GARF's capability boundary.
  - **Actionable:** since some thin pottery succeeds, the lever is the per-instance
    fracture signal — rim-focused resampling (Exp 3) and/or fine-tuning on
    worn/archaeological breaks — not "GARF can't do thin walls."

- **2026-06-10** — Exp 6 (pairwise oracle decomposition) designed + submitted.
  Rationale: since Exp 4c everything was measured on Fractura proxies; nothing
  yet separates the two remaining failure modes on Juglet itself —
  (a) **perception failure** (encoder extracts no mating signal from worn rims)
  vs (b) **joint-inference failure** (pairwise signal exists but the 9-piece
  joint flow can't compose it). Method: `scripts/build_juglet_pairs_hdf5.py`
  copies the canonical rebuilt local02 geometry into all C(9,2)=36 two-piece
  samples (`input/juglet_pairs_local02.hdf5`); `exp6_pairwise_oracle.slurm`
  evals all pairs (deploy, anchor_free) × seeds {41,42,43};
  `scripts/summarize_pairwise_oracle.py` scores each pair by contact frac +
  interpenetration (meshes now watertight → meaningful) + cross-seed
  relative-pose dispersion (2 parts → ONE rel pose; a repeated unique pose
  across seeds is strong evidence of a real mate, sidestepping the Exp-1
  symmetry pitfall) → `logs/diagnostics/pairs_<stamp>/summary.md` + GLB
  gallery. Decision rule: zero mated pairs ⇒ perception failure (next: rim
  matchability probe / erosion bridge); true pairs mate ⇒ search failure
  (next: hierarchical pairwise-greedy assembly + iters lever). Needs physical
  adjacency list from user to label which of the 36 pairs are TRUE mates.

- **2026-07-09** — Exp 6 made interpretable + Exp 6b control + pairwise chamfer
  re-scoring. **VERDICT: pairwise PERCEPTION failure confirmed.**
  - **Adjacency from PF++** (`scripts/derive_pfpp_adjacency.py`): reproduced
    PF++'s renderer `compute_final_transformation` in numpy, applied it to the
    stored per-part clouds; the PF++ assembly compacts to a tight vessel and
    yields 18/36 touching pairs = true-mate labels
    (`logs/diagnostics/juglet_adjacency/`).
  - **Cross-tab of true mates vs Exp 6 metrics**: GARF shows NO signal — true
    mates' cross-seed rot dispersion 71.4° vs non-mates 67.1°; of the 18 lowest-
    dispersion pairs only 8 are true mates (< chance). Suggested perception
    failure but not yet conclusive.
  - **Exp 6b positive control** (job 27000787, `scripts/build_control_pairs_hdf5.py`,
    `summarize_control_pairs.py`): identical pairwise oracle on known-good
    Fractura ceramics (pink_bowl/narrow_bottle2/narrow_bottle4/blue_pot, all
    part_acc ≥0.92; 22 pairs, 18 GT mates). **The rot-dispersion proxy is
    INVALID**: control true mates also ~71.8° dispersion (non-mates 78.7°) — the
    probe can't separate mates even on the success cases. Confounded by per-sherd
    rotational symmetry (surfaces of revolution), exactly the Exp-1c pitfall.
  - **Symmetry-invariant re-scoring** (`scripts/pair_reference_chamfer.py`):
    per-pair assembled-shape chamfer vs a reference pose (real GT for control,
    PF++ pseudo-GT for Juglet), correspondence-free multi-init ICP so the
    unobservable symmetry DOF is not penalised. Metric VALIDATED on control:
    true mates chamfer/diag **0.024** vs non-mates **0.039** (clear separation;
    best true mates ~0.008). On **Juglet: true mates 0.070 vs non-mates 0.073 —
    NO separation, and ~3× worse than the control's true mates**. Even Juglet's
    best true mate (0.039) is worse than the control's median true mate.
  - **Conclusion:** GARF cannot align even genuinely-mating Juglet sherd pairs —
    it places a true pair no better than a non-mating pair, and far worse than
    fresh-ceramic mates. The failure is **perceptual at the pairwise level** (no
    usable mating signal from worn archaeological rims), NOT primarily 9-piece
    joint-inference/search. Consistent with Exp 5 (inference-lever sweep barely
    helps) and the worn-rim relief handicap. Outputs:
    `logs/diagnostics/{pair_chamfer_control,pair_chamfer_juglet}/summary.md`,
    `logs/diagnostics/ctrl_pairs_20260709_144515/summary.md`.
  - **Actionable remedy (unchanged direction, now evidenced):** target the
    per-instance fracture signal — rim-focused resampling and/or fine-tuning on
    worn/eroded breaks — not joint-inference knobs. Next confirmatory test:
    rim-erosion domain bridge (add an `erode` transform to `domain_bridge.py`:
    smooth a working ceramic's fracture relief toward Juglet's level, GT pose
    preserved, and confirm pairwise chamfer collapses).

- **2026-07-10** — Exp 7 (rim-erosion bridge, confirmation) + Exp 8 (rim-
  oversampling remedy) designed, validated locally, and submitted.
  - **Method correction 1 (erosion):** the first `erode_fracture_rim`
    (dihedral-masked Taubin blend) barely moved relief_p90 (0.168→0.157 at
    strength 1.0) — Taubin/Laplacian smoothing diffuses at a mesh-resolution-
    dependent rate, useless on ~1e6-face Fractura scans. Replaced for Exp 7 by
    `fracture_mesh_ops.erode_fracture_band`: the TRUE fracture band is found
    physically (verts within 2% of object scale of another piece in the stored
    ASSEMBLED pose, feathered), then mollified — Gaussian-weighted average of
    surface samples within a FIXED PHYSICAL radius (strength × 5% piece scale).
    Resolution-independent; in-place vertex edit so GT stays valid. Calibration
    (blue_pot probe): strength 0.5 lands mean relief_p90 ≈ Juglet's 0.171;
    1.0 overshoots to ~0.11.
  - **Exp 7 submitted (job 27038450)** `exp7_rim_erosion_bridge.slurm`:
    `build_control_pairs_hdf5.py --erode-strength` × {0, .25, .5, .75, 1.0} on
    the 4 known-good ceramics, pairwise oracle (seeds 41/42/43), symmetry-
    invariant chamfer per strength, `summarize_erosion_bridge.py` decision →
    `logs/diagnostics/exp7_erosion_<stamp>/summary.md`. CONFIRMED if true-mate
    chamfer collapses from 0.024 toward Juglet's 0.070 and mate/non separation
    dies as relief approaches Juglet's level.
  - **Method correction 2 (remedy):** the planned `densify` (subdividing rim
    faces) is a NO-OP — GARF samples `trimesh.sample.sample_surface`
    (area-weighted) / `sample_surface_even`, and midpoint subdivision preserves
    area, so point density never changes. Docstrings corrected; the remedy
    moved into the SAMPLER: new `data.rim_oversample_frac` (+ `rim_band_frac`,
    `rim_relief_pct`) in `assembly/data/breaking_bad/{base,weighted,module}.py`
    + `configs/data/breaking_bad.yaml` forces that fraction of each part's
    point budget onto the rim band. Band detection is relief-at-physical-radius
    (top-15% relief anchors, faces within 5% of piece scale) because dihedral
    thresholds fail on finely-meshed worn fillets (Juglet band came out 91% of
    the piece with dihedral, 37% with relief). Detector validated on blue_pot
    against the true assembled contact band: precision 0.30–0.99, recall
    0.63–0.97; sampled points on Juglet band 0.40→0.54 at frac 0.35. frac=0.0
    reproduces the original behaviour bit-for-bit (default).
  - **Exp 8 submitted (job 27038543)** `exp8_rim_oversample_remedy.slurm`:
    Juglet 36 pairs at frac {0.35, 0.60} + control-pairs regression at 0.35 +
    full 9-piece Juglet at both fracs; scored with the same chamfer oracle;
    `summarize_rim_remedy.py` decision → `logs/diagnostics/exp8_rim_<stamp>/`.
    If pairwise chamfer improves >10% and mate/non separation emerges → remedy
    works, promote to full assembly; if not → fine-tuning on eroded breaks
    (training-time `erode_fracture_band` augmentation) is the remaining lever.

- **2026-07-10 (results)** — Exp 7 NOT confirmed, Exp 8 no effect. Both
  completed same day (jobs 27038450 1h36m, 27038543 1h59m).
  - **Exp 7 verdict: NOT confirmed** —
    `logs/diagnostics/exp7_erosion_20260710_141409/summary.md`. Eroding the
    control ceramics' true fracture bands degrades pairwise perception
    monotonically (true-mate chamfer/diag 0.0229→0.0317 mean over strengths
    0→1.0) and weakens mate/non separation (1.61x→1.24x), but plateaus at less
    than half of Juglet's failure level (0.070, separation 1.04x). **Caveat:**
    the achieved band relief_p90 saturated at ~0.201 for strengths 0.5–1.0 and
    never reached Juglet's measured 0.171 — the mollifier stops carving once
    the band is locally smooth at its fixed physical radius, so the bridge
    undershot its calibration target. Interpretation: worn-rim relief is a
    CONTRIBUTING factor (monotonic degradation is real) but is NOT sufficient
    to reproduce the Juglet failure; some other Juglet-specific property
    carries most of the deficit.
  - **Exp 8 verdict: no effect — remedy refuted** —
    `logs/diagnostics/exp8_rim_20260710_142521/summary.md`. Juglet true-mate
    median chamfer/diag: baseline 0.0731 → frac 0.35: 0.0705 → frac 0.60:
    0.0694 (best −5%, fails the pre-registered >10% bar) and NO mate/non
    separation emerges (0.92–1.02x; at frac 0.60 true mates score WORSE than
    non-mates). Control regression at frac 0.35 PASSES (true-mate median
    0.0259 vs non-mate 0.0359 — no harm to working cases). 9-piece deploy GLBs
    at both fracs written to `logs/deploy/exp8_j9pc_20260710_142521_f{035,060}/`
    (not separately scored; pairwise gate already failed). Conclusion: the
    perception deficit is NOT point-starvation at the rim — the encoder sees
    the rim points and still extracts no mating signal.
  - **Combined implication:** the original remedy rationale (worn rims starve
    the encoder of signal; oversample or fine-tune on eroded breaks) is now
    only partially supported. Rim wear degrades but does not break perception,
    and giving the encoder more rim points does not help. Before committing to
    fine-tuning, the open question is WHAT Juglet-specific property remains:
    candidates not yet isolated include fracture-band geometry class (sharp
    conchoidal vs rounded/weathered fillet profile, beyond relief amplitude),
    surface texture/curvature statistics on the break faces, and PF++
    pseudo-GT label error inflating the apparent deficit. A cheap next probe:
    re-run the erosion bridge at larger mollify radius (to break the relief
    plateau) and/or score Juglet pairs against a relaxed reference (best-fit
    over both pieces) to bound pseudo-GT error. Fine-tuning on
    `erode_fracture_band`-augmented breaks remains the actionable lever, but
    expectations should be tempered: Exp 7 suggests eroded-fresh-ceramic is
    not a faithful proxy for the Juglet domain.

- **2026-07-10 (follow-up probes submitted)** — Exp 7b (erosion radius sweep,
  job 27043960) + Exp 9 (pseudo-GT label-error bound, job 27043961).
  - **Exp 7b `exp7b_erosion_radius.slurm`:** breaks Exp 7's relief plateau.
    Diagnosis of the plateau: (1) the mollifier stops carving once the band is
    locally smooth at its fixed physical radius (strength × 0.05 × piece
    scale); (2) `erode_fracture_band`'s `knn=48` averages only the 48 nearest
    of 20k surface samples, silently shrinking the effective kernel at larger
    radii. Sweep holds strength=1.0 and grows the kernel: kernel_frac
    {0.10, 0.15, 0.20} with knn scaled ~quadratically {192, 432, 768}
    (`build_control_pairs_hdf5.py` gained `--erode-kernel-frac`/`--erode-knn`;
    `summarize_erosion_bridge.py` gained `--x-label`). Bridge curve indexed by
    effective mollify radius, reusing exp7's scored 0.00/0.05 points.
    CONFIRMED if relief reaches ≤0.171 and true-mate chamfer collapses toward
    0.070; if relief reaches Juglet's level and chamfer stays low, full-depth
    wear is ruled out as sufficient mechanism.
  - **Exp 9 `exp9_pseudo_gt_bound.slurm` + `scripts/juglet_pseudo_gt_bound.py`
    (CPU-only, sapphire):** bounds how much of Juglet's 0.070 true-mate
    chamfer is PF++ label error. Per true-mate pair: trimmed band-constrained
    ICP (keep=0.7 — plain ICP over-pulls even real-GT bands, drift ~0.009)
    refines the reference pose; drift = label shift in headline chamfer/diag
    units; existing Exp 6/6b prediction GLBs re-scored against the refined
    reference. Control ceramics (real GT) run identically as built-in
    validation; the metric's sampling-noise identity floor (~0.005 at n=4000,
    measured: pair scored against itself) is reported so sub-floor drifts are
    not over-read. Decision: rescored Juglet median >0.050 → labels fine,
    perception-failure conclusion stands; collapse toward 0.024 → replace the
    pseudo-GT before further remedy work.

- **2026-07-13** — Exp 10 (encoder introspection, the long-deferred Exp 5) built
  and run: `scripts/fracseg_introspection.py`, `exp10_fracseg_introspection.slurm`
  (job 27188479). First look INSIDE the frozen feature extractor. It is a
  `FracSeg` module whose `coarse_segmenter` head outputs per-point P(fracture);
  the same backbone feeds the denoiser. Three arms: labeled synthetic pig
  fractures (probe validation), fresh real ceramics (GARF works), Juglet (fails).
  - **Probe validated:** on synthetic breaks the frozen segmenter fires on the
    correct points, AUC vs true fracture GT = **0.95**. (Relief-band AUC is a bad
    proxy — P(fracture) anti-correlates with relief because relief anchors the
    rough ORIGINAL surface; fired% is the trustworthy readout.)
  - **RESULT — encoder blind to worn breaks:** fracture-response (fired %) =
    **9.7%** synthetic / **3.4%** fresh real ceramics / **0.57%** Juglet
    (0.17× control). GARF's pretrained fracture representation assigns near-zero
    fracture probability across Juglet's worn rims — it does not perceive them as
    fractures, so the denoiser gets features with no mating cue. This is the
    DIRECT feature-level mechanism behind the Exp 6 pairwise perception failure,
    and identifies the encoder (not the denoiser) as the lever.
    Output: `logs/diagnostics/exp10_fracseg_20260713_174759/summary.md`.
- **2026-07-13** — Exp 10b (causal bridge) submitted:
  `exp10b_erosion_blindness.slurm` (job 27188744). Erodes the TRUE fracture bands
  of the labeled synthetic breaks toward Juglet-like wear (labels preserved,
  `--synth-erode-strength` in `fracseg_introspection.py`) at strengths
  {0, 0.5, 1.0} and re-measures the encoder's fired% / AUC-vs-GT. Confirms wear is
  the CAUSE of the blindness if fired% collapses toward Juglet's level as erosion
  grows — which justifies the remedy (Exp 11: fine-tune the FracSeg backbone on
  eroded-break-augmented synthetic data, then re-extract features + re-run the
  pairwise oracle / 9-piece assembly).

## Deliverables

- `scripts/no_gt_probes.py` + reusable geometry utils.
- `scripts/derive_pfpp_adjacency.py`, `build_control_pairs_hdf5.py`,
  `summarize_control_pairs.py`, `pair_reference_chamfer.py` (Exp 6 closure).
- Instrumented sampling/coverage script.
- Per-experiment `logs/diagnostics/...` CSV/MD outputs.
- `JUGLET_ROOTCAUSE_FINDINGS.md` final writeup naming the mechanism + remedy.
