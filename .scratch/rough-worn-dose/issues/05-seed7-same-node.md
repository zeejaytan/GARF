# 05: Seed-7 adapters over ten presentations, one GPU node

**What to build:** score the second training run (seed 7) of clean-break, rough 1× and worn 4×
on the same ten Juglet/Fractura presentations as ticket 02, all on one GPU node, so the
comparison repeats the *training*, not only the scoring. Same job, same node: re-score seed-42
clean-break and worn 4× on presentations 7, 9, 10 (worn's +0.15 / +0.20 wins and its one loss
in ticket 02, which ran clean and worn on different nodes, gpgpu111 / gpgpu122).

**Answers:** G1 (whether "rough training helps" is earned, or only "this rough adapter beats
this clean-break adapter")

**Blocked by:** 02, 03 (done)

**Status:** ready-for-agent

Fixed before results (2026-10-02), using the ticket-02 rule. Juglet sherds per attempt,
mean of 20, per presentation:
- Seed-7 rough (or worn) above seed-7 clean-break on **≥9/10 presentations, each by more
  than 0.15** → "rough (worn) training helps on the Juglet, small effect, two training runs".
- 7-8/10 → inconclusive; G1 keeps "one adapter beats one adapter".
- ≤6/10 → the seed-42 gain was that adapter's luck; G1 says rough training does not
  reliably help.
- Seed-42 worn vs clean on 7/9/10, same node: if worn is not ahead on both 7 and 9, worn's
  ticket-02 count becomes 7/10 and worn is written as no shown effect.
- Fractura reported per pot (narrow_bottle1 and narrow_bottle3 are the only pots that move).

- [x] Job submitted, polled, final sacct State/ExitCode recorded; node recorded (31939551, COMPLETED 0:0, 1h41, spartan-gpgpu107; laptop poll hit its 2 h cap, state read from sacct)
- [x] Reconcile 0 mismatches in every run (66 runs)
- [x] Table + sign counts against the rule above
- [x] G1 updated with the result and the date (2026-10-02)

Script: `slurm/rwlora_same_node.slurm` (calls `slurm/rwlora_arm.slurm` per arm).

## Results (2026-10-02)

Juglet, sherds per attempt (mean of 20, pinned sherd included), seed-7 adapters, one node:

| presentation | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | mean | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean-break s7 | 5.30 | 3.90 | 3.30 | 2.90 | 2.55 | 2.35 | 1.25 | 2.50 | 1.20 | 1.80 | 2.71 | 8/9 |
| rough 1× s7 | 6.55 | 3.90 | 5.00 | 5.35 | 3.35 | 3.25 | 1.85 | 2.70 | 1.95 | 1.55 | 3.55 | 9/9 (3 of 200) |
| worn 4× s7 | 7.10 | 4.15 | 5.05 | 6.70 | 3.55 | 2.55 | 1.40 | 2.75 | 1.35 | 2.20 | 3.68 | 9/9 (8 of 200) |

- Rough s7 − clean s7: +1.25 0 +1.70 +2.45 +0.80 +0.90 +0.60 +0.20 +0.75 −0.25 → **8/10 by more
  than 0.15 → inconclusive** by the rule. Mean +0.84 (seed 42: +1.41).
- Worn s7 − clean s7: +1.80 +0.25 +1.75 +3.80 +1.00 +0.20 +0.15 +0.25 +0.15 +0.40 → above on
  **10/10**, never below, but two wins are exactly 0.15, so **8/10 by the rule → inconclusive**.
  Mean +0.97 (seed 42: +1.08).
- Seed-42 re-score on gpgpu107, presentations 7/9/10: clean 1.15/1.45/2.35, worn 1.30/1.65/2.30 —
  **identical to ticket 02** (gpgpu111/gpgpu122). The node did not move these runs; worn is ahead
  on both 7 and 9, so its ticket-02 count stays 9/10 (by +0.15 and +0.20, i.e. 3-4 more sherds
  over 20 attempts).
- Fractura (sum over 20 attempts, per presentation, vs clean s7): narrow_bottle1 rough above 8/10,
  worn 10/10; narrow_bottle3 rough 8/10, worn 8/10. The other pots do not move.
- Across both training runs: rough ahead of clean-break on 17 of 20 (training run × presentation),
  worn on 19 of 20.

Reading: the direction **replicates in a second training run** for both rough and worn, at about
two-thirds the size for rough; neither clears the strict bar fixed beforehand. G1 keeps "one
adapter beats one adapter" wording per the rule, with this line added.

Render: `artifacts/rwlora/jug_t05.png` — the attempt nearest each model's mean (not the best),
presentations 4 and 7, each point coloured by its distance from home in mm. Agrees with the
counts: on presentation 4 worn's typical attempt has the lower body mostly within 4 mm, rough's
partly, clean-break's not; on presentation 7 every model's lower body is 20-30 mm off.

**Found while rendering (not part of the rule):** own_place counts a sherd home by where its
points sit, not which way it faces. On presentations 4 and 7, sherds scored home but whose points
are a median >4.6 mm from home (in place but turned): clean-break 19 of 43, rough 23 of 104,
worn 30 of 122. A stricter, orientation-aware count would widen the gap, not close it; two
presentations only. Worth its own ticket before any orientation-aware ruler is used.

## Full reassembly, own vs orientation-aware (2026-10-02, all 1,400 saved Juglet attempts)

Strict = own AND the sherd's points sit a median <4.6 mm from home (pred/gt points correspond
one-to-one; mm = 65 / longest gt box side). Per arm, 200 attempts (10 presentations × 20 starts).

| arm | own 9/9 | own ≥8 | own ≥7 | strict 9/9 | strict ≥8 | strict ≥7 | mean own | mean strict |
|---|---|---|---|---|---|---|---|---|
| released s42 | 0 | 0 | 0 | 0 | 0 | 0 | 2.04 | 1.30 |
| clean s42 | 0 | 2 | 5 | 0 | 0 | 0 | 2.46 | 1.64 |
| rough s42 | 5 | 15 | 35 | 1 | 2 | 6 | 3.87 | 2.56 |
| worn s42 | 5 | 12 | 21 | 0 | 2 | 5 | 3.54 | 2.46 |
| clean s7 | 0 | 2 | 3 | 0 | 0 | 0 | 2.71 | 1.72 |
| rough s7 | 3 | 13 | 20 | 0 | 1 | 7 | 3.54 | 2.44 |
| worn s7 | 8 | 15 | 27 | 1 | 2 | 10 | 3.68 | 2.60 |

- Of 21 own-scored 9/9s, **2 are genuine** (rough s42 presentation 1 #19: every sherd ≤13°,
  ≤2 mm; worn s7 presentation 4 #11: max 17°, ≤3.9 mm). The rest have sherd 7 (or 8) spun
  170-179° about its own face normal (centroid ≤2.4 mm, chamfer 1.2 mm, pointwise median
  15.7 mm), or sherds 2/4/5 turned 28-46°.
- The 9/9 the conservator saw (`visual-qa/viewer/pairs/garf_juglet_worn4x_rw01.json`, seed-42
  worn, presentation 42, attempt 5): per-sherd turns 0/27/17/143/25/169/19/179/60° — sherds 3,
  5, 7 spun half a turn. **Not a genuine reassembly. Measurement broken, not method.**
- Near-complete, strict, both training runs: clean-break 0/400 at ≥7, rough 13/400, worn 15/400.
- Render: `artifacts/rwlora/sherd7_flip.png` (face-on, home vs placed, coloured by home height).
- Literature part accuracy (chamfer <0.01) has the same blind spot; no reassembly paper found
  that reports an all-sherds-correct rate (paper-reader, 2026-10-02).
