# 02: Ten presentations of the Juglet and Fractura, four models

**What to build:** the four existing models (released GARF, clean-break fine-tune, rough 1×,
worn 4×; adapters from ticket 01, no retraining) each run on the Juglet and the 8 Fractura
pots from **10 new presentations** (eval seeds 1-10), 20 attempts each. A presentation is
what the eval seed fixes once per run and all 20 attempts share: the surface points sampled
from each sherd and the orientation each sherd is handed in. The 20 shuffled starts vary
within it. Read-out per presentation: best of 20 (attempts reaching it) and mean sherds
seated per attempt, scored by `tora/scripts/own_place.py` (`--pot-mm 65` for the Juglet).

**Answers:** G1 (whether rough or worn training has any effect on the Juglet once the
presentation is varied; ticket 01 tested 3 presentations, too few to rule a small effect in
or out)
**Blocked by:** None (can start immediately)
**Status:** ready-for-agent

Why: on ticket 01's three presentations, worn 4× averaged 4.7, 1.3 and 2.3 sherds per
attempt with the same model and sherds. The shuffles alone cannot make that spread, so the
honest sample size is the number of presentations, not the number of attempts.

Readings, fixed before results (per presentation, compare each model's mean against the
clean-break fine-tune's mean on the same presentation):
- Rough 1× ahead on ≥ 9 of 10 → a small real effect (chance ≈ 1 in 100); report its size in
  sherds. ≤ 6 of 10 → no effect. 7-8 → inconclusive, say so.
- Same rule for worn 4×.
- Also report the spread of released GARF's mean across the 10 presentations: how much the
  Juglet result depends on presentation alone, with no fine-tune involved.

- [ ] `rwlora_arm.slurm` run with `ADAPTER=` and `DRAW_SEEDS="1 2 3 4 5 6 7 8 9 10"`,
      `SETS="juglet_gt fractura_fresh"`, one job per model; final sacct State/ExitCode recorded
- [ ] Reconcile 0 mismatches in every run
- [ ] Table: model × presentation, Juglet best/mean, Fractura sum; the sign counts above
- [ ] Best Juglet attempt of the best model rendered beside the conservator's reassembly
      before reporting; worst-presentation attempt rendered too
- [ ] G1 updated with the result and the date

## Results (2026-10-01)

Jobs 31835633 (untouched), 31835634 (fresh), 31835635 (rough 1×), 31835636 (worn 4×): all
COMPLETED 0:0, 31-33 min each. Reconcile 0 mismatches, 0 borderline in all 80 runs.
Collected with `logs/rwlora/rw_collect.py` → `artifacts/rwlora/rw_all.json`; table from the
scratch script `rw_t02.py`.

Juglet, best of 20 (attempts reaching it) and mean sherds seated per attempt, of 9:

| presentation | released GARF | clean-break | rough 1× | worn 4× |
|---|---|---|---|---|
| 1 | 4 (1) 2.00 | 8 (2) 5.15 | 9 (3) 7.00 | 9 (3) 7.00 |
| 2 | 6 (1) 2.75 | 6 (1) 3.25 | 7 (1) 3.75 | 7 (1) 4.20 |
| 3 | 4 (1) 1.60 | 7 (1) 3.15 | 8 (4) 6.00 | 8 (1) 5.25 |
| 4 | 6 (1) 3.90 | 4 (3) 2.55 | 9 (2) 6.00 | 9 (2) 5.65 |
| 5 | 4 (3) 2.30 | 3 (2) 1.55 | 7 (1) 3.65 | 5 (4) 3.05 |
| 6 | 4 (1) 2.15 | 4 (1) 1.70 | 5 (3) 2.70 | 5 (1) 2.20 |
| 7 | 3 (2) 1.40 | 2 (3) 1.15 | 4 (1) 1.85 | 2 (6) 1.30 |
| 8 | 3 (2) 1.40 | 5 (1) 2.25 | 7 (2) 3.70 | 8 (1) 2.75 |
| 9 | 2 (3) 1.15 | 3 (2) 1.45 | 3 (7) 2.00 | 3 (3) 1.65 |
| 10 | 3 (3) 1.75 | 5 (1) 2.35 | 3 (6) 2.00 | 4 (1) 2.30 |
| mean of means | 2.04 | 2.46 | 3.87 | 3.54 |

- **Rough 1× mean above clean-break on 9/10 presentations** (below on 10) → pre-registered
  reading: a real effect. Size: +1.4 sherds per attempt on average (3.87 vs 2.46 of 9).
- **Worn 4× above on 9/10** (below on 10, 2.30 vs 2.35) → real effect, +1.1 sherds.
- 9/9 reached by rough 1× and worn 4× on presentations 1 and 4 (5 attempts of 200 each);
  clean-break's best is 8/9, released GARF's 6/9.
- Released GARF's mean ranges 1.15-3.90 across presentations; worn 4×'s 1.30-7.00. The
  presentation moves the result more than the fine-tune does.
- Released GARF vs clean-break: above on 4/10 → no effect of clean-break fine-tuning.

Fractura, sum over 8 pots of best of 20 (47 sherds per presentation, 470 over 10):
released 358, clean-break 371, rough 1× 399 (above clean-break on 10/10, below on 0),
worn 4× 402 (above on 7/10, below on 0, tied 3).

Presentation 7 here is the same presentation as ticket 01's redraw seed 7; its means agree
to within 0.05 sherd (rough 1.85 vs 1.90; see ticket 03 on run-to-run repeatability).
Across all 12 distinct presentations (1-10, 42, 123) rough 1× is above on 11, worn 4× on 10.

Looked at (`artifacts/rwlora/jug_t02.png`, best attempt on presentations 1 and 7 per model,
beside the conservator's reassembly): the 9/9s are whole Juglets lying on the reference;
on presentation 7 every model places the neck and jumbles the lower body. Rendered before
this read-out.

- [x] run, sacct recorded · [x] reconcile 0 · [x] table and sign counts · [x] rendered
- [ ] G1 updated (pending refute-finding)
