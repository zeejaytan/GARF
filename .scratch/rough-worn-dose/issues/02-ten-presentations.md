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
