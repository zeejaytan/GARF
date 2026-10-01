# 03: Is it the presentation or the starting positions?

**What to build:** a way to vary the two kinds of randomness separately, then use it to show
which one moves the Juglet result. Today one eval seed fixes both:
- the presentation: surface points sampled per sherd and the orientation each is handed in,
  once per run;
- the 20 shuffled starting poses, one per attempt.

Add a second seed so the presentation and the starts can be set independently (e.g. seed the
data with the eval seed and reseed the start noise from `noise_seed` at test start).
Leaving `noise_seed` unset gives today's behaviour. Then run worn 4× and the clean-break
fine-tune on the Juglet in two arms:
- presentation 42 fixed, starts from 3 new seeds;
- starts fixed, presentations 7 and 123.

**Answers:** G1 (if the Juglet result depends mainly on how the sherds are sampled and
presented, that sensitivity is itself a candidate mechanism, measurable on a second object)
**Blocked by:** None (can start immediately; independent of 02)
**Status:** ready-for-agent

Readings, fixed before results:
- Presentation 42 fixed and new starts keep worn 4× near its old mean (≥ 4 sherds per
  attempt), while new presentations with fixed starts drop it to ~1-2 → **the presentation
  decides it**. Then render what differs between presentations 42 and 7 on the lower-body
  sherds (point density on the break faces, at a view resolving ~1 mm).
- Fixed presentation and new starts already drop it to ~1-2 → the 9/9 needed both a lucky
  presentation and lucky starts; no single factor.

- [ ] Knob added, default unchanged (rerun of presentation 42 / starts 42 reproduces ticket
      01's worn 4× Juglet counts exactly)
- [ ] Both arms run, sacct State/ExitCode recorded, reconcile 0 mismatches
- [ ] Table of means and bests per arm; render of the deciding comparison before reporting
- [ ] G1 updated with the result and the date

## Results (2026-10-01)

Knob: `++model.inference_config.noise_seed` (commit 05a501e) reseeds torch and numpy at the
top of `test_step`; the presentation stays with the eval seed. Jobs 31835696-99 COMPLETED 0:0,
reconcile 0 mismatches in all 12 runs.

Juglet, best of 20 (attempts) and mean per attempt:

| | presentation 42, original starts | 42, starts 1 | 42, starts 2 | 42, starts 3 | presentation 7, starts 1 | presentation 123, starts 1 |
|---|---|---|---|---|---|---|
| worn 4× | 9 (1) 4.65 | 7 (2) 4.65 | 8 (1) 4.55 | 7 (2) 4.50 | 3 (1) 1.20 | 4 (4) 2.40 |
| clean-break | 6 (1) 2.25 | 7 (1) 2.85 | 5 (2) 2.45 | 4 (3) 2.45 | 3 (3) 1.45 | 5 (2) 2.80 |

Pre-registered reading: new starts on presentation 42 keep worn 4× at ~4.5 per attempt;
presentations 7 and 123 with the starts held fixed drop it to 1.2 and 2.4 → **the
presentation decides it**. The top attempt (9/9 vs 7-8) did also need the original starts.

Acceptance, default unchanged: worn 4× rerun is **bit-identical** to ticket 01 (all 20
attempts, every sherd distance). Clean-break is **not**: same code path, adapter and data
files untouched since ticket 01, but attempt distances differ from attempt 0, three
attempts gain one sherd (mean 2.10 → 2.25). The knob is not the cause (unset = skipped,
and worn reproduced). Suspected GPU non-determinism; two identical clean-break reruns
(31879925-26) test it. Size so far: ≤ 0.15 sherd per attempt, an order below the presentation
effect.

- [x] Knob added; default reproduces worn 4× exactly, clean-break within 0.15 (see above)
- [x] Both arms run, sacct recorded, reconcile 0
- [x] Table
- [ ] Render of the deciding comparison: what differs between presentations 42 and 7 on the
      lower-body sherds (sampled points vs input orientation)
- [ ] G1 updated
