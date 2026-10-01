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
