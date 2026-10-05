# 07: Does the random turn each sherd is handed in decide the Juglet result?

**What to build:** a knob that re-draws only the random turn applied to each sherd before
GARF sees it, keeping the sampled surface points exactly as presentation 42 drew them. Then
run worn 4× and the clean-break fine-tune on the Juglet with 10 new turn sets.

Ticket 03 showed the presentation decides the Juglet result (worn 4× mean per attempt
1.3-7.0 across 12 presentations, while new starts on one presentation stay within 4.50-4.65).
A presentation is two draws: sampled points and per-sherd turns. The points were tested on the
laptop and do not follow the score. The turns are not saved, so they need this run.

**Answers:** G1 (if the turn decides it, the Juglet result is mostly luck of orientation, not
something about worn breaks, and every arm comparison so far needs reading through that)
**Blocked by:** 03
**Status:** resolved (2026-10-05)

Knob: `++data.rot_seed=<n>` (test split only). Each sherd's turn comes from its own generator
seeded by (rot_seed, object index); the default turn is still drawn and discarded so the
point shuffle and everything after is untouched. Unset = today's behaviour. Slurm:
`ROT_SEEDS="none 1 … 10"`, run tag `_rs<n>`. Starts held fixed with `NOISE_SEEDS=1`.

Readings, fixed before results (worn 4×, mean sherds in own place per attempt, 20 attempts per
turn set; spread = standard deviation of the 10 turn-set means):
- Spread ≥ 1.2 sherds (presentations gave 1.8; new starts on one presentation ~0.1) →
  **the turn decides it.**
- Spread ≤ 0.4 and all 10 means within 3.5-5.8 → **the turn does not decide it**; the
  presentation effect then sits in the sampled points in a way join-edge coverage does not
  capture, or in the two together. The next step would be the mirror run: turns fixed,
  points re-drawn.
- In between → both share it; report the size of each.
Clean-break is read the same way, as a second training. It does not decide the reading alone.

Control: `rs none` with starts 1 must reproduce ticket 03's "presentation 42, starts 1"
(worn 4×: best 7 (2), mean 4.65; clean-break 7 (1), 2.85), bit-identical on the same GPU node,
within ≤0.15 sherd per attempt on another.

- [x] Knob added, default unchanged (synthetic check: default turns repeat, gt points and the
      numpy/python random streams untouched with rot_seed set; turns differ by seed and repeat)
- [x] Jobs run, sacct State/ExitCode recorded, reconcile 0 mismatches
- [x] Control reproduces ticket 03 (worn 7 (2) 4.65 exactly; clean-break 2.90 vs 2.85, other node)
- [x] Table of best and mean per turn set; render of one high and one low turn set before
      reporting (`artifacts/rwlora/jug_t07_turns_high_low.png`)
- [x] G1 updated with the result and the date (2026-10-05, after refute wf_2ae0daa1-d9b)

## Runs (2026-10-03)

- 32101397 clean-break: COMPLETED 0:0, gpgpu104, 6 min 43 s; reconcile 0 mismatches in all 11.
- 32101396 worn 4×: FAILED 1:0 after 5 s, `ARM: set ARM`. The settings never reached the
  job: `pull_and_sbatch.sh` re-quotes its arguments with `printf %q`, which broke the
  `--export=` value that holds spaces. Submit error, not a code or method fault. Resubmitted
  directly over ssh as **32143607**: COMPLETED 0:0, gpgpu107, 6 min 51 s, 2026-10-03;
  reconcile 0 mismatches in all 11.

Clean-break, interim (the reading rests on worn 4×). Sherds in own place per attempt, best of
20 (attempts reaching it) and mean:

| turns | none (control) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| own place | 7 (1) 2.90 | 5 (3) 2.90 | 6 (4) 4.00 | 8 (1) 3.75 | 4 (2) 1.80 | 7 (1) 3.45 | 5 (1) 2.50 | 4 (6) 2.80 | 5 (2) 2.85 | 5 (2) 2.60 | 6 (1) 2.85 |

- Control: ticket 03 gave 7 (1) 2.85 for these settings; 2.90 here, on another node, inside the
  ≤0.15 node tolerance.
- Spread of the 10 turn-set means: **0.64 sherd** (range 1.80-4.00). For comparison, the same
  model across 10 presentations: 1.18. New starts on one presentation: ~0.2.
- So for clean-break the turn carries roughly half the presentation effect: "in between".
Scores: `artifacts/rwlora/t07/jug_t07_scores.json`, script scratchpad `t07_score.py`.

## Worn 4× and reading (2026-10-05)

| turns | none (control) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| worn 4×, own place | 7 (2) 4.65 | 6 (1) 3.35 | 7 (6) 5.45 | 8 (1) 4.40 | 5 (2) 2.90 | 8 (2) 4.35 | 8 (1) 3.45 | 8 (2) 4.90 | 8 (3) 4.40 | **9 (1)** 4.40 | 7 (1) 3.30 |
| worn 4×, right way round | 5 (1) 2.35 | 3 (2) 1.55 | 5 (1) 2.55 | 6 (1) 2.50 | 2 (10) 1.50 | 3 (2) 1.60 | 4 (1) 1.60 | 4 (2) 2.15 | 5 (1) 2.25 | 5 (1) 2.50 | 3 (2) 1.60 |

- Spread of the 10 turn-set means: **0.81 sherd** (range 2.90-5.45). Across 10 presentations
  the same model spread 1.90; new starts on one presentation ~0.1. Part of the 0.81 is just
  20 attempts being a small sample (~0.35 per set mean), which the presentation figure carries too.
- **Pre-registered reading: in between.** 0.81 is above the 0.4 "turn does not decide it" line
  and below the 1.2 "turn decides it" line; four sets fall outside 3.5-5.8. Clean-break reads
  the same (0.64 against 1.18).
- The turn alone never pulled worn 4× below 2.9 per attempt; whole presentations reached 1.3.
  So the worst presentations need the sampled points, or the points and turns together, as
  well as the turn.
- Render (typical attempt of turn sets 2 and 4, both models): the neck/handle block seats on
  every one; what differs is the lower-body sherds, red on the low sets. Same place the
  presentation effect showed in ticket 03.

Worn 4× against clean-break, paired on the same turn set: worn higher on **11 of 11** (gap
0.45-2.10, mean 1.2 sherds per attempt; Wilcoxon p 0.001). Whether a turn set is good for one
model only loosely predicts whether it is good for the other (rank correlation +0.46, p 0.16
over the 11 sets). One pot, one training per model, one set of sampled points.

## Refute-finding (2026-10-05, wf_2ae0daa1-d9b): 0 refuted, 3 weakened, 2 stand

The numbers reproduce, the knob holds on the real data (true-pot points identical across all
22 runs to about 0.01 mm), and every run was scored against the same reassembly. Corrections,
each re-computed by the lead from `jug_t07_scores.json` and `jug_t06_rescore.json`:
- **"About half" overstated it.** 0.81 / 1.90 is a ratio of spreads. Removing the luck of 20
  attempts (about 0.35 sherd on a set mean) and comparing shares of the variation, the turn
  carries about **15%** of the presentation effect for worn 4× (0.53 of 3.55) and 25% for
  clean-break (0.33 of 1.33); on the right-way-round count 7% and 26%. The turn is a real but
  **minority** part; the sampled points, or points and turns together, carry most of it. With
  10 sets the spread's 95% range is about 0.56-1.47, so "in between" is not cleanly separated
  from "turn decides it" on the pre-registered line; the share-of-variation reading is firmer.
- **Paired, right way round:** worn 4× ahead on **10 of 11** (loses set 10 by 0.20), mean gap
  **0.46** sherd (p 0.006), against 11 of 11 and 1.2 on own place. Most of the own-place margin
  is sherds in place but turned. Five right-way-round gaps are ≤ 0.30, near the ≤ 0.15 node
  difference.
- **All 11 pairs share presentation 42's points**, the presentation worn 4× was chosen on (best
  of six strengths). On ticket 02's presentations the gap was small or reversed on three. So
  "ahead on every turn" holds on a favourable point set only.
- **The neck/handle block is the pinned anchor** (sherd 0, top 69% of the height, placed by
  construction). It seats every time because it is not tested; every figure counts it. Of the
  8 sherds in play, worn 4× places 1.9-4.45 per attempt across turn sets.
- The "~0.1 across new starts" yardstick is below the 0.35 that 20-attempt luck alone gives;
  the honest floor is ~0.35. The turn is still clearly real (one-way ANOVA p 2e-6).

Meaning for G1: the random input turn is a real but minority part of the presentation effect
(about 15% of it for worn 4×). Most of what a presentation does sits in the sampled points or
in points and turns together. On presentation 42's points, worn 4× stays ahead of clean-break
across turns, by about half a sherd per attempt the right way round.
