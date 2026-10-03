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
**Status:** in progress

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
- [ ] Jobs run, sacct State/ExitCode recorded, reconcile 0 mismatches
- [ ] Control reproduces ticket 03
- [ ] Table of best and mean per turn set; render of one high and one low turn set before
      reporting
- [ ] G1 updated with the result and the date

## Runs (2026-10-03)

- 32101397 clean-break: COMPLETED 0:0, gpgpu104, 6 min 43 s; reconcile 0 mismatches in all 11.
- 32101396 worn 4×: FAILED 1:0 after 5 s, `ARM: set ARM`. The settings never reached the
  job: `pull_and_sbatch.sh` re-quotes its arguments with `printf %q`, which broke the
  `--export=` value that holds spaces. Submit error, not a code or method fault. Resubmitted
  directly over ssh as **32143607** (pending).

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
