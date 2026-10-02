# G1 — Why does GARF fail on the Juglet?

**Status:** open — three mechanisms ruled out; rough/worn break-face training a small gain, presentation dominates; full right-way-round reassembly rare (1-7 of 1,400), never from clean-break; no mechanism found · **Blocked by:** none
**Effort:** the investigation is largely done; what remains is a decision about how much
more to spend

## Why it matters

GARF assembles broken objects of six pieces or fewer, synthetic and real, well. It fails
on a nine-piece archaeological pot. Something about **this** material breaks it, and until
we can name that thing we cannot say whether the problem is the software, our data, or the
way we are scoring it — and those three call for three different responses.

It also decides whether GARF is worth carrying further at all, which is
[G3](G3-second-architecture-for-u2.md).

## Ruled out so far

- **Relief amplitude** — the depth of surface texture on the break faces. Exp 7b/9,
  2026-07-13.
- **PF++ pseudo-GT label error** — borrowed reference poses being wrong is not what
  explains the gap. Same date.
- **Worn-rim erosion** contributes but is **not sufficient**, and over-sampling the rim
  does not fix it. Exp 7/8, 2026-07-10.

All three were scored on "true mates" taken from PF++'s stand-in assembly, before the
conservator's own reassembly existed (`juglet_gt`, 2026-08-10). Exp 9 found no touching band
under those poses at all. They stand as recorded, but they were never checked against the
correct reassembly, so a later result scored against `juglet_gt` outranks them.

Recording what has been ruled out is half the value here. Do not re-run these.

**Not ruled out: whole-vessel shape** (umbrella `U10`, closed 2026-09-25, tested on TORA).
A LoRA adapter trained on juglets shaped like the answer seated fewer sherds (1 of 9) than
the untouched model (3), but that fine-tune also damaged placement on its own choosing
vessels and on an unworn pot (galli_pot 8 → 2 of 10).
So U10 does **not** narrow this question toward the break edges. It is not a GARF result.

## Break-face training: a small gain for one rough adapter; presentation dominates (2026-10-01)

Fine-tuning GARF's placement stage (LoRA adapter, encoder untouched) on training breaks made
**rough** or **worn** instead of clean. Scored against the conservator's reassembly; a sherd
counts as placed within 7% of pot size (~4.6 mm). The largest sherd is pinned at its true
place and always counts, so the contest is over the other 8.

**What the model is handed matters far more than how it was trained.** A *presentation* is the
surface points sampled from each sherd and the orientation it is handed in. The same worn 4×
adapter seats 4.65 of 9 per attempt on one presentation and 1.20 on another; new random
starting positions on a fixed presentation move it by 0.15 at most (ticket 03). Across ten
presentations, every model's mean ranges over 4-6 sherds.

**Juglet, ten presentations × 20 attempts, paired** (ticket 02, jobs 31835633-36):

| | released GARF | clean-break | rough 1× | worn 4× |
|---|---|---|---|---|
| sherds per attempt, mean of 10 presentations (incl. pinned) | 2.04 | 2.46 | 3.87 | 3.54 |
| above clean-break | 4/10 (not paired: different point sample) | — | 9/10, smallest win +0.5 | 9/10, two wins of +0.15 / +0.20 |
| best attempt, own place (blind to turned sherds) | 6/9 | 8/9 | 9/9 (5 of 200) | 9/9 (5 of 200) |
| best attempt, right way round | 3/9 | 6/9 (1 of 200) | 9/9 (1 of 200) | 8/9 (2 of 200) |

- **Rough 1× beats clean-break on the Juglet** — about 1.4 more sherds per attempt, of 8
  that can move; it still wins 7 of 8 with the two 9/9 presentations removed. This is a
  **small** effect by the rule fixed before the run, not a reliable assembler.
- **Worn 4× is borderline.** Two of its nine wins are no bigger than the 0.15-sherd difference
  between GPU nodes, and the clean-break and worn jobs ran on different nodes (gpgpu111,
  gpgpu122). Counting only clear wins it is 8/10, which the rule calls inconclusive.
- **Clean-break fine-tuning alone does nothing** (above released GARF on 4/10).
- **The conservator's worn 4× "9/9" (presentation 42) is not a full reassembly.** Sherds 3, 5
  and 7 sit on their own home surfaces but are turned 143°, 169° and 179° on their own faces;
  only 2 of 9 are within 4.6 mm point for point (others are turned 17-60°). It also needed those particular starting
  positions: three new sets of starts on the same presentation gave a best of 7, 8 and 7.
- **Fractura gains sit on two pots.** Five of the eight pots are fully reassembled by every
  model. The gain is narrow_bottle1 (known to swing by up to 5 sherds with the presentation
  alone) and narrow_bottle3; narrow_bottle3 on its own is above clean-break on 8/10 (rough)
  and 9/10 (worn). The erosion ladder (same 8 pots, eroded) favours rough at heavy erosion on
  all 3 presentations tried, +7 to +11 of 47; no render of that gain exists yet.

**How much weight this bears:** one pot, and **one trained adapter per arm** — the ten
presentations repeat the scoring, not the training. Rough 1× and worn 4× were each the best
of six strengths on presentation 42; clean-break was not picked from a field. So what is
shown is "this rough adapter beats this clean-break adapter", not "rough training helps".
The repeat training (seed 7) was scored on presentation 42 only.

This is **the method**, not the ruler or the answer key: a threshold sweep from 3.5% to 14%
keeps the order, the GARF and own-place scorers agree (0 mismatches in 80 runs), and the
reference's known faults (sherd 7, 5-11°; joins overlapping 0.2-0.5 mm) do not decide which
sherds the arms gain. Checked by a five-skeptic refute (0 refuted, 4 narrowed, 1 stands).

**Second training run (seed 7, one GPU node, 2026-10-02, ticket 05):** the direction repeats
but does not clear the bar fixed beforehand (≥9/10 by more than 0.15). Rough is ahead of
clean-break on 8/10 (+0.84 per attempt); worn on 10/10 but by more than 0.15 on only 8. Both
read **inconclusive** by the rule, so the claim stays "this adapter beats that one". Across both
training runs rough is ahead on 17 of 20 presentations, worn on 19 of 20. The node worry is
retired for these runs: the seed-42 scores came back identical on a third node.
The typical (not best) attempt has been looked at (`jug_t05.png`, coloured by mm from home) and
agrees with the counts. **Not yet looked at:** the presentation-42 vs presentation-7 comparison
(ticket 03).

**Full reassembly, scored the right way round (2026-10-02, ticket 06).** The own-place count was
**broken for turned sherds** (measurement broken, not method): a sherd spun half a turn on its own
face still lies on its home surface and counted as placed. The literature's part accuracy has the
same blind spot. A sherd now also has to sit the right way round: its points a median under
4.6 mm from their own home points. Re-scoring all 1,400 saved attempts:
- Of 21 attempts that scored 9 of 9 the old way, **2 are full reassemblies** (rough seed 42,
  presentation 1; worn seed 7, presentation 4 — the latter by 0.15 mm on one sherd). 14 had a
  sherd turned 95-179°, 5 a sherd tilted 28-44° and 6-10 mm off.
- **Full reassembly is rare and its count depends on the ruler:** 2 of 1,400 at 4.6 mm, up to 7
  with a looser gate (turn under 45°). Not robust enough to compare arms on.
- **Near-complete is robust:** clean-break never seats 8 of 9 the right way round under any gate
  tried (≥7: 0-4 of 400); rough and worn reach ≥7 in 13-36 of 400 each. Released GARF's best is 3.
- **Where the gain sits:** 26 of the 28 ≥7 attempts are on presentations 1 and 4. The gain is
  what break-face training adds when the presentation is favourable, not a general lift.
- **Weight:** one pot. The seed-7 and seed-42 worn adapters place sherds almost identically
  from the same start (0.2-2.1 mm apart), so the second training run is a weak repeat, not an
  independent one. Five-skeptic refute (wf_21ba12d0-63c): 0 refuted, 4 narrowed, 1 stands; the
  wording here carries the narrowing.
- **Witnessed (2026-10-02):** a spun "9 of 9" beside a genuine one in visual-qa
  (`garf_juglet_spun_t06`, `garf_juglet_genuine_t06`). The conservator confirmed the spun
  sherd is wrong once told which it was (a blind first look passed both), so right way round
  matches what a conservator calls reassembled. Own place stays reported beside it as "the
  vessel's shape reads, sherds need manual adjustment".

Tickets: `.scratch/rough-worn-dose/issues/01`-`06`. Renders: `artifacts/rwlora/jug_t02.png`
(best attempts only), `jug_t05.png` (typical attempts, mm from home), `jug_arms_redraw.png`.

## Done when

- [ ] A named mechanism, stated as something measurable on a second object — not "domain
      gap", which is a label for not knowing
- [ ] **Or** an explicit decision to stop looking, with the ruled-out list written up as
      the finding: a well-evidenced "we could not find it" is publishable and a vague one
      is not
- [ ] The proposed assembly rendered at a view that shows whether the break faces meet —
      not a thumbnail of the whole pot, which looks equally wrong at every failure mode

## The trap this is most exposed to

The Juglet has **one reference: the conservator's own reassembly** (`juglet_gt`, 2026-08-10),
a single reading of a single pot. Results before it were scored against PF++'s stand-in. Every
number is still a proxy for whether the breaks meet, and a
proxy can rank a wrong answer above a right one — a collapsed turntable solve once scored
a *better* reprojection error than the correct one. Whatever mechanism is proposed has to
survive being looked at, not only measured.

## Source

`docs/notes/JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md`, `JUGLET_ROOTCAUSE_FINDINGS.md`
(2026-07-09, addenda 07-10 and 07-13), `JUGLET_DEPLOY_INFERENCE_ANALYSIS.md`.
