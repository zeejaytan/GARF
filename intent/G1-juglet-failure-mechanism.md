# G1 — Why does GARF fail on the Juglet?

**Status:** open — a lead (break-face training), no mechanism named yet · **Blocked by:** none
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

## Lead: what the break faces look like in training (2026-10-01)

Fine-tuning GARF's placement stage (LoRA adapter, encoder untouched) on breaks whose faces were
made **rough** or **worn** places more Juglet sherds than the same fine-tune on clean breaks.
Scored against the conservator's reassembly; a sherd counts as placed within 7% of pot size
(~4.6 mm on the Juglet).

- **Juglet, all 20 attempts:** rough 1× ~4.8 and worn 4× ~5.0 sherds per attempt, against ~2.8
  for clean-break training, at two training runs each. Rough and worn are not distinguishable.
- **Juglet, best attempt:** worn 4× seated all 9 once (8 of 9 at the second training run);
  rough 1× 7 of 9 at both; clean-break training 6; released GARF 4. The conservator looked at the
  9/9 in visual-qa: "all sherds in their sort of correct place … the best result we have so far
  on TORA or GARF. Not perfect, but a great lead." The look confirms placement, not that the
  break faces meet.
- **Eroded pots (8 Fractura pots, 47 sherds):** rough 1×-4× adds 7-9 sherds at heavy erosion
  (19 → 25-28) where worn adds 2-4. Numbers only: no reassembly from this ladder has been
  looked at.
- **Strength is bounded:** 8× already overshoots for both (rough 8× falls to 4/9 on the Juglet),
  and 16× did not improve on 8×. Training strengths that help are 1×-4×, which is 1-3× rougher
  or more open than the Juglet's own breaks (~0.17 mm gap).
- **Retracted:** an apparent cost of every fine-tune on narrow_bottle1 (6 → 2 of 12) came from the
  random starting positions, not the training. The same pot scored again by the same model gets
  4-7.

Weight: one real pot, one reference reading, the same 20 starting positions in every run.
Re-test from two new sets of starting positions is running (GARF jobs 31834501-04). It decides
whether worn 4× is singled out or whether the lead is "rough or worn training adds ~2 sherds".

This is not a named mechanism. It does not contradict "relief amplitude ruled out". That tested
whether altering the *test* sherds' relief explained the encoder's blindness. This changes the
*training* breaks seen by the placement stage. Ticket:
`.scratch/rough-worn-dose/issues/01-garf-rough-worn-dose-sweep.md`.

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
