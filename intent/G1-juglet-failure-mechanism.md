# G1 — Why does GARF fail on the Juglet?

**Status:** open — four mechanisms ruled out, none found · **Blocked by:** none
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

## Break-face training: no Juglet lead (2026-10-01)

Fine-tuning GARF's placement stage (LoRA adapter, encoder untouched) on training breaks made
**rough** or **worn** looked like it placed more Juglet sherds than clean-break training. It
does not hold up when the 20 attempts start from different random positions. Scored against
the conservator's reassembly; a sherd counts as placed within 7% of pot size (~4.6 mm).

| Juglet, best of 20 (sherds per attempt, mean of 20) | starting set 42 | set 7 | set 123 |
|---|---|---|---|
| released GARF | 4 (2.2) | 3 (1.4) | 3 (1.6) |
| clean-break fine-tune | 6 (2.1) | 2 (1.2) | 5 (2.7) |
| rough 1× | 7 (5.2) | 4 (1.9) | 6 (2.9) |
| worn 4× | **9** (4.7) | 2 (1.3) | 4 (2.3) |

Every earlier run, including the repeat at a second training seed, used starting set 42. On
that set alone, rough and worn scored well. The conservator's look at the worn 4× 9/9 stands as
a look at that attempt: placement within ~3.3 mm, every sherd in roughly its place. But it is
not what the model usually does. On new starts it seats 2-4. Renders: `artifacts/rwlora/jug_arms_redraw.png`.
On new starts the lower-body sherds are jumbled and the same two (sherds 1 and 2) are swapped in
most attempts.

This is **the method failing**, not the ruler: the scores match the pictures, and the GARF and
own-place scorers agree (0 mismatches). What it rules out: rough or worn training of the
placement stage alone, at 1×-16×, is not what GARF is missing on the Juglet.

Also retracted: an apparent cost of every fine-tune on narrow_bottle1 (6 → 2 of 12) was the
starting positions. The gain on eroded Fractura pots (rough 1×-4×, +7-9 of 47 at heavy erosion)
was measured on starting set 42 only; its re-test is running (jobs 31835231-34).

Ticket: `.scratch/rough-worn-dose/issues/01-garf-rough-worn-dose-sweep.md`.

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
