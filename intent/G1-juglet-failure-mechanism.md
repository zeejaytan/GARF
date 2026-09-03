# G1 — Why does GARF fail on the Juglet?

**Status:** open — three mechanisms ruled out, none found · **Blocked by:** none
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

Recording what has been ruled out is half the value here. Do not re-run these.

## Done when

- [ ] A named mechanism, stated as something measurable on a second object — not "domain
      gap", which is a label for not knowing
- [ ] **Or** an explicit decision to stop looking, with the ruled-out list written up as
      the finding: a well-evidenced "we could not find it" is publishable and a vague one
      is not
- [ ] The proposed assembly rendered at a view that shows whether the break faces meet —
      not a thumbnail of the whole pot, which looks equally wrong at every failure mode

## The trap this is most exposed to

The Juglet has **no correct reassembly to score against**. Every number is a proxy, and a
proxy can rank a wrong answer above a right one — a collapsed turntable solve once scored
a *better* reprojection error than the correct one. Whatever mechanism is proposed has to
survive being looked at, not only measured.

## Source

`docs/notes/JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md`, `JUGLET_ROOTCAUSE_FINDINGS.md`
(2026-07-09, addenda 07-10 and 07-13), `JUGLET_DEPLOY_INFERENCE_ANALYSIS.md`.
