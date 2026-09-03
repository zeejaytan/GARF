# G4 — What can a conservator do with a proposal that has no answer key?

**Status:** open · **Blocked by:** [`../../intent/U1`](../../intent/U1-judging-without-answer-key.md)

## Why it matters

The normal archaeological case has **no correct reassembly to compare against**. The
benchmark path — `eval_complex`, part accuracy, RMSE against dataloader poses — does not
exist there, and the "Ground truth" panels in the visualisations are not ground truth,
which is a way to be confidently wrong.

So the deployable product is not a score. It is a **proposal a conservator can accept,
reject, or partly accept**, and we have not defined what that looks like.

## Done when

- [ ] The deploy path produces output judged only on the **prediction**, with no panel
      that could be mistaken for a reference answer
- [ ] Anchor-centring applied and verified — raw scan coordinates put fragments metres
      apart, and the models were trained on a local layout
- [ ] A stated way for a conservator to say "these two are right, that one is not",
      **per fragment** rather than per object: a 9-piece proposal with 7 right is useful,
      and a single number hides that
- [ ] Rendered at a view that shows whether adjacent break faces actually meet

## Gate

This inherits its criteria from `U1`. If `U1` has no measurable conservator criterion —
thickness continuity across a join, rim profile agreement, no interpenetration — then this
question cannot be closed, and it should say so rather than invent one.

## Source

`docs/notes/ARCHAEOLOGICAL_DEPLOYMENT.md`;
`../../intent/U1-judging-without-answer-key.md`.
