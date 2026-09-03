# What we are trying to establish with GARF

**GARF is the pretrained reassembly model we did not build.** It arrived claiming
zero-shot generalisation from 1.9 million synthetic fractures (ICCV 2025,
arXiv:2504.05400), and on our material it does not deliver that. The questions here are
about **why**, and about what GARF can still be used for.

This folder is **state, not a log**. Edit a line when it turns out wrong; git holds the
history. The runs themselves live in [`../docs/notes/`](../docs/notes/).

Prefix **`G`**, permanent. Numbers are never reused. **G5 is next.**

| # | Question | Status | Blocked by |
|---|---|---|---|
| [G1](G1-juglet-failure-mechanism.md) | Why does GARF fail on the Juglet? | open — several mechanisms ruled out, none found | none |
| [G2](G2-does-piece-count-break-it.md) | Does piece count break it, or is that a coincidence? | open | none |
| [G3](G3-second-architecture-for-u2.md) | Can GARF be the second architecture that tests U2? | open | [G1](G1-juglet-failure-mechanism.md) |
| [G4](G4-deploy-without-ground-truth.md) | What can a conservator do with a GARF proposal that has no answer key? | open | `../../intent/U1` |

## What is established

| Claim | Weight | Source |
|---|---|---|
| GARF assembles synthetic Breaking Bad and real Fractura ceramics of six pieces or fewer well. | many objects | `docs/notes/GARF_vs_PuzzleFusion_comparison.md` |
| On Tray-000 — 40 real archaeological sherds — part accuracy was 2.5%: one sherd of forty. | 1 tray, 1 run | `docs/notes/SESSION_INSIGHTS.md` |
| On the 9-piece Juglet scan the shape does not close and fractured edges do not align. | 1 object, repeated | `docs/notes/JUGLET_ROOTCAUSE_FINDINGS.md` |
| Relief amplitude is **ruled out** as the Juglet mechanism (Exp 7b/9, 2026-07-13). | 1 object | ibid., second addendum |
| PF++ pseudo-GT label error is **ruled out** as an explanation of the gap. | 1 object | ibid. |
| Worn-rim erosion contributes but is **not sufficient**, and rim-oversampling does not remedy it (Exp 7/8, 2026-07-10). | 1 object | ibid., first addendum |
| Raw scan coordinates put fragments metres apart; the models were trained on a local layout, so anchor-centring on the largest sherd is required before any deploy run. | procedural | `docs/notes/ARCHAEOLOGICAL_DEPLOYMENT.md` |

## The thing to keep saying out loud

Every Juglet number is a **no-GT proxy**: the Juglet has no correct reassembly to score
against. A bad number there can mean the method failed, **or** that we measured it wrong,
and those lead to opposite decisions. That is why G1 is a root-cause question and not an
accuracy benchmark, and why it is gated on
[`U1`](../../intent/U1-judging-without-answer-key.md).

## Related

Workspace questions in [`../../intent/`](../../intent/) — `U1` (judging without an answer
key), `U2` (perception or placement), `U7` (what is actually being compared). The
comparison partners have their own folders: `../../tora/intent/`,
`../../puzzlefusion-plusplus/intent/`.
