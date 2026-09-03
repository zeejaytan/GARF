# G3 — Can GARF be the second architecture that tests U2?

**Status:** open · **Blocked by:** [G1](G1-juglet-failure-mechanism.md)
**Feeds:** [`../../intent/U2`](../../intent/U2-perception-or-placement.md)

## Why it matters

`U2` asks whether the difficulty with worn pottery is in **seeing** the break surface or
in **placing** the fragment once it is seen. That needs a second architecture: a result
from TORA alone cannot separate a property of worn pottery from a property of TORA.

GARF is the obvious candidate — it is here, it is pretrained, it is a different design.
The catch is that a second architecture is only useful as a control **if it works well
enough to show a difference.** A model that fails everything discriminates nothing.

## Done when

- [ ] A stated floor: the accuracy GARF must reach on our material before its numbers can
      separate perception from placement
- [ ] Measured against that floor on the material `U2` actually uses
- [ ] A verdict: GARF is the control, or a different second architecture is needed, or
      `U2` cannot be answered with the models we have

## Gate

If GARF cannot clear the floor, **say so and change `U2`** rather than running it anyway
and reporting two failures as a comparison. Two broken instruments do not average into one
working one.

## Source

`../../intent/U2-perception-or-placement.md`;
`docs/notes/GARF_vs_PuzzleFusion_comparison.md`.
