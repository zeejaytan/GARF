# 06: Full reassembly, scored the right way round

**What to build:** the own-place scorer also says, per sherd, how far it is **turned** from
home (degrees) and how far its points sit from their own home points (mm), and counts a sherd
only when it is in its own place **and** facing the right way. Every saved Juglet attempt
(ticket 02 and ticket 05, 1,400 attempts) is re-scored on the CPU, and the headline becomes
**full reassembly**: how many attempts seat all 9 sherds the right way round, with ≥8 and ≥7
beside it, reported as the best of N with how many attempts reached it.

**Answers:** G1 (whether break-face training brings the Juglet closer to a whole pot)

**Blocked by:** 05 (done)

**Status:** ready-for-agent

**Needs-eye:** a spun-sherd "9/9" beside a genuine 9/9, staged in visual-qa; the conservator's
note decides whether the new count matches what a conservator calls reassembled.

Why: the current count (chamfer under tolerance, as in the papers' part accuracy) passes a
sherd spun half a turn on its own face; of 21 scored 9/9s, 14 had a sherd turned 95-179° and 5 a
sherd tilted 28-44°, and the separate 9/9 the conservator saw had three sherds spun (ticket 05,
last section). Measurement broken, not method.

The rule, fixed before re-scoring:
- **Right way round** = own place AND median distance of the sherd's points from their own
  home points < the seating tolerance (7.07% of pot size, 4.6 mm on the Juglet). Points
  correspond one-to-one because each predicted sherd is the true sherd moved rigidly.
- Turn angle (rotation that best maps home points onto placed points) is reported, not gated.
- Default output of the scorer unchanged; the new fields are added beside it.

- [x] Scorer reports turn (°) and point distance (% / mm) per sherd, and the right-way-round
      count; hand-built tests: a sherd spun 180° on its own face stays "own" and is not
      right way round; a 5° turn is; the 7.3× bigger pot gives the same answer
      (tora `scripts/own_place.py` fields `oriented`, `point_pct`, `turn_deg`;
      `scripts/test_own_place.py` all pass, 2026-10-02)
- [x] Re-score of all 1,400 saved attempts matches ticket 05's table (two genuine 9/9s)
      (`artifacts/rwlora/jug_t06_rescore.json`; identical counts, rough s42 mean 2.55)
- [ ] visual-qa pairs staged (done 2026-10-02: `garf_juglet_spun_t06` = "A", worn s42
      presentation 1 attempt 19, sherd 7 turned 174°; `garf_juglet_genuine_t06` = "B", rough
      s42 presentation 1 attempt 19, every sherd ≤13°; surface coincidence 0.22 / 0.19 mm).
      Titles are the same question on purpose, so the eye is not told which is which.
      Conservator note read back and replied to: pending
- [x] Glossary entry and lesson written (umbrella `docs/glossary.md` "own place / right way round", "full reassembly"; `docs/lessons.md` "9 of 9 with three sherds facing the wrong way")
- [x] G1 updated after refute-finding wf_21ba12d0-63c (0 refuted, 4 narrowed, 1 stands), 2026-10-02
