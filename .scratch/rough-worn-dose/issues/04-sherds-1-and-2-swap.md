# 04: Why are Juglet sherds 1 and 2 swapped?

**What to build:** across the 13 saved Juglet runs (260 attempts, every model), sherds 1 and 2
(lower body) are placed in each other's spots in 37 attempts (0-7 per run), at 32-38% of pot
size from home. The pair swaps under every model, so it does not come from rough or worn
training. First check whether this pair swaps more than any other pair. Only if it does, find
out whether the two sherds are genuinely look-alikes: similar size, wall curvature and
break-edge outline. If they are, GARF is not making a wild error. It is facing an ambiguity
a person might also face.

**Answers:** G1 (a named, measurable candidate: look-alike sherds whose break edges do not
tell them apart. It is testable on a second object by finding a look-alike pair there)
**Blocked by:** None (can start immediately)
**Status:** ready-for-agent
**Needs-eye:** a visual-qa pair of sherds 1 and 2 side by side, and one swapped attempt
beside the conservator's reassembly; stage via `visual-qa/viewer/stage_pair.py` (desc not yet written)

- [ ] Count every swapped pair across all saved Juglet attempts. **Stop here** with one line
      in G1 if 1-2 is not clearly the most frequent swap
- [ ] Measure both sherds: area, longest dimension, wall curvature, break-edge length, in mm
- [ ] Stage the pair; the conservator says whether they could be told apart by hand, and how
- [ ] Look-alikes → write the candidate into G1 with the measurement that defines it. Not
      look-alikes → one line in G1 saying the swap is not a shape ambiguity
