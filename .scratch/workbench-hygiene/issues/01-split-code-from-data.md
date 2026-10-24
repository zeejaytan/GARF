# 01: Split GARF's Spartan folder into GARF/repo plus a working area

**What to build:** The canonical checkout moves to `/data/gpfs/projects/punim2657/GARF/repo`; `input/`, `output/` and logs stay in `GARF/` as the working area, which gains `runs/` and `attic/`; every batch script's paths follow.

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

Amendment 2.5, the conservator's choice 2026-10-24 ("Split it (ticket)"). Today the repo directory is also the data directory, so it can never be re-cloned safely. Do it with no job running from it (`squeue`), move with `mv` on the same filesystem, fix paths in `slurm/` and `scripts/`, and record the move in the attic INDEX. Also untrack `.DS_Store` (2.4) and make every `--output` absolute (2.3).

- [ ] Checkout at `GARF/repo`, fetch-only, clean
- [ ] Data untouched in place; one smoke job runs from the new layout
- [ ] `runs/` + `attic/` made; `AGENTS.md` paths updated
