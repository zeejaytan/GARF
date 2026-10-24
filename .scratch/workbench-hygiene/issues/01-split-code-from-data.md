# 01: Split GARF's Spartan folder into GARF/repo plus a working area

**What to build:** The canonical checkout moves to `/data/gpfs/projects/punim2657/GARF/repo`; `input/`, `output/` and logs stay in `GARF/` as the working area, which gains `runs/` and `attic/`; every batch script's paths follow.

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** in-progress — paths fixed on main, Spartan move relayed

**Working in:** project thread "GARF code and data split" · Spartan move `garfsplit_move.sh` (2026-10-24)

Amendment 2.5, the conservator's choice 2026-10-24 ("Split it (ticket)"). Today the repo directory is also the data directory, so it can never be re-cloned safely. Do it with no job running from it (`squeue`), move with `mv` on the same filesystem, fix paths in `slurm/` and `scripts/`, and record the move in the attic INDEX. Also untrack `.DS_Store` (2.4) and make every `--output` absolute (2.3).

- [ ] Checkout at `GARF/repo`, fetch-only, clean
- [ ] Data untouched in place; one smoke job runs from the new layout
- [ ] `runs/` + `attic/` made; `AGENTS.md` paths updated

## Log

- 2026-10-24 survey (`/mnt/project-files/garfsplit_survey.txt`): checkout clean at 49abd65, 435
  tracked files, 32 tracked top-level entries, 223 untracked (data, logs, loose `*.out`/`*.ply`).
  No `GARF/.venv` (jobs use `CRAG/.venv`); nothing in that env points at GARF; no GARF jobs.
- Paths: 45 `slurm/` jobs set `GARF_WORK`/`GARF_REPO`, cd to the working area, call code by
  `$GARF_REPO`, absolute `--output` under `logs/slurm/`. Six analysis scripts read `GARF_WORK`
  for data. PuzzleFusion++ `eval_t2_wear`/`eval_t2t3_fix` call `$G/repo/scripts/`.
- "Fetch-only" read as: the checkout only takes commits from GitHub (`pull --ff-only`), never edited.
  Moving GARF jobs onto `run_from_commit.sh` is a later step; `GARF_REPO` is the one line to change.
