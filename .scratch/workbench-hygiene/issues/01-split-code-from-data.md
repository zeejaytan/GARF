# 01: Split GARF's Spartan folder into GARF/repo plus a working area

**What to build:** The canonical checkout moves to `/data/gpfs/projects/punim2657/GARF/repo`; `input/`, `output/` and logs stay in `GARF/` as the working area, which gains `runs/` and `attic/`; every batch script's paths follow.

**Answers:** none (workbench routine).

**Blocked by:** None (can start immediately).

**Status:** resolved — code in GARF/repo, data in place, smoke job 33745395 passed (2026-10-24)

Amendment 2.5, the conservator's choice 2026-10-24 ("Split it (ticket)"). Today the repo directory is also the data directory, so it can never be re-cloned safely. Do it with no job running from it (`squeue`), move with `mv` on the same filesystem, fix paths in `slurm/` and `scripts/`, and record the move in the attic INDEX. Also untrack `.DS_Store` (2.4) and make every `--output` absolute (2.3).

- [x] Checkout at `GARF/repo`, fetch-only, clean
- [x] Data untouched in place; one smoke job runs from the new layout
- [x] `runs/` + `attic/` made; `AGENTS.md` paths updated

## Log

- 2026-10-24 survey (`/mnt/project-files/garfsplit_survey.txt`): checkout clean at 49abd65, 435
  tracked files, 32 tracked top-level entries, 223 untracked (data, logs, loose `*.out`/`*.ply`).
  No `GARF/.venv` (jobs use `CRAG/.venv`); nothing in that env points at GARF; no GARF jobs.
- Paths: 45 `slurm/` jobs set `GARF_WORK`/`GARF_REPO`, cd to the working area, call code by
  `$GARF_REPO`, absolute `--output` under `logs/slurm/`. Six analysis scripts read `GARF_WORK`
  for data. PuzzleFusion++ `eval_t2_wear`/`eval_t2t3_fix` call `$G/repo/scripts/`.
- "Fetch-only" read as: the checkout only takes commits from GitHub (`pull --ff-only`), never edited.
  Moving GARF jobs onto `run_from_commit.sh` is a later step; `GARF_REPO` is the one line to change.
- 2026-10-24 move (conservator's go-ahead in this thread; `/mnt/project-files/garfsplit_move.txt`):
  `.git` and the 32 tracked top-level entries renamed into `GARF/repo` with `mv`, nothing copied
  or deleted. HEAD 49abd65 -> 62b0a01 (ff-only), 0 porcelain lines before and after. Record:
  `GARF/attic/split-code-from-data-20261024-INDEX.txt`. The script stopped on a SIGPIPE after the
  move (`ls | head` under pipefail, since fixed); Spartan ran the last steps by hand as written.
- Smoke `slurm/smoke_test.slurm`, job 33745395, from `GARF/repo` at 62b0a01: COMPLETED 0:0 in
  2m38s, 5 epochs, log ends "SMOKE TEST COMPLETE" (`GARF/logs/slurm/smoke_test_33745395.out`).
  Index row `20261024-1952_62b0a01_split-smoke`. Verdict is left for a person.
- Left as found: ~200 loose `*.out`/`*.ply`/`*.obj` at the top of the working area (older Slurm
  logs and renders). They are data now, outside the code; tidying them is a separate pass.
