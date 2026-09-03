# AGENTS.md — GARF (project)

Follow the workspace root **`../AGENTS.md`** (laptop ↔ GitHub ↔ Spartan) for all shared rules. This file only adds GARF-specific paths and domain notes.

## GARF paths

| Role | Value |
|------|--------|
| GitHub fork (`origin`) | `zeejaytan/GARF` |
| Upstream | `ai4ce/GARF` |
| Spartan checkout (`REMOTE_ROOT`) | `/data/gpfs/projects/punim2657/GARF` |
| SSH | `Host spartan`, user `zhuojiat` |
| Remote helpers | `scripts/remote/pull_and_sbatch.sh`, `job_status.sh`, `fetch_artifacts.sh` |
| Setup docs | `docs/local-claude-hpc.md`, `docs/laptop-agent-setup.md` |

Heavy data on Spartan only (gitignored): `input/`, `output/`, `logs`, `*.hdf5` / `*.h5`, checkpoints. Local rsync landing zone: `artifacts/`.

**Write rules:** new code → `scripts/` (or existing packages); Slurm → `slurm/` or `scripts/`; method notes → `docs/notes/`; fetched samples → `artifacts/` (not source); HPC paths → `CLAUDE.local.md`. Do not add files at the GARF root.

Typical loop:

```bash
git push origin HEAD
./scripts/remote/pull_and_sbatch.sh slurm/smoke_test.slurm
./scripts/remote/job_status.sh
./scripts/remote/fetch_artifacts.sh logs/some_run ./artifacts/
```

## Domain / debugging

Prefer probes over guesses (Hydra/OmegaConf, dataloader length, HDF5 splits). Do not invent config keys. Inspect Spartan inputs/logs over SSH; keep diffs minimal. Session notes may live under `docs/notes/`. If `graphify-out/GRAPH_REPORT.md` exists, use it for architecture questions.

## Agent skills

Configured here so this repo works when opened on its own, not only from the `C:\PR`
umbrella. The full text of each convention lives at the workspace root; these are the
parts an agent needs before it can act.

- **Issue tracker — local markdown.** One feature per directory: the spec at
  `.scratch/<feature>/spec.md`, tickets one per file at
  `.scratch/<feature>/issues/<NN>-<slug>.md`, numbered from `01` in dependency order.
  Every ticket carries an **`Answers:`** line naming the question in `intent/` it exists
  to settle -- `G1` for this project, `U6` for the workspace, or `none` for routine
  work. Conventions and the ticket template: `../docs/agents/issue-tracker.md`.
- **Triage labels.** `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`,
  `wontfix`, recorded as a `Status:` line near the top of the ticket. Details:
  `../docs/agents/triage-labels.md`.
- **Domain docs — single-context.** Three different things, kept apart: **this file** is
  how to work here and the traps; **`CONTEXT.md`** at the repo root is the glossary, and
  `/domain-modeling` creates it lazily when the first term is actually resolved — do not
  create it empty; **`../docs/glossary.md`** is the cross-project measurement vocabulary
  (`part_acc`, chamfer distance, best-of-N) and outranks any local redefinition. ADRs go
  under `docs/adr/`. Details: `../docs/agents/domain.md`.
- **Intent.** [`intent/`](intent/) holds what we are trying to establish and what would
  settle it -- prefix **`G`**, permanent, numbers never reused. `/to-intent` opens a
  question or writes a finished ticket's result back into one. Check the loop is wired
  with `python ../scripts/check_intent_links.py`.

**Do not run `/setup-matt-pocock-skills` in this repo.** It would replace the above with
its own defaults, and its ticket template has no `Answers:` line -- tickets would stop
being connected to the question they exist to answer, silently.
