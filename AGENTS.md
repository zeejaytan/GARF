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

Typical loop:

```bash
git push origin HEAD
./scripts/remote/pull_and_sbatch.sh slurm/smoke_test.slurm
./scripts/remote/job_status.sh
./scripts/remote/fetch_artifacts.sh logs/some_run ./artifacts/
```

## Domain / debugging

Prefer probes over guesses (Hydra/OmegaConf, dataloader length, HDF5 splits). Do not invent config keys. Inspect Spartan inputs/logs over SSH; keep diffs minimal. Session notes may live under `docs/notes/`. If `graphify-out/GRAPH_REPORT.md` exists, use it for architecture questions.
