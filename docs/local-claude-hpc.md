# Local Claude Code + Spartan HPC (GARF)

Run Claude Code on your laptop. Use GitHub for code sync. Use Spartan only for Slurm jobs and heavy data under `/data/gpfs/projects/punim2657/GARF`.

```
Laptop                          GitHub (your fork)         Spartan
──────                          ─────────────────         ───────
Claude Code + local clone  ──push──►  repo  ──pull──►  code checkout
       │                                                    │
       └────────── ssh: sbatch / squeue / sacct ────────────┘
       ◄───────── rsync: small logs / metrics / samples ────┘
```

## One-time laptop setup

### 1. Install Claude Code

Use the CLI (`npm install -g @anthropic-ai/claude-code` or the native installer) or Cursor with a **local** folder open — not a Remote-SSH window into Spartan for the agent. Login follows your claude.ai account.

### 2. SSH connection multiplexing

Add to `~/.ssh/config` on the laptop:

```
Host spartan
    HostName spartan.hpc.unimelb.edu.au
    User zhuojiat
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

Authenticate once (`ssh spartan`); later `ssh`/`rsync`/`scp` reuse the multiplexed connection for ~8h (critical with MFA).

Verify:

```bash
ssh spartan hostname
ssh spartan hostname   # should be instant, no new MFA
```

### 3. Git: fork as origin, upstream = ai4ce

The HPC clone historically tracked `ai4ce/GARF`. For this workflow:

1. Create or use your fork (e.g. `zeejaytan/GARF`).
2. On the laptop: `git clone` that fork and open it in Claude Code / Cursor.
3. On Spartan:

```bash
cd /data/gpfs/projects/punim2657/GARF
git remote rename origin upstream   # if origin was ai4ce
git remote add origin git@github.com:zeejaytan/GARF.git
# or: git remote set-url origin <your-fork-url>
git remote add upstream https://github.com/ai4ce/GARF.git   # if not already
```

Thereafter: edit/commit/push on the laptop; on the cluster only `git pull --ff-only`. Do not push Spartan WIP to `ai4ce`.

### 4. What stays on HPC (never clone as source of truth)

Already gitignored: `input/`, `output/`, `logs`, `*.hdf5` / `*.h5`, checkpoints. `/progress` logs live in `.progress/` and can be versioned with git.

## Day-to-day loop

From the laptop clone:

```bash
git add … && git commit && git push
./scripts/remote/pull_and_sbatch.sh smoke_test.slurm
./scripts/remote/job_status.sh
./scripts/remote/fetch_artifacts.sh logs/some_run ./artifacts/
```

Environment overrides (optional):

- `SPARTAN_HOST` (default `spartan`)
- `REMOTE_ROOT` (default `/data/gpfs/projects/punim2657/GARF`)

## Trade-offs

- Inspecting large remote files goes through SSH (`ssh spartan 'tail -50 …/logs/….out'`) — slower than local reads, fine in practice.
- Interactive GPU debugging on a compute node is clunkier from a local agent; prefer batch jobs + fetched logs.
- Cluster CLAUDE.md policy is in this repo so laptop agents load it with the clone.
