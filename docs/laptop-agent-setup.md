# Laptop agent brief: GARF local Claude + Spartan

**Instructions for an agent running on the laptop.** Repo-side work is already done on Spartan under `/data/gpfs/projects/punim2657/GARF` (`CLAUDE.md` remote-compute section, `docs/local-claude-hpc.md`, `scripts/remote/*`, `artifacts/` in `.gitignore`). Do **not** re-implement those from scratch — pull them via git after the fork is wired.

## Goal

Claude Code / Cursor runs against a **local** GARF clone. Heavy data and Slurm stay on Spartan. GitHub (your fork) is the bridge. Agent state stays in laptop `~/.claude`.

```
Laptop                          GitHub fork                 Spartan
──────                          ───────────                 ───────
Claude Code + local clone  ──push──►  zeejaytan/GARF  ──pull──►  /data/gpfs/.../GARF
       │                                                              │
       └────────── ssh (ControlMaster): sbatch / squeue / sacct ──────┘
       ◄────────── rsync: small logs / metrics / samples ─────────────┘
```

## Preconditions

- GitHub account that can own fork `zeejaytan/GARF` (or user provides existing fork URL)
- Spartan SSH as `zhuojiat` (MFA) — user may need to type passphrase/MFA
- Laptop has `ssh`, `rsync`, `git`; Claude Code CLI and/or Cursor

Mark each todo complete as you finish it. Stop and ask the user if a step needs credentials, MFA, or a commit of unrelated WIP.

## Todo checklist

1. SSH multiplexing for `Host spartan`
2. Ensure GitHub fork exists
3. Point Spartan remotes at fork; push branch with laptop-HPC docs/scripts
4. Clone fork locally; open as local folder (not Remote-SSH)
5. Verify Claude Code / Cursor local workspace
6. Smoke-test `scripts/remote/*` over SSH

---

### 1. SSH multiplexing

Edit `~/.ssh/config` on the laptop. If a `Host spartan` block already exists, **merge** ControlMaster settings into it (do not duplicate Host entries):

```
Host spartan
    HostName spartan.hpc.unimelb.edu.au
    User zhuojiat
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

Verify (user completes MFA on first call):

```bash
ssh spartan hostname
ssh spartan hostname   # must be instant / no second MFA
```

If the second call re-prompts, fix ControlPath/config before continuing.

---

### 2. GitHub fork

```bash
# If fork missing:
gh repo fork ai4ce/GARF --clone=false
# Prefer SSH:
# git@github.com:zeejaytan/GARF.git
ssh -T git@github.com
```

If the fork URL differs, substitute it everywhere below and tell the user the URL you used.

---

### 3. Seed the fork from Spartan (once)

Inspect first (read-only):

```bash
ssh spartan 'cd /data/gpfs/projects/punim2657/GARF && git remote -v && git status -sb && git branch -vv'
```

Point `origin` at the fork; keep `upstream` = ai4ce:

```bash
ssh spartan 'cd /data/gpfs/projects/punim2657/GARF && \
  (git remote get-url upstream >/dev/null 2>&1 || git remote rename origin upstream) && \
  git remote remove origin 2>/dev/null || true; \
  git remote add origin git@github.com:zeejaytan/GARF.git; \
  git remote add upstream https://github.com/ai4ce/GARF.git 2>/dev/null || true; \
  git remote -v'
```

If Spartan cannot push to GitHub via SSH, use HTTPS from Spartan or push from the laptop after fetching another way — ask the user.

Push the branch that contains `docs/local-claude-hpc.md` and `scripts/remote/`:

```bash
ssh spartan 'cd /data/gpfs/projects/punim2657/GARF && git push -u origin HEAD'
```

If those files exist only as **uncommitted** changes on Spartan: **ask the user before committing**. Do **not** push to `upstream` / `ai4ce`. Do not force-push unless the user explicitly asks.

---

### 4. Local clone

```bash
git clone git@github.com:zeejaytan/GARF.git ~/code/GARF
cd ~/code/GARF
git remote add upstream https://github.com/ai4ce/GARF.git 2>/dev/null || true
chmod +x scripts/remote/*.sh
```

Open **`~/code/GARF` as a local folder** in Cursor / Claude Code — not a Remote-SSH window into Spartan for the coding agent.

Confirm after clone:

- [ ] `docs/local-claude-hpc.md`
- [ ] `scripts/remote/pull_and_sbatch.sh`
- [ ] `scripts/remote/job_status.sh`
- [ ] `scripts/remote/fetch_artifacts.sh`
- [ ] `CLAUDE.md` has a "Remote compute" section
- [ ] `.gitignore` lists `artifacts/`

If missing, fork was not seeded — return to step 3.

---

### 5. Claude Code / Cursor

- Install if needed: `npm install -g @anthropic-ai/claude-code` or native installer; or use Cursor on the local folder.
- Login with the user’s claude.ai account (model access follows the account).
- Agent transcripts/history live under laptop `~/.claude` — expected and desired.

---

### 6. Smoke-test remote helpers

From `~/code/GARF` with multiplexed SSH already up:

```bash
./scripts/remote/job_status.sh
```

Optional (only if user OK’s submitting a real Slurm job):

```bash
./scripts/remote/pull_and_sbatch.sh slurm/smoke_test.slurm
./scripts/remote/job_status.sh
# After completion, e.g.:
# ./scripts/remote/fetch_artifacts.sh logs ./artifacts/
```

Defaults: `SPARTAN_HOST=spartan`, `REMOTE_ROOT=/data/gpfs/projects/punim2657/GARF`.

If `git pull --ff-only` on Spartan fails (diverged history), **stop and report** — do not force-pull.

---

## Day-to-day rules (for this and future laptop agents)

- Edit/commit only on the laptop → push fork → Spartan `git pull --ff-only` only.
- Never edit the same tracked files on both sides.
- Heavy paths stay on HPC (`input/`, `output/`, `logs`, `*.hdf5`/`*.h5`, checkpoints) — do not clone them as source of truth.
- Prefer `scripts/remote/*`; no recursive `find`/`du` over `/data/gpfs`.
- `.progress/` may be committed; `artifacts/` is local-only (gitignored).

## Done when

- [ ] Second `ssh spartan` is instant (multiplex OK)
- [ ] Fork is `origin` on laptop and Spartan; `upstream` = ai4ce
- [ ] Local clone has remote helpers; opened as local workspace
- [ ] `./scripts/remote/job_status.sh` works
- [ ] User told about any optional smoke `sbatch` and any Spartan WIP still not on the fork

## Out of scope

- Rewriting Slurm scripts
- Puzzlefusion / other projects
- Committing unrelated Spartan WIP unless user asks
- Force-push to `ai4ce/GARF`
