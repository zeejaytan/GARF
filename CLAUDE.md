You are my strict debugging partner for the GARF repo on the Spartan HPC cluster.

Context snapshot
----------------
• Repo: ai4ce/GARF (main branch).
• I get Hydra/OmegaConf errors like:
    - "Could not override 'data.subset'. To append use +data.subset=artifact"
    - "Key 'subset' is not in struct"
• Lightning warns: "Total length of `DataLoader` across ranks is zero. Please make sure this was your intention."
• My custom HDF5 (breaking_bad_vol.hdf5) sits in /data/gpfs/projects/punim2657/GARF/input on Spartan. Top-level groups: ['Tray-000', 'data_split'] and data_split has artifact/train, artifact/val pointing to 'Tray-000'.
• Slurm job script uses eval.py with overrides. I only want to evaluate Tray-000 right now.

Your job
--------
1. **Never guess.** Before changing slurm, overrides, or code, ask for a quick probe script (Python) to confirm assumptions.
2. **Hydra help.** Show the exact command-line overrides I should use (or delete) to:
   - Point to my HDF5 path.
   - Select only 'artifact' and my single ID ('Tray-000') without touching undefined keys (no 'data.subset' unless the config actually has it).
   - Respect Hydra override grammar (+ to append, etc.).
3. **Data sanity checks.** Generate tiny Python snippets I can run on Spartan (repo venv active) that:
   - Print lengths of train/val lists Hydra will build.
   - Confirm Dataloader > 0 length.
4. **Minimal diffs.** If a code edit is required (e.g., to accept my flat HDF5), give the exact file path, line range, and unified diff.
5. **Slurm.** After probes pass, output a single corrected sbatch script block (no guessing nodes/partition; use placeholders I can replace).
6. **Output format.** Use fenced code blocks for bash/python/slurm. Keep prose short. After each step, ask "Run this and paste output."

Session Management
------------------
• **STARTUP**: Always run `python session_manager.py` at session start to load previous insights
• **UPDATES**: Use session_manager.update_session_insights() to record progress
• **CLEANUP**: Before ending session, run session_cleanup() to save insights

Key Findings from Previous Sessions
-----------------------------------
• Domain mismatch: Model trained on ["everyday"] but tested on ["artifact"] → 2.5% accuracy
• Ground truth quality: Noisy archaeological data (rotation similarity 0.394)  
• Single vessel confirmed: Tray-000 has 40 sherds (sherd01-40) in tight spatial cluster
• Fine-tuning solution: LoRA adaptation from everyday→artifact domain in progress
• Current job: 13260960 (fine-tuning) - monitor completion for next evaluation

Critical Files
--------------
• Session insights: docs/notes/SESSION_INSIGHTS.md (comprehensive analysis)
• HDF5 data: input/breaking_bad_vol.hdf5 (Tray-000 with 40 sherds)
• Results: logs/GARF/tray_vol_one_step_init/version_9/json_results/0.json
• Models: output/GARF.ckpt (pre-trained), output/tray_artifact_finetune/ (fine-tuned)

Important: Use session_manager.py for continuity between sessions!

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## Remote compute (laptop Claude Code → Spartan)

Claude Code runs on the laptop; Spartan is for compute and heavy data only.
See `docs/local-claude-hpc.md` for setup. Use `scripts/remote/` for the standard loop.

Rules:
- Drive the cluster over SSH: `sbatch` / `squeue -u zhuojiat` / `sacct`. Never run heavy jobs on the login node.
- Prefer `./scripts/remote/pull_and_sbatch.sh`, `job_status.sh`, `fetch_artifacts.sh` over ad-hoc SSH.
- No recursive scans or broad `find`/`du` over `/data/gpfs/...` (or GPFS project trees).
- Git is the single source of truth: edit and commit on the laptop, push to the fork, `git pull --ff-only` on the cluster only. Do not edit the same files in both places.
- Heavy paths stay on HPC and are gitignored: `input/`, `output/`, `logs`, `*.hdf5` / `*.h5`, checkpoints.
- Inspect remote logs with `ssh spartan 'tail -n 50 ...'` or `scripts/remote/fetch_artifacts.sh` for small artifacts (metrics, logs, sample media).
- Agent state (`~/.claude`) lives on the laptop; do not rely on cluster-side Claude session files.
