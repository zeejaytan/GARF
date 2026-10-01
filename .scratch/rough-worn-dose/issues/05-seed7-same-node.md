# 05: Seed-7 adapters over ten presentations, one GPU node

**What to build:** score the second training run (seed 7) of clean-break, rough 1× and worn 4×
on the same ten Juglet/Fractura presentations as ticket 02, all on one GPU node, so the
comparison repeats the *training*, not only the scoring. Same job, same node: re-score seed-42
clean-break and worn 4× on presentations 7, 9, 10 (worn's +0.15 / +0.20 wins and its one loss
in ticket 02, which ran clean and worn on different nodes, gpgpu111 / gpgpu122).

**Answers:** G1 (whether "rough training helps" is earned, or only "this rough adapter beats
this clean-break adapter")

**Blocked by:** 02, 03 (done)

**Status:** ready-for-agent

Fixed before results (2026-10-02), using the ticket-02 rule. Juglet sherds per attempt,
mean of 20, per presentation:
- Seed-7 rough (or worn) above seed-7 clean-break on **≥9/10 presentations, each by more
  than 0.15** → "rough (worn) training helps on the Juglet, small effect, two training runs".
- 7-8/10 → inconclusive; G1 keeps "one adapter beats one adapter".
- ≤6/10 → the seed-42 gain was that adapter's luck; G1 says rough training does not
  reliably help.
- Seed-42 worn vs clean on 7/9/10, same node: if worn is not ahead on both 7 and 9, worn's
  ticket-02 count becomes 7/10 and worn is written as no shown effect.
- Fractura reported per pot (narrow_bottle1 and narrow_bottle3 are the only pots that move).

- [ ] Job submitted, polled, final sacct State/ExitCode recorded; node recorded
- [ ] Reconcile 0 mismatches in every run
- [ ] Table + sign counts against the rule above
- [ ] G1 updated with the result and the date

Script: `slurm/rwlora_same_node.slurm` (calls `slurm/rwlora_arm.slurm` per arm).
