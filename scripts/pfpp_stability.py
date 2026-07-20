#!/usr/bin/env python3
"""T1b — cross-seed placement consistency of PF++ juglet layouts.

Discriminates F1 (prior hallucination: each seed a *different* plausible pot —
high per-pair relative-pose dispersion, low form-metric dispersion) from
F2 (genuine perception: the same specific arrangement each seed — low
per-pair dispersion).

Inputs: N PF++ inference sample dirs of the SAME object (different test_seed),
plus the pc npz they ran on.

Usage:
  python scripts/pfpp_stability.py \
      --run-dirs .../inference/t1b_seed41/0 ... t1b_seed45/0 \
      --pc-npz .../data/pc_data/juglet_deploy/val/00000.npz \
      --out logs/diagnostics/pfpp_t1b_<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pfpp_layout_probes import (  # noqa: E402
    apply, compactness, load_pfpp_layout, pair_metrics, part_diag, vesselness)


def rot_geodesic_deg(Ra, Rb):
    tr = np.trace(Ra.T @ Rb)
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--pc-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for rd in args.run_dirs:
        clouds, mats = load_pfpp_layout(rd, args.pc_npz)
        runs.append({"dir": str(rd), "clouds": clouds, "mats": mats})
    P = len(runs[0]["clouds"])
    S = len(runs)
    mean_diag = float(np.mean([part_diag(c) for c in runs[0]["clouds"]]))

    # per-seed form metrics
    form_rows = []
    for r in runs:
        posed = [apply(r["mats"][i], r["clouds"][i]) for i in range(P)]
        pm = pair_metrics(posed)
        v = vesselness(posed)
        form_rows.append({
            "run": Path(r["dir"]).parent.name,
            "compactness": compactness(posed),
            "coarse_pairs": pm["coarse_pairs"],
            "fine_pairs": pm["fine_pairs"],
            "sor_residual": v["sor_residual"],
            "profile_frac": v["profile_frac"],
        })

    # per-pair relative-pose dispersion across seeds
    pair_rows = []
    for i, j in combinations(range(P), 2):
        rels = []
        for r in runs:
            rel = np.linalg.inv(r["mats"][i]) @ r["mats"][j]
            rels.append(rel)
        rot_ds, trans = [], []
        for a, b in combinations(range(S), 2):
            rot_ds.append(rot_geodesic_deg(rels[a][:3, :3], rels[b][:3, :3]))
        for rel in rels:
            trans.append(rel[:3, 3])
        trans = np.stack(trans)
        t_disp = float(np.linalg.norm(trans.std(0)) / mean_diag)
        pair_rows.append({"pair": f"p{i+1:02d}{j+1:02d}",
                          "rot_dispersion_deg": float(np.mean(rot_ds)),
                          "trans_dispersion": t_disp})

    rot_all = [r["rot_dispersion_deg"] for r in pair_rows]
    t_all = [r["trans_dispersion"] for r in pair_rows]
    summary = {
        "n_seeds": S,
        "rot_dispersion_deg": {"mean": float(np.mean(rot_all)),
                               "median": float(np.median(rot_all)),
                               "min": float(np.min(rot_all)),
                               "max": float(np.max(rot_all))},
        "trans_dispersion": {"mean": float(np.mean(t_all)),
                             "median": float(np.median(t_all))},
        "form_metric_dispersion": {
            k: {"mean": float(np.mean([f[k] for f in form_rows])),
                "std": float(np.std([f[k] for f in form_rows]))}
            for k in ("compactness", "coarse_pairs", "sor_residual",
                      "profile_frac")},
        "per_seed_form": form_rows,
        "per_pair": sorted(pair_rows, key=lambda r: r["rot_dispersion_deg"]),
    }
    with open(args.out / "stability.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# T1b — PF++ cross-seed stability\n",
          f"{S} seeds, {P} parts.\n",
          "## Per-seed form metrics\n",
          "| run | compactness | coarse | fine | SoR resid | profile frac |",
          "|---|---|---|---|---|---|"]
    for f_ in form_rows:
        md.append(f"| {f_['run']} | {f_['compactness']:.3f} | {f_['coarse_pairs']} | "
                  f"{f_['fine_pairs']} | {f_['sor_residual']:.4f} | {f_['profile_frac']:.3f} |")
    md += ["\n## Relative-pose dispersion across seeds\n",
           f"rotation: mean {summary['rot_dispersion_deg']['mean']:.1f} deg, "
           f"median {summary['rot_dispersion_deg']['median']:.1f} deg "
           f"(min {summary['rot_dispersion_deg']['min']:.1f}, "
           f"max {summary['rot_dispersion_deg']['max']:.1f})",
           f"translation (over mean part diag): mean "
           f"{summary['trans_dispersion']['mean']:.3f}, median "
           f"{summary['trans_dispersion']['median']:.3f}\n",
           "| pair | rot disp (deg) | trans disp |", "|---|---|---|"]
    for r in summary["per_pair"]:
        md.append(f"| {r['pair']} | {r['rot_dispersion_deg']:.1f} | "
                  f"{r['trans_dispersion']:.3f} |")
    (args.out / "stability.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:20]))
    print(f"\nwrote {args.out}/stability.md")


if __name__ == "__main__":
    main()
