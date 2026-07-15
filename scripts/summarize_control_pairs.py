#!/usr/bin/env python3
"""Exp 6b — summarize the control (known-good) pairwise-oracle runs and decide
whether the 2-piece proxy can register a true mate.

For each control pair across seeds: contact fraction + cross-seed relative-pose
dispersion (reusing no_gt_probes), joined with the GT true-mate label. Then
report true-mate vs non-mate separation, and the analogous Juglet numbers if the
Juglet adjacency + pairs.json are provided, for a side-by-side verdict.

Usage
-----
  python scripts/summarize_control_pairs.py \
      --run-dirs logs/deploy/exp6b_ctrl_<stamp>_s41 ..._s42 ..._s43 \
      --adjacency logs/diagnostics/control_ceramics_adjacency.json \
      --out logs/diagnostics/ctrl_pairs_<stamp> \
      [--juglet-adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
       --juglet-pairs logs/diagnostics/pairs_20260610_162659/pairs.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median, mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from no_gt_probes import contact_probe, stability_probe  # noqa: E402


def find_pair_samples(run_dir: Path) -> dict[str, Path]:
    """sample key '<obj>__p<ij>' -> dir holding predicted_assembly.{glb,json}."""
    root = run_dir / "version_0" / "assembly_results"
    out = {}
    for glb in sorted(root.glob("*/*__p*/predicted_assembly.glb")):
        out[glb.parent.name] = glb.parent
    return out


def fmt(x, nd=3):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def summarize_set(rows, key):
    vals = [r[key] for r in rows if r[key] == r[key]]
    return {"n": len(vals), "mean": mean(vals) if vals else float("nan"),
            "median": median(vals) if vals else float("nan"),
            "min": min(vals) if vals else float("nan"),
            "max": max(vals) if vals else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--adjacency", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--juglet-adjacency", type=Path)
    ap.add_argument("--juglet-pairs", type=Path)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--tau-frac", type=float, default=0.01)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    adj = json.load(open(args.adjacency))["pairs"]  # {key: {true_mate,...}}
    per_run = {d: find_pair_samples(d) for d in args.run_dirs}
    samples = sorted({s for m in per_run.values() for s in m})
    if not samples:
        sys.exit("no control pair assembly results found")
    print(f"{len(samples)} control pairs across {len(args.run_dirs)} runs")

    rows = []
    for s in samples:
        cfracs, jsons = [], []
        for d in args.run_dirs:
            adir = per_run[d].get(s)
            if adir is None:
                continue
            res = contact_probe(adir / "predicted_assembly.glb",
                                n_points=args.n_points, tau_frac=args.tau_frac)
            cfracs.append(res["pairs"][0]["contact_frac"])
            jsons.append(adir / "predicted_assembly.json")
        rot_disp = float("nan")
        if len(jsons) >= 2:
            rot_disp = stability_probe(jsons)["summary"]["mean_rel_rot_dispersion_deg"]
        label = adj.get(s, {})
        rows.append({
            "sample": s,
            "true_mate": bool(label.get("true_mate", False)),
            "gt_contact_frac": label.get("contact_frac", float("nan")),
            "n_runs": len(jsons),
            "pred_contact_frac": median(cfracs) if cfracs else float("nan"),
            "rot_dispersion_deg": rot_disp,
        })

    mates = [r for r in rows if r["true_mate"]]
    nonm = [r for r in rows if not r["true_mate"]]
    ctrl_mate_rot = summarize_set(mates, "rot_dispersion_deg")
    ctrl_non_rot = summarize_set(nonm, "rot_dispersion_deg")

    md = ["# Exp 6b — control (known-good ceramics) pairwise oracle\n",
          f"Runs: {', '.join(str(d) for d in args.run_dirs)}\n",
          f"Control pairs: {len(rows)} ({len(mates)} true mates, {len(nonm)} non-mates)\n",
          "## Control: rotation dispersion (deg), true mates vs non-mates\n",
          "| set | n | mean | median | min | max |",
          "|---|---|---|---|---|---|",
          f"| TRUE MATES | {ctrl_mate_rot['n']} | {fmt(ctrl_mate_rot['mean'],1)} | "
          f"{fmt(ctrl_mate_rot['median'],1)} | {fmt(ctrl_mate_rot['min'],1)} | {fmt(ctrl_mate_rot['max'],1)} |",
          f"| non-mates | {ctrl_non_rot['n']} | {fmt(ctrl_non_rot['mean'],1)} | "
          f"{fmt(ctrl_non_rot['median'],1)} | {fmt(ctrl_non_rot['min'],1)} | {fmt(ctrl_non_rot['max'],1)} |\n",
          "## Per-pair (sorted by rot dispersion)\n",
          "| pair | true mate | pred contact | rot disp (deg) |",
          "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["rot_dispersion_deg"]
                                         if r["rot_dispersion_deg"] == r["rot_dispersion_deg"] else 1e9)):
        md.append(f"| {r['sample']} | {'YES' if r['true_mate'] else ''} | "
                  f"{fmt(r['pred_contact_frac'])} | {fmt(r['rot_dispersion_deg'],1)} |")

    # ---- optional side-by-side with Juglet ----
    if args.juglet_adjacency and args.juglet_pairs:
        jadj = set(json.load(open(args.juglet_adjacency))["true_mates"])
        jpairs = json.load(open(args.juglet_pairs))

        def jkey(sample):
            return "p" + sample.split("/")[-1].split("-p")[-1]
        jrows = [{"true_mate": jkey(p["sample"]) in jadj,
                  "rot_dispersion_deg": p["rot_dispersion_deg"]} for p in jpairs]
        jm = summarize_set([r for r in jrows if r["true_mate"]], "rot_dispersion_deg")
        jn = summarize_set([r for r in jrows if not r["true_mate"]], "rot_dispersion_deg")
        md += ["\n## VERDICT: Juglet vs control, rot dispersion of TRUE MATES\n",
               "| dataset | true-mate mean | true-mate median | non-mate mean | separation |",
               "|---|---|---|---|---|",
               f"| control ceramics (works) | {fmt(ctrl_mate_rot['mean'],1)} | "
               f"{fmt(ctrl_mate_rot['median'],1)} | {fmt(ctrl_non_rot['mean'],1)} | "
               f"{fmt(ctrl_non_rot['mean']-ctrl_mate_rot['mean'],1)} |",
               f"| Juglet (fails) | {fmt(jm['mean'],1)} | {fmt(jm['median'],1)} | "
               f"{fmt(jn['mean'],1)} | {fmt(jn['mean']-jm['mean'],1)} |\n",
               "If control true mates are clearly LOWER dispersion than control",
               "non-mates AND than Juglet true mates -> the proxy works and Juglet",
               "is a real perception failure. If control shows no separation either",
               "-> the 2-piece proxy is invalid and Exp 6 is inconclusive.\n"]

    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    with open(args.out / "control_pairs.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"control TRUE MATES rot disp: mean {fmt(ctrl_mate_rot['mean'],1)} "
          f"median {fmt(ctrl_mate_rot['median'],1)}")
    print(f"control non-mates  rot disp: mean {fmt(ctrl_non_rot['mean'],1)}")
    print(f"wrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
