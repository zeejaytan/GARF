#!/usr/bin/env python3
"""Exp 7 — rim-erosion domain bridge summary (Juglet root-cause confirmation).

Aggregates, per erosion strength, the symmetry-invariant pairwise chamfer
(pair_reference_chamfer.py output) and the measured fracture-band relief
(recorded by build_control_pairs_hdf5.py --erode-strength) into one bridge
curve, and applies the decision rule from JUGLET_ROOTCAUSE_FINDINGS.md:

  If eroding a known-good ceramic's fracture band toward Juglet's worn relief
  level (~0.171 relief_p90) collapses its true-mate chamfer from ~0.024 toward
  Juglet's ~0.070 AND destroys the mate/non-mate separation, then worn-rim
  perception is confirmed as the failure mechanism.

Usage
-----
  python scripts/summarize_erosion_bridge.py \
      --entry 0.0  logs/diagnostics/pair_chamfer_erode000 logs/diagnostics/erode_adj_000.json \
      --entry 0.5  logs/diagnostics/pair_chamfer_erode050 logs/diagnostics/erode_adj_050.json \
      ... \
      --out logs/diagnostics/erosion_bridge_<stamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

JUGLET_RELIEF_P90 = 0.171   # fracture_sharpness_analysis.py, 2026-06-09
JUGLET_TRUE_MATE_CHAMFER = 0.070   # pair_chamfer_juglet, 2026-07-09
JUGLET_NON_MATE_CHAMFER = 0.073


def load_entry(strength: float, chamfer_dir: Path, adj_json: Path,
               labels: dict[str, bool] | None = None) -> dict:
    rows = json.load(open(chamfer_dir / "pairs.json"))
    if labels is not None:
        # Score every strength under one fixed label set (mate detection uses
        # surface sampling, so borderline pairs can flip between builds).
        for r in rows:
            r["true_mate"] = labels[r["sample"]]
    mates = [r["chamfer_over_diag"] for r in rows
             if r["true_mate"] and r["chamfer_over_diag"] == r["chamfer_over_diag"]]
    nons = [r["chamfer_over_diag"] for r in rows
            if not r["true_mate"] and r["chamfer_over_diag"] == r["chamfer_over_diag"]]
    adj = json.load(open(adj_json))
    reliefs = [p["relief_p90_eroded"] for obj in adj.get("relief", {}).values() for p in obj]
    reliefs_orig = [p["relief_p90_orig"] for obj in adj.get("relief", {}).values() for p in obj]
    return {
        "strength": strength,
        "relief_p90": mean(reliefs) if reliefs else float("nan"),
        "relief_p90_orig": mean(reliefs_orig) if reliefs_orig else float("nan"),
        "mate_mean": mean(mates) if mates else float("nan"),
        "mate_median": median(mates) if mates else float("nan"),
        "non_mean": mean(nons) if nons else float("nan"),
        "non_median": median(nons) if nons else float("nan"),
        "n_mates": len(mates),
        "n_nons": len(nons),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", nargs=3, action="append", required=True,
                    metavar=("STRENGTH", "CHAMFER_DIR", "ADJ_JSON"),
                    help="One erosion level: strength, pair_reference_chamfer out dir, "
                         "adjacency JSON with relief stats.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels-from", type=Path, default=None,
                    help="Adjacency JSON whose true_mate labels are applied to "
                         "EVERY entry (removes label noise between builds).")
    ap.add_argument("--x-label", default="erode strength",
                    help="Name of the swept variable for the table/decision text "
                         "(e.g. 'mollify radius / piece scale' for the Exp 7b "
                         "kernel-radius sweep). Entries are still sorted by the "
                         "numeric value passed to --entry.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = None
    if args.labels_from is not None:
        adj = json.load(open(args.labels_from))
        labels = {k: bool(v["true_mate"]) for k, v in adj["pairs"].items()}
    entries = [load_entry(float(s), Path(c), Path(a), labels) for s, c, a in args.entry]
    entries.sort(key=lambda e: e["strength"])

    md = ["# Exp 7 — rim-erosion domain bridge (control ceramics pairs)\n",
          "Erode the TRUE fracture contact band of known-good ceramics toward",
          f"Juglet's worn relief level (relief_p90 ~{JUGLET_RELIEF_P90}); GT pose preserved.",
          "Chamfer metric = symmetry-invariant assembled-pair chamfer / diagonal",
          "(pair_reference_chamfer.py), median over seeds 41/42/43.\n",
          f"| {args.x_label} | relief_p90 (band) | TRUE-mate chamfer mean/med | non-mate mean/med | mate/non separation |",
          "|---|---|---|---|---|"]
    for e in entries:
        sep = e["non_median"] / e["mate_median"] if e["mate_median"] > 0 else float("nan")
        md.append(f"| {e['strength']:.2f} | {e['relief_p90']:.3f} "
                  f"| {e['mate_mean']:.4f} / {e['mate_median']:.4f} "
                  f"| {e['non_mean']:.4f} / {e['non_median']:.4f} "
                  f"| {sep:.2f}x |")
    md.append(f"| — Juglet (measured) | {JUGLET_RELIEF_P90:.3f} "
              f"| {JUGLET_TRUE_MATE_CHAMFER:.4f} | {JUGLET_NON_MATE_CHAMFER:.4f} | 1.04x |\n")

    base = entries[0]
    worst = max(entries, key=lambda e: e["strength"])
    md.append("## Decision\n")
    checks = []
    checks.append((f"{args.x_label} 0 sanity reproduces baseline (~0.024 true-mate)",
                   abs(base["mate_median"] - 0.024) < 0.010 if base["strength"] == 0.0 else None))
    checks.append((f"true-mate chamfer degrades toward Juglet's 0.070 at max {args.x_label}",
                   worst["mate_median"] > 0.045))
    checks.append((f"mate/non-mate separation destroyed at max {args.x_label} (ratio < 1.3x)",
                   (worst["non_median"] / worst["mate_median"]) < 1.3
                   if worst["mate_median"] > 0 else None))
    for label, ok in checks:
        mark = "?" if ok is None else ("PASS" if ok else "FAIL")
        md.append(f"- [{mark}] {label}")
    confirmed = all(ok is True for _, ok in checks)
    md.append("")
    md.append("**VERDICT: worn-rim perception CONFIRMED as the Juglet failure mechanism.**"
              if confirmed else
              "**VERDICT: NOT confirmed — erosion did not reproduce the Juglet failure; "
              "revisit the mechanism (see JUGLET_ROOTCAUSE_FINDINGS.md).**")

    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    with open(args.out / "bridge.json", "w") as f:
        json.dump(entries, f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
