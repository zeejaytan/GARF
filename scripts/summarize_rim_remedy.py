#!/usr/bin/env python3
"""Exp 8 — fracture-rim oversampling remedy summary (Juglet root-cause fix test).

The Juglet failure is a pairwise perception failure on worn fracture rims
(JUGLET_ROOTCAUSE_FINDINGS.md). Remedy under test: `data.rim_oversample_frac`
forces a fraction of each part's point budget onto the geometrically detected
fracture-rim band (relief-at-physical-radius detection, see
assembly/data/breaking_bad/base.py:rim_face_weights), giving the encoder more
of the only surface that carries mating signal on thin worn sherds.

Reads pair_reference_chamfer.py outputs (pairs.json) for:
  - Juglet pairs at each tested oversampling fraction (vs measured baseline
    true-mates 0.0731 / non-mates 0.0726 median at frac=0.0), and
  - control ceramics pairs regression (vs baseline true-mates 0.0242 /
    non-mates 0.0390 median) — the remedy must not break the working case.

Usage
-----
  python scripts/summarize_rim_remedy.py \
      --juglet-entry 0.35 logs/diagnostics/exp8_.../juglet_chamfer_f035 \
      --juglet-entry 0.60 logs/diagnostics/exp8_.../juglet_chamfer_f060 \
      --control-entry 0.35 logs/diagnostics/exp8_.../control_chamfer_f035 \
      --out logs/diagnostics/exp8_.../summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

# Measured baselines at rim_oversample_frac = 0.0 (2026-07-09):
JUGLET_BASE = {"mate_median": 0.0731, "non_median": 0.0726}
CONTROL_BASE = {"mate_median": 0.0242, "non_median": 0.0390}


def load(chamfer_dir: Path) -> dict:
    rows = json.load(open(chamfer_dir / "pairs.json"))
    mates = [r["chamfer_over_diag"] for r in rows
             if r["true_mate"] and r["chamfer_over_diag"] == r["chamfer_over_diag"]]
    nons = [r["chamfer_over_diag"] for r in rows
            if not r["true_mate"] and r["chamfer_over_diag"] == r["chamfer_over_diag"]]
    return {"mate_mean": mean(mates) if mates else float("nan"),
            "mate_median": median(mates) if mates else float("nan"),
            "non_mean": mean(nons) if nons else float("nan"),
            "non_median": median(nons) if nons else float("nan"),
            "n_mates": len(mates), "n_nons": len(nons)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--juglet-entry", nargs=2, action="append", default=[],
                    metavar=("FRAC", "CHAMFER_DIR"))
    ap.add_argument("--control-entry", nargs=2, action="append", default=[],
                    metavar=("FRAC", "CHAMFER_DIR"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.juglet_entry and not args.control_entry:
        ap.error("provide at least one --juglet-entry or --control-entry")
    args.out.mkdir(parents=True, exist_ok=True)

    md = ["# Exp 8 — fracture-rim oversampling remedy (pairwise oracle)\n",
          "Metric: symmetry-invariant assembled-pair chamfer / diagonal",
          "(pair_reference_chamfer.py), median over seeds 41/42/43.\n"]
    results = {"juglet": [], "control": []}

    md += ["## Juglet pairs (failure case — remedy target)\n",
           "| rim_oversample_frac | TRUE-mate mean/med | non-mate mean/med | non/mate separation |",
           "|---|---|---|---|",
           f"| 0.00 (baseline) | — / {JUGLET_BASE['mate_median']:.4f} "
           f"| — / {JUGLET_BASE['non_median']:.4f} "
           f"| {JUGLET_BASE['non_median'] / JUGLET_BASE['mate_median']:.2f}x |"]
    for frac, d in args.juglet_entry:
        r = load(Path(d))
        r["frac"] = float(frac)
        results["juglet"].append(r)
        sep = r["non_median"] / r["mate_median"] if r["mate_median"] > 0 else float("nan")
        md.append(f"| {float(frac):.2f} | {r['mate_mean']:.4f} / {r['mate_median']:.4f} "
                  f"| {r['non_mean']:.4f} / {r['non_median']:.4f} | {sep:.2f}x |")

    if args.control_entry:
        md += ["\n## Control ceramics pairs (working case — regression check)\n",
               "| rim_oversample_frac | TRUE-mate mean/med | non-mate mean/med |",
               "|---|---|---|",
               f"| 0.00 (baseline) | — / {CONTROL_BASE['mate_median']:.4f} "
               f"| — / {CONTROL_BASE['non_median']:.4f} |"]
        for frac, d in args.control_entry:
            r = load(Path(d))
            r["frac"] = float(frac)
            results["control"].append(r)
            md.append(f"| {float(frac):.2f} | {r['mate_mean']:.4f} / {r['mate_median']:.4f} "
                      f"| {r['non_mean']:.4f} / {r['non_median']:.4f} |")

    md.append("\n## Decision\n")
    if results["juglet"]:
        best = min(results["juglet"], key=lambda r: r["mate_median"])
        improved = best["mate_median"] < 0.9 * JUGLET_BASE["mate_median"]
        separated = (best["non_median"] / best["mate_median"] > 1.3
                     if best["mate_median"] > 0 else False)
        md.append(f"- Best Juglet frac {best['frac']:.2f}: true-mate median "
                  f"{best['mate_median']:.4f} (baseline {JUGLET_BASE['mate_median']:.4f}); "
                  f"[{'PASS' if improved else 'FAIL'}] >10% improvement; "
                  f"[{'PASS' if separated else 'FAIL'}] mate/non separation emerges (>1.3x)")
        if improved and separated:
            md.append("- **REMEDY EFFECTIVE at the pairwise level — proceed to full "
                      "9-piece Juglet assembly with this fraction.**")
        elif improved:
            md.append("- **Partial effect — alignment improves but mates are still not "
                      "distinguished; sampling alone is insufficient, fine-tuning on "
                      "eroded breaks is the next lever.**")
        else:
            md.append("- **No effect — the perception deficit is not point-starvation at "
                      "the rim; fine-tuning GARF on worn/eroded breaks (training-time "
                      "erode augmentation via fracture_mesh_ops.erode_fracture_band) is "
                      "the remaining remedy.**")
    if results["control"]:
        worst = max(results["control"], key=lambda r: r["mate_median"])
        ok = worst["mate_median"] < 0.035
        md.append(f"- Control regression: worst true-mate median {worst['mate_median']:.4f} "
                  f"[{'PASS' if ok else 'FAIL'}] stays well below non-mate level")

    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    with open(args.out / "remedy.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(md))
    print(f"\nwrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
