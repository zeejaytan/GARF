#!/usr/bin/env python3
"""Exp 6 — summarize the Juglet pairwise-oracle runs.

For each pair sample (artifact/Juglet-pXXYY) across seed runs:
  - contact probe on each seed's predicted_assembly.glb (min gap, contact
    fraction, interpenetration — meaningful now that meshes are watertight),
  - cross-seed relative-pose stability of the single pair (one rel pose per
    run; low dispersion + clean contact = the model found a repeatable mate).

A pair is flagged ``mated`` (heuristic, for ranking only — final judgment is
the GLB gallery) when across seeds it shows: median contact_frac >= 0.02,
median interpenetration <= 0.02, and rotation dispersion <= 15 deg.

Usage:
  python scripts/summarize_pairwise_oracle.py \
      --run-dirs logs/deploy/exp6_pairs_<stamp>_s41 logs/deploy/..._s42 ... \
      --out logs/diagnostics/pairs_<stamp>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent))

from no_gt_probes import contact_probe, stability_probe  # noqa: E402

MATED_CONTACT = 0.02
MATED_PEN = 0.02
MATED_ROT_DEG = 15.0


def find_pair_samples(run_dir: Path) -> dict[str, Path]:
    """sample name -> assembly dir holding predicted_assembly.{glb,json}."""
    root = run_dir / "version_0" / "assembly_results"
    out = {}
    for glb in sorted(root.glob("*/Juglet-p*/predicted_assembly.glb")):
        out[f"{glb.parent.parent.name}/{glb.parent.name}"] = glb.parent
    return out


def fmt(x, nd=3):
    if x is None or x != x:
        return "-"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                    help="logs/deploy/<experiment_name> dirs, one per seed.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-points", type=int, default=5000)
    ap.add_argument("--tau-frac", type=float, default=0.01)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_run = {d: find_pair_samples(d) for d in args.run_dirs}
    all_samples = sorted({s for m in per_run.values() for s in m})
    if not all_samples:
        sys.exit("no Juglet-p* assembly results found under the given run dirs")
    print(f"{len(all_samples)} pair samples across {len(args.run_dirs)} runs")

    rows = []
    for sample in all_samples:
        contact_fracs, pens, gaps = [], [], []
        jsons = []
        for d in args.run_dirs:
            adir = per_run[d].get(sample)
            if adir is None:
                continue
            res = contact_probe(adir / "predicted_assembly.glb",
                                n_points=args.n_points, tau_frac=args.tau_frac)
            pair = res["pairs"][0]  # 2 parts -> exactly one pair row
            contact_fracs.append(pair["contact_frac"])
            pens.append(pair["interpenetration_frac"])
            gaps.append(pair["min_gap_over_scale"])
            jsons.append(adir / "predicted_assembly.json")

        rot_disp = trans_disp = float("nan")
        if len(jsons) >= 2:
            stab = stability_probe(jsons)["summary"]
            rot_disp = stab["mean_rel_rot_dispersion_deg"]
            trans_disp = stab["mean_rel_trans_dispersion"]

        med_contact = median(contact_fracs) if contact_fracs else float("nan")
        valid_pens = [p for p in pens if p == p]
        med_pen = median(valid_pens) if valid_pens else float("nan")
        med_gap = median(gaps) if gaps else float("nan")
        mated = (
            med_contact == med_contact and med_contact >= MATED_CONTACT
            and med_pen == med_pen and med_pen <= MATED_PEN
            and rot_disp == rot_disp and rot_disp <= MATED_ROT_DEG
        )
        rows.append({
            "sample": sample,
            "n_runs": len(jsons),
            "median_contact_frac": med_contact,
            "median_interpen_frac": med_pen,
            "median_min_gap_over_scale": med_gap,
            "rot_dispersion_deg": rot_disp,
            "trans_dispersion": trans_disp,
            "mated_heuristic": mated,
        })
        print(f"  {sample}: contact {fmt(med_contact)} pen {fmt(med_pen)} "
              f"gap {fmt(med_gap, 4)} rotdisp {fmt(rot_disp, 1)} mated={mated}")

    # rank: most-likely-mated first (stable + touching + not interpenetrating)
    def rank_key(r):
        rd = r["rot_dispersion_deg"]
        cf = r["median_contact_frac"]
        return (not r["mated_heuristic"], rd if rd == rd else 1e9, -(cf if cf == cf else -1))

    rows.sort(key=rank_key)

    with open(args.out / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "pairs.json", "w") as f:
        json.dump(rows, f, indent=2)

    n_mated = sum(r["mated_heuristic"] for r in rows)
    md = [
        "# Exp 6 - Juglet pairwise oracle\n",
        f"Runs: {', '.join(str(d) for d in args.run_dirs)}\n",
        f"Pairs flagged mated (heuristic): **{n_mated} / {len(rows)}**",
        f"(contact>={MATED_CONTACT}, interpen<={MATED_PEN}, rot disp<={MATED_ROT_DEG} deg)\n",
        "Interpretation: if ZERO pairs are mated (incl. all true mating pairs)",
        "-> perception failure (no usable rim signal on Juglet sherds).",
        "If several true pairs mate but the 9-piece run fails -> joint-inference",
        "failure. Cross-check the flagged pairs against physical adjacency and",
        "the GLB gallery.\n",
        "| pair | runs | contact_frac | interpen | min_gap/scale | rot disp (deg) | trans disp | mated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['sample'].split('/')[-1]} | {r['n_runs']} "
            f"| {fmt(r['median_contact_frac'])} | {fmt(r['median_interpen_frac'])} "
            f"| {fmt(r['median_min_gap_over_scale'], 4)} | {fmt(r['rot_dispersion_deg'], 1)} "
            f"| {fmt(r['trans_dispersion'])} | {'YES' if r['mated_heuristic'] else ''} |"
        )
    report = args.out / "summary.md"
    report.write_text("\n".join(md) + "\n")
    print(f"\nmated pairs (heuristic): {n_mated}/{len(rows)}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
