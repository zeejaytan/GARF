#!/usr/bin/env python3
"""T2a — per-part pose delta between two PF++ runs of the same object.

Compares the final assembled slot transforms of run A (e.g. original juglet)
and run B (e.g. de-weathered juglet) part by part, after removing the global
gauge (align by the anchor part). Gates (plan): mean rotation delta < 10 deg
and mean translation delta < 0.05 x mean part diagonal => output invariance.

Usage:
  python scripts/pfpp_pose_delta.py \
      --run-a <inference>/t4_none/0     --npz-a <pc_data>/juglet_deploy/val/00000.npz \
      --run-b <inference>/t2_dewear/0   --npz-b <pc_data>/juglet_dewear/val/00000.npz \
      --out logs/diagnostics/pfpp_t2a_<ts>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pfpp_layout_probes import load_pfpp_layout, part_diag  # noqa: E402


def rot_geodesic_deg(Ra, Rb):
    tr = np.trace(Ra.T @ Rb)
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--npz-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--npz-b", type=Path, required=True)
    ap.add_argument("--anchor", type=int, default=None,
                    help="anchor part index (default: from npz ref_part)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clouds_a, mats_a = load_pfpp_layout(args.run_a, args.npz_a)
    clouds_b, mats_b = load_pfpp_layout(args.run_b, args.npz_b)
    P = len(clouds_a)
    assert len(clouds_b) == P, "part count mismatch"

    if args.anchor is None:
        ref = np.load(args.npz_a, allow_pickle=True)["ref_part"]
        args.anchor = int(np.where(ref[:P])[0][0])

    # remove global gauge: express every slot in the anchor's frame
    inv_a = np.linalg.inv(mats_a[args.anchor])
    inv_b = np.linalg.inv(mats_b[args.anchor])
    rel_a = [inv_a @ m for m in mats_a]
    rel_b = [inv_b @ m for m in mats_b]

    mean_diag = float(np.mean([part_diag(c) for c in clouds_a]))
    rows = []
    for i in range(P):
        d = np.linalg.inv(rel_a[i]) @ rel_b[i]
        rot = rot_geodesic_deg(np.eye(3), d[:3, :3])
        # translation delta of the part centroid, gauge-fixed
        ca = (rel_a[i] @ np.r_[clouds_a[i].mean(0), 1])[:3]
        cb = (rel_b[i] @ np.r_[clouds_b[i].mean(0), 1])[:3]
        trans = float(np.linalg.norm(ca - cb) / mean_diag)
        rows.append({"part": i, "rot_delta_deg": rot, "trans_delta": trans,
                     "is_anchor": i == args.anchor})

    non_anchor = [r for r in rows if not r["is_anchor"]]
    summary = {
        "anchor": args.anchor,
        "rot_delta_deg": {"mean": float(np.mean([r["rot_delta_deg"] for r in non_anchor])),
                          "median": float(np.median([r["rot_delta_deg"] for r in non_anchor])),
                          "max": float(np.max([r["rot_delta_deg"] for r in non_anchor]))},
        "trans_delta": {"mean": float(np.mean([r["trans_delta"] for r in non_anchor])),
                        "median": float(np.median([r["trans_delta"] for r in non_anchor])),
                        "max": float(np.max([r["trans_delta"] for r in non_anchor]))},
        "gate_rot_lt_10deg": float(np.mean([r["rot_delta_deg"] for r in non_anchor])) < 10.0,
        "gate_trans_lt_0.05": float(np.mean([r["trans_delta"] for r in non_anchor])) < 0.05,
        "per_part": rows,
    }
    with open(args.out / "pose_delta.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = ["# T2a — PF++ pose delta (A vs B)\n",
          f"run A: {args.run_a}", f"run B: {args.run_b}\n",
          f"non-anchor rotation delta: mean {summary['rot_delta_deg']['mean']:.1f} deg, "
          f"median {summary['rot_delta_deg']['median']:.1f}, max {summary['rot_delta_deg']['max']:.1f}",
          f"non-anchor translation delta (/part diag): mean {summary['trans_delta']['mean']:.3f}, "
          f"median {summary['trans_delta']['median']:.3f}, max {summary['trans_delta']['max']:.3f}",
          f"gates: rot<10deg {'PASS' if summary['gate_rot_lt_10deg'] else 'FAIL'}, "
          f"trans<0.05 {'PASS' if summary['gate_trans_lt_0.05'] else 'FAIL'}\n",
          "| part | rot delta (deg) | trans delta | anchor |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['part']} | {r['rot_delta_deg']:.1f} | {r['trans_delta']:.3f} | "
                  f"{'*' if r['is_anchor'] else ''} |")
    (args.out / "pose_delta.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
