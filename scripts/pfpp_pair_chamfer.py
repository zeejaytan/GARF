#!/usr/bin/env python3
"""T3 — pairwise mating oracle for PF++, scored with the Exp 6 instrument.

Runs the same symmetry-invariant registered-chamfer metric that indicted GARF
(pair_reference_chamfer.register_chamfer) on PF++ 2-piece denoiser-only runs.

  juglet : predicted pair layouts vs the PF++ 9-piece pseudo-GT reference
           (pieces i,j posed by the deploy layout) — self-consistency is the
           point: does PF++ reproduce ITS OWN 9-pc relative placement when
           given only the two sherds?
  control: predicted pair layouts vs the real GT (the stored assembled
           coordinates in the pair npz). Mate labels derived from the GT
           geometry itself (same thresholds as derive_pfpp_adjacency).

Usage:
  python scripts/pfpp_pair_chamfer.py juglet \
      --pairs-data $PF/data/pc_data/t3_juglet_pairs \
      --inference-dirs $PF/output/.../inference/t3_juglet_s41 ..._s42 ..._s43 \
      --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
      --ref-pfpp-dir $PF/output/.../inference/juglet_deploy/0 \
      --ref-npz $PF/data/pc_data/juglet_deploy/val/00000.npz \
      --out logs/diagnostics/pfpp_t3_juglet

  python scripts/pfpp_pair_chamfer.py control \
      --pairs-data $PF/data/pc_data/t3_control_pairs \
      --inference-dirs $PF/output/.../inference/t3_control_s41 ..._s42 ..._s43 \
      --out logs/diagnostics/pfpp_t3_control
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from pair_reference_chamfer import register_chamfer  # noqa: E402
from pfpp_layout_probes import apply, load_pfpp_layout  # noqa: E402


def derive_mate_label(pc0, pc1, gap_tau=0.03, contact_tau=0.03):
    allp = np.concatenate([pc0, pc1], 0)
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    t0, t1 = cKDTree(pc0), cKDTree(pc1)
    d01 = t1.query(pc0)[0]
    d10 = t0.query(pc1)[0]
    min_gap = float(min(d01.min(), d10.min())) / diag
    cfrac = float(max(np.mean(d01 < contact_tau * diag),
                      np.mean(d10 < contact_tau * diag)))
    return min_gap < gap_tau and cfrac > contact_tau


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["juglet", "control"])
    ap.add_argument("--pairs-data", type=Path, required=True)
    ap.add_argument("--inference-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--adjacency", type=Path, default=None)
    ap.add_argument("--ref-pfpp-dir", type=Path, default=None)
    ap.add_argument("--ref-npz", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.pairs_data / "manifest.json").read_text())

    true_mates = None
    if args.mode == "juglet":
        adj = json.loads(args.adjacency.read_text())
        true_mates = set(adj["true_mates"])
        ref_clouds, ref_mats = load_pfpp_layout(args.ref_pfpp_dir, args.ref_npz)
        ref_posed = [apply(ref_mats[i], ref_clouds[i])
                     for i in range(len(ref_clouds))]

    rows = []
    for did_s, info in sorted(manifest.items(), key=lambda kv: int(kv[0])):
        did = int(did_s)
        pair_npz = args.pairs_data / "val" / f"{did:05d}.npz"
        base = np.load(pair_npz, allow_pickle=True)
        gt_pcs = base["part_pcs_gt"].astype(np.float64)

        if args.mode == "juglet":
            i, j = info["i"], info["j"]
            ref_pts = np.concatenate([ref_posed[i], ref_posed[j]], 0)
            is_mate = info["pair"] in true_mates
        else:
            ref_pts = np.concatenate([gt_pcs[0], gt_pcs[1]], 0)
            is_mate = derive_mate_label(gt_pcs[0], gt_pcs[1])

        errs = []
        for inf_dir in args.inference_dirs:
            sdir = inf_dir / str(did)
            if not sdir.exists():
                continue
            clouds, mats = load_pfpp_layout(sdir, pair_npz)
            pred_pts = np.concatenate(
                [apply(mats[k], clouds[k]) for k in range(len(clouds))], 0)
            errs.append(register_chamfer(pred_pts, ref_pts))
        if not errs:
            print(f"WARNING: no runs for pair {info['pair']}")
            continue
        rows.append({"pair": info["pair"], "true_mate": bool(is_mate),
                     "chamfer_med": float(median(errs)),
                     "chamfer_mean": float(mean(errs)),
                     "per_seed": [float(e) for e in errs]})

    mates = [r for r in rows if r["true_mate"]]
    nons = [r for r in rows if not r["true_mate"]]

    def agg(sel, key="chamfer_med"):
        vals = [r[key] for r in sel]
        return {"n": len(vals), "mean": float(mean(vals)) if vals else None,
                "median": float(median(vals)) if vals else None}

    m_agg, n_agg = agg(mates), agg(nons)
    sep = (n_agg["median"] / m_agg["median"]
           if m_agg["median"] and n_agg["median"] else None)
    result = {"mode": args.mode, "true_mates": m_agg, "non_mates": n_agg,
              "separation_nonmate_over_mate": sep,
              "gate_median_le_0.045": (m_agg["median"] is not None
                                       and m_agg["median"] <= 0.045),
              "gate_separation_ge_1.25": (sep is not None and sep >= 1.25),
              "pairs": sorted(rows, key=lambda r: r["chamfer_med"])}
    with open(args.out / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    md = [f"# T3 — PF++ pairwise oracle ({args.mode})\n",
          f"true mates n={m_agg['n']}: mean {m_agg['mean']:.4f}, "
          f"median {m_agg['median']:.4f}",
          f"non-mates n={n_agg['n']}: mean {n_agg['mean']:.4f}, "
          f"median {n_agg['median']:.4f}",
          f"separation (non/mate, median): "
          f"{sep:.2f}x" if sep else "separation: n/a",
          f"gates: median<=0.045 {'PASS' if result['gate_median_le_0.045'] else 'FAIL'}, "
          f"sep>=1.25x {'PASS' if result['gate_separation_ge_1.25'] else 'FAIL'}\n",
          "| pair | mate | chamfer med | mean |", "|---|---|---|---|"]
    for r in result["pairs"]:
        md.append(f"| {r['pair']} | {'YES' if r['true_mate'] else ''} | "
                  f"{r['chamfer_med']:.4f} | {r['chamfer_mean']:.4f} |")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:12]))
    print(f"\nwrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
