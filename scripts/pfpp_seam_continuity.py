#!/usr/bin/env python3
"""T1c — seam continuity of the PF++ juglet layout.

For every coarse-touching pair in the PF++ layout, compare wall-thickness and
local-curvature statistics of the two seam bands (the point sets facing each
other across the join). If PF++ genuinely places sherds by macro geometry, the
two sides of a seam should agree better than under identity-permuted layouts
(the F1-vs-F2 discriminator at the seam level; also directly probes the
wall-profile channel proposed for Exp 16).

Estimates on the 1000-pt clouds:
  thickness at p — distance to the nearest same-part point q whose offset is
    near-parallel to the local (unoriented) normal at p, |cos| > 0.8, with
    ||q-p|| > eps (skips same-surface neighbours); capped at 0.3 part diag.
  curvature at p — smallest-eigenvalue fraction of the k=16 neighbourhood PCA.

Statistic per seam: |t_i - t_j| / mean(t) and |c_i - c_j| / mean(c) over band
means. Null: the same statistic over the touching pairs of N identity-permuted
layouts. One-sided Mann-Whitney U (true < null), gate p < 0.05.

Usage:
  python scripts/pfpp_seam_continuity.py \
      --pfpp-dir <inference>/juglet_deploy/0 \
      --pc-npz <pc_data>/juglet_deploy/val/00000.npz \
      --out logs/diagnostics/pfpp_t1c_<ts> [--n-null 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from pfpp_layout_probes import (  # noqa: E402
    apply, load_pfpp_layout, part_diag, permuted_layout)


def local_normals_curvature(pc, k=16):
    tree = cKDTree(pc)
    _, idx = tree.query(pc, k=k)
    normals = np.zeros_like(pc)
    curv = np.zeros(len(pc))
    for i in range(len(pc)):
        nb = pc[idx[i]] - pc[idx[i]].mean(0)
        cov = nb.T @ nb
        w, v = np.linalg.eigh(cov)
        normals[i] = v[:, 0]
        curv[i] = w[0] / (w.sum() + 1e-12)
    return normals, curv


def thickness(pc, normals, eps, cap):
    tree = cKDTree(pc)
    th = np.full(len(pc), np.nan)
    idx = tree.query_ball_point(pc, cap)
    for i in range(len(pc)):
        best = np.inf
        for j in idx[i]:
            d = pc[j] - pc[i]
            r = np.linalg.norm(d)
            if r < eps:
                continue
            if abs(np.dot(d / r, normals[i])) > 0.8 and r < best:
                best = r
        if np.isfinite(best):
            th[i] = best
    return th


def seam_stats(posed, diag, thick, curv):
    """Touching pairs + band mismatch stats for a posed layout."""
    trees = [cKDTree(p) for p in posed]
    tau = 0.03 * diag
    rows = []
    for i, j in combinations(range(len(posed)), 2):
        dij = trees[j].query(posed[i])[0]
        dji = trees[i].query(posed[j])[0]
        bi = dij < tau
        bj = dji < tau
        if bi.mean() < 0.02 or bj.mean() < 0.02:
            continue
        ti = np.nanmean(thick[i][bi])
        tj = np.nanmean(thick[j][bj])
        ci = np.nanmean(curv[i][bi])
        cj = np.nanmean(curv[j][bj])
        if not (np.isfinite(ti) and np.isfinite(tj)):
            continue
        rows.append({
            "pair": f"p{i+1:02d}{j+1:02d}",
            "band_i": float(bi.mean()), "band_j": float(bj.mean()),
            "thick_mismatch": float(abs(ti - tj) / (0.5 * (ti + tj) + 1e-12)),
            "curv_mismatch": float(abs(ci - cj) / (0.5 * (ci + cj) + 1e-12)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pfpp-dir", type=Path, required=True)
    ap.add_argument("--pc-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scan_clouds, mats = load_pfpp_layout(args.pfpp_dir, args.pc_npz)
    P = len(scan_clouds)

    # per-part local properties (pose-invariant: computed in scan frame)
    thick, curv = [], []
    for pc in scan_clouds:
        eps = 0.02 * part_diag(pc)
        cap = 0.30 * part_diag(pc)
        n, c = local_normals_curvature(pc)
        thick.append(thickness(pc, n, eps, cap))
        curv.append(c)

    posed = [apply(mats[i], scan_clouds[i]) for i in range(P)]
    allp = np.concatenate(posed, 0)
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    true_rows = seam_stats(posed, diag, thick, curv)

    null_thick, null_curv = [], []
    for _ in range(args.n_null):
        clouds_n, _, perm = permuted_layout(scan_clouds, mats, rng)
        thick_n = [thick[i] for i in range(P)]  # properties travel with the part
        curv_n = [curv[i] for i in range(P)]
        rows_n = seam_stats(clouds_n, diag, thick_n, curv_n)
        null_thick += [r["thick_mismatch"] for r in rows_n]
        null_curv += [r["curv_mismatch"] for r in rows_n]

    t_true = [r["thick_mismatch"] for r in true_rows]
    c_true = [r["curv_mismatch"] for r in true_rows]
    res = {"n_true_seams": len(t_true), "n_null_seams": len(null_thick)}
    for name, a, b in (("thickness", t_true, null_thick),
                       ("curvature", c_true, null_curv)):
        if a and b:
            u = mannwhitneyu(a, b, alternative="less")
            res[name] = {"true_median": float(np.median(a)),
                         "null_median": float(np.median(b)),
                         "p_less": float(u.pvalue),
                         "gate_p_lt_0.05": bool(u.pvalue < 0.05)}
    res["true_seams"] = true_rows

    with open(args.out / "seam_continuity.json", "w") as f:
        json.dump(res, f, indent=2)

    md = ["# T1c — seam continuity (PF++ layout vs permuted null)\n",
          f"true seams: {res['n_true_seams']}, null seams: {res['n_null_seams']}\n"]
    for name in ("thickness", "curvature"):
        if name in res:
            r = res[name]
            md.append(f"- {name} mismatch: true median {r['true_median']:.3f} "
                      f"vs null {r['null_median']:.3f}, one-sided p={r['p_less']:.4f} "
                      f"({'PASS' if r['gate_p_lt_0.05'] else 'FAIL'})")
    md += ["\n| seam | band_i | band_j | thick mism | curv mism |", "|---|---|---|---|---|"]
    for r in sorted(true_rows, key=lambda x: x["thick_mismatch"]):
        md.append(f"| {r['pair']} | {r['band_i']:.2f} | {r['band_j']:.2f} | "
                  f"{r['thick_mismatch']:.3f} | {r['curv_mismatch']:.3f} |")
    (args.out / "seam_continuity.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:10]))
    print(f"\nwrote {args.out}/seam_continuity.md")


if __name__ == "__main__":
    main()
