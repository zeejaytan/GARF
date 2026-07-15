#!/usr/bin/env python3
"""Test the symmetry-ambiguity hypothesis for GARF assembly failure.

For each Exp4c object (assembled GT from view_gt.glb), estimate how close it is
to a surface of revolution: rotate the assembled point cloud about its best axis
by many angles and measure normalized Chamfer self-distance. A body of revolution
maps onto itself at every angle (low avg CD) -> shard arrangement is geometrically
ambiguous (slide/rotate around the axis) -> hard for GARF. Featured vessels
(handles/spouts/rims) have high CD -> unambiguous.

Correlate axisymmetry score with part_acc to test whether symmetry, not part
count or thinness, predicts failure.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]


def assembled_points(glb_path: Path, n: int = 4000) -> np.ndarray:
    scene = trimesh.load(glb_path, force="scene")
    meshes = [scene] if isinstance(scene, trimesh.Trimesh) else scene.dump(concatenate=False)
    pts = []
    for m in meshes:
        if len(getattr(m, "faces", [])) == 0:
            continue
        try:
            p, _ = trimesh.sample.sample_surface_even(m, max(n // max(len(meshes), 1) * 2, 200))
        except Exception:
            p, _ = trimesh.sample.sample_surface(m, max(n // max(len(meshes), 1) * 2, 200))
        pts.append(np.asarray(p))
    allp = np.vstack(pts)
    if len(allp) > n:
        idx = np.random.default_rng(0).choice(len(allp), n, replace=False)
        allp = allp[idx]
    return allp


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = tb.query(a)
    db, _ = ta.query(b)
    return float(0.5 * (da.mean() + db.mean()))


def axisymmetry_score(pts: np.ndarray, n_angles: int = 12) -> dict:
    """Lower score = closer to a surface of revolution (more ambiguous)."""
    c = pts.mean(0)
    p = pts - c
    scale = float(np.linalg.norm(p, axis=1).max())  # bounding radius
    # candidate axes = principal axes
    cov = np.cov(p.T)
    _, vecs = np.linalg.eigh(cov)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)[1:]  # skip 0
    best_axis_score = np.inf
    per_axis = []
    for k in range(3):
        axis = vecs[:, k]
        cds = []
        for ang in angles:
            rot = R.from_rotvec(axis * ang).as_matrix()
            pr = p @ rot.T
            cds.append(chamfer(p, pr) / scale)
        avg = float(np.mean(cds))
        per_axis.append(avg)
        best_axis_score = min(best_axis_score, avg)
    return {"axisym_score": best_axis_score, "per_axis": per_axis}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinviz-stamp", default="20260605_163400")
    ap.add_argument("--thinwall-stamp", default="20260605_162150")
    ap.add_argument("--n-points", type=int, default=4000)
    args = ap.parse_args()

    # metrics map
    metrics = {}
    for d in glob.glob(f"logs/diagnostics/thinwall_{args.thinwall_stamp}_*_init1_s41/version_0/json_results/*.json"):
        j = json.load(open(d))
        metrics[j["name"]] = j

    rows = []
    deploy = ROOT / "logs" / "deploy"
    for cat_dir in sorted(deploy.glob(f"thinviz_{args.thinviz_stamp}_*")):
        cat = cat_dir.name.split(f"thinviz_{args.thinviz_stamp}_", 1)[-1]
        asm = cat_dir / "version_0" / "assembly_results" / cat
        if not asm.is_dir():
            continue
        for sd in sorted(asm.iterdir()):
            gt = sd / "view_gt.glb"
            if not gt.exists():
                continue
            key = f"{cat}/{sd.name}"
            m = metrics.get(key, {})
            pts = assembled_points(gt, args.n_points)
            s = axisymmetry_score(pts)
            rows.append({
                "key": key, "parts": m.get("num_parts"),
                "part_acc": m.get("part_acc", float("nan")),
                "axisym": s["axisym_score"],
            })
            print(f"  {key:30s} P={m.get('num_parts'):>2} acc={m.get('part_acc',0):.2f} axisym={s['axisym_score']:.4f}")

    print("\n=== sorted by axisymmetry (low = surface-of-revolution = ambiguous) ===")
    rows.sort(key=lambda r: r["axisym"])
    print(f"{'object':30s} {'P':>2} {'acc':>5} {'axisym':>8}")
    for r in rows:
        flag = " <- FAILS" if r["part_acc"] < 0.6 else ""
        print(f"{r['key']:30s} {r['parts']:>2} {r['part_acc']:>5.2f} {r['axisym']:>8.4f}{flag}")

    # correlation
    acc = np.array([r["part_acc"] for r in rows])
    sym = np.array([r["axisym"] for r in rows])
    if len(rows) > 2:
        corr = float(np.corrcoef(sym, acc)[0, 1])
        print(f"\nPearson r(axisym, part_acc) = {corr:.3f}  (positive => more symmetric -> lower acc)")
    out = ROOT / "logs" / "diagnostics" / "symmetry_analysis.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
