#!/usr/bin/env python3
"""Distinctiveness of the FRACTURE/CONTACT band — the signal GARF matches on.

For each Exp4c object (GT-assembled pieces in view_gt.glb), find, for every
touching pair, the contact band (points on piece i within tau of piece j) and
measure its surface relief (scale-normalized normal variation within a fixed
radius). A flat/smoothly-curved seam (low relief) is ambiguous -> pieces can
slide -> hard. A rough interlocking seam (high relief) is uniquely matchable.

Per-object score = contact-area-weighted mean band relief. Correlate with
part_acc. This localizes 'sharpness' to the actual mating surfaces (unlike the
whole-piece relief probe) and 'distinctiveness' to contact (unlike raw area).
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def part_samples(glb_path: Path, n_per: int = 4000):
    scene = trimesh.load(glb_path, force="scene")
    meshes = [scene] if isinstance(scene, trimesh.Trimesh) else scene.dump(concatenate=False)
    parts = []
    for m in meshes:
        if len(getattr(m, "faces", [])) == 0:
            continue
        try:
            p, fid = trimesh.sample.sample_surface(m, n_per)
        except Exception:
            continue
        parts.append((np.asarray(p), np.asarray(m.face_normals[fid])))
    return parts


def band_relief(pts, nrm, mask, scale, radius_frac=0.04):
    """Mean normal-variation among contact-band points within radius."""
    idx = np.where(mask)[0]
    if len(idx) < 5:
        return None
    cp = pts[idx]
    cn = nrm[idx]
    r = radius_frac * scale
    tree = cKDTree(cp)
    nbrs = tree.query_ball_point(cp, r)
    rel = []
    for a, ne in enumerate(nbrs):
        if len(ne) < 3:
            continue
        cos = cn[ne] @ cn[a]
        rel.append(1.0 - float(np.clip(cos, -1, 1).mean()))
    return float(np.mean(rel)) if rel else None


def object_band_relief(parts, tau_frac=0.02, radius_frac=0.04):
    P = len(parts)
    allp = np.vstack([p for p, _ in parts])
    scale = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    tau = tau_frac * scale
    trees = [cKDTree(p) for p, _ in parts]
    weighted, wsum = 0.0, 0.0
    bands = []
    for i, j in combinations(range(P), 2):
        pi, ni = parts[i]
        pj, nj = parts[j]
        dij, _ = trees[j].query(pi)
        mi = dij < tau
        dji, _ = trees[i].query(pj)
        mj = dji < tau
        for (pp, nn, mm) in [(pi, ni, mi), (pj, nj, mj)]:
            if mm.sum() < 5:
                continue
            br = band_relief(pp, nn, mm, scale, radius_frac)
            if br is None:
                continue
            w = float(mm.sum())
            weighted += br * w
            wsum += w
            bands.append(br)
    if wsum == 0:
        return None
    return {"band_relief_w": weighted / wsum, "band_relief_mean": float(np.mean(bands)), "n_bands": len(bands)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinviz-stamp", default="20260605_163400")
    ap.add_argument("--thinwall-stamp", default="20260605_162150")
    args = ap.parse_args()

    metrics = {}
    for d in glob.glob(f"logs/diagnostics/thinwall_{args.thinwall_stamp}_*_init1_s41/version_0/json_results/*.json"):
        j = json.load(open(d)); metrics[j["name"]] = j

    rows = []
    for cat in ["ceramics", "egg", "bones"]:
        base = ROOT / f"logs/deploy/thinviz_{args.thinviz_stamp}_{cat}/version_0/assembly_results/{cat}"
        if not base.is_dir():
            continue
        for sd in sorted(base.iterdir()):
            gt = sd / "view_gt.glb"
            if not gt.exists():
                continue
            key = f"{cat}/{sd.name}"; m = metrics.get(key, {})
            parts = part_samples(gt)
            if len(parts) < 2:
                continue
            s = object_band_relief(parts)
            if s is None:
                continue
            rows.append({"key": key, "cat": cat, "parts": m.get("num_parts"),
                         "part_acc": m.get("part_acc", float("nan")), **s})
            print(f"  {key:28s} P={m.get('num_parts'):>2} acc={m.get('part_acc',0):.2f} bandRelief={s['band_relief_w']:.3f}")

    print("\n=== ceramics+egg sorted by contact-band relief (low = smooth/ambiguous seam) ===")
    sub = [r for r in rows if r["cat"] in ("ceramics", "egg")]
    sub.sort(key=lambda r: r["band_relief_w"])
    print(f"{'object':26s} {'P':>2} {'acc':>5} {'bandRelief':>10}")
    for r in sub:
        flag = " <- FAILS" if r["part_acc"] < 0.6 else ""
        print(f"{r['key']:26s} {r['parts']:>2} {r['part_acc']:>5.2f} {r['band_relief_w']:>10.3f}{flag}")

    for grp, sel in [("ceramics+egg", ("ceramics", "egg")), ("all", ("ceramics", "egg", "bones"))]:
        g = [r for r in rows if r["cat"] in sel]
        a = np.array([r["part_acc"] for r in g]); b = np.array([r["band_relief_w"] for r in g])
        if len(g) > 2:
            print(f"\nr(bandRelief, part_acc) [{grp}] = {np.corrcoef(b, a)[0,1]:.3f}")
    (ROOT / "logs/diagnostics/contact_band_relief.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
