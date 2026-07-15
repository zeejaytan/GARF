#!/usr/bin/env python3
"""Resolution-independent fracture-edge sharpness: Juglet vs Fractura ceramics.

GARF matches pieces by their fracture surfaces. Fresh ceramic breaks have SHARP
crisp fracture edges (the wall->break transition is a hard crease) that GARF's
encoder keys on. Archaeological sherds (Juglet) have edges worn/abraded over
centuries -> rounded transitions -> the fracture signal is degraded.

We measure, per piece, a scale-normalized surface-relief statistic that is
independent of mesh resolution: sample points+normals, and for each point compute
normal variation among neighbours within a FIXED physical radius (fraction of the
piece's size). Sharp fracture creases -> high normal variation; worn/smooth ->
low. Report the fraction of 'sharp' surface and the 90th-pct relief per piece,
aggregated per dataset.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def iter_pieces_h5(group):
    if "pieces" in group:
        pg = group["pieces"]
        for k in sorted(pg.keys(), key=int):
            yield np.asarray(pg[k]["vertices"][:]), np.asarray(pg[k]["faces"][:])
    else:
        for k in sorted([k for k in group.keys() if k.isdigit()], key=int):
            yield np.asarray(group[k]["vertices"][:]), np.asarray(group[k]["faces"][:])


def piece_relief(verts, faces, radius_frac=0.03, n_pts=6000, sharp_thr=0.15, rng=None):
    """Scale-normalized surface relief (resolution-independent).

    Returns (sharp_frac, relief_p90). relief = 1 - mean cos(angle) of normals
    among neighbours within radius_frac*scale of each sampled point."""
    if len(faces) < 8 or len(verts) < 8:
        return None
    rng = rng or np.random.default_rng(0)
    m = trimesh.Trimesh(verts.astype(np.float64), faces.astype(np.int64), process=False)
    scale = float(max(m.extents))
    if scale <= 0:
        return None
    try:
        pts, fid = trimesh.sample.sample_surface(m, n_pts)
    except Exception:
        return None
    fn = m.face_normals[fid]
    r = radius_frac * scale
    tree = cKDTree(pts)
    nbrs = tree.query_ball_point(pts, r)
    relief = np.zeros(len(pts))
    for i, ne in enumerate(nbrs):
        if len(ne) < 3:
            relief[i] = 0.0
            continue
        cos = fn[ne] @ fn[i]
        relief[i] = 1.0 - float(np.clip(cos, -1, 1).mean())
    return {
        "scale": scale,
        "sharp_frac": float(np.mean(relief > sharp_thr)),
        "relief_p90": float(np.percentile(relief, 90)),
        "relief_mean": float(np.mean(relief)),
    }


def collect(path: Path, split_key: str, max_samples: int, tag: str, rng):
    out = []
    with h5py.File(path, "r") as h:
        samples = [s.decode() if isinstance(s, bytes) else s for s in h["data_split"][split_key]["val"][:]]
        seen = set()
        for s in samples:
            if s in seen:
                continue
            seen.add(s)
            if len(seen) > max_samples:
                break
            for v, f in iter_pieces_h5(h[s]):
                r = piece_relief(v, f, rng=rng)
                if r:
                    r["tag"] = tag
                    r["sample"] = s
                    out.append(r)
    return out


def summarize(tag, rows):
    if not rows:
        return f"{tag:18s} (no pieces)"
    sf = np.array([r["sharp_frac"] for r in rows])
    p90 = np.array([r["relief_p90"] for r in rows])
    rm = np.array([r["relief_mean"] for r in rows])
    return (f"{tag:18s} n={len(rows):>3}  sharp_frac={sf.mean():.3f}±{sf.std():.3f}  "
            f"relief_p90={p90.mean():.3f}  relief_mean={rm.mean():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--radius-frac", type=float, default=0.03)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    datasets = [
        ("Juglet", ROOT / "input/juglet_deploy.hdf5", "juglet_deploy"),
        ("Fractura-ceramics", ROOT / "input/Fractura/fractura_real.hdf5", "ceramics"),
        ("Fractura-egg", ROOT / "input/Fractura/fractura_real.hdf5", "egg"),
        ("BB-everyday", Path("/data/gpfs/projects/punim2657/TORA/dataset/breaking_bad_vol.hdf5"), "everyday"),
    ]
    all_rows = {}
    for tag, path, key in datasets:
        if not path.exists():
            print(f"  skip {tag}: {path} missing")
            continue
        rows = collect(path, key, args.max_samples, tag, rng)
        all_rows[tag] = rows
        print(summarize(tag, rows))

    # per-sample breakdown for Juglet (only 1 object) + ceramics objects
    print("\n=== per-piece sharp_frac (Juglet) ===")
    for r in all_rows.get("Juglet", []):
        print(f"  {r['sample']:24s} sharp_frac={r['sharp_frac']:.3f} relief_p90={r['relief_p90']:.3f} scale={r['scale']:.3f}")


if __name__ == "__main__":
    main()
