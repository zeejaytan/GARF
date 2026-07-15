#!/usr/bin/env python3
"""Compare per-fragment geometric statistics across datasets.

Goal: explain WHY GARF tolerates BB/Fractura but fails on Juglet, focusing on
the "surface noise" axis that the domain bridge (Exp 4a) found to be the
collapse trigger. The bridge added noise as displacement along vertex normals
~ N(0, strength * frag_scale); strength=0.01 (1% of max extent) already
collapsed accuracy. So we estimate each REAL mesh's intrinsic high-frequency
normal roughness normalized by frag_scale, directly comparable to `strength`.

Metrics per fragment (frag_scale = max bbox extent):
  n_faces, n_verts                      mesh resolution (count)
  edge_med/scale                        relative edge length (resolution)
  dihedral_mean_deg, dihedral_p90_deg   faceting / roughness (adjacent-face angle)
  hf_noise/scale                        high-freq normal residual / scale  <-- bridge analog
  hf_noise2/scale                       Taubin (curvature-removed) residual / scale

Aggregated as mean across sampled fragments per dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import trimesh


def iter_pieces(group):
    if "pieces" in group:
        pg = group["pieces"]
        for k in sorted(pg.keys(), key=lambda x: int(x)):
            yield np.asarray(pg[k]["vertices"][:]), np.asarray(pg[k]["faces"][:])
    else:
        for k in sorted([k for k in group.keys() if k.isdigit()], key=int):
            yield np.asarray(group[k]["vertices"][:]), np.asarray(group[k]["faces"][:])


def laplacian_residual_normal(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex |n . (v - mean(neighbors))| (one umbrella step)."""
    nbrs = mesh.vertex_neighbors
    v = mesh.vertices
    n = mesh.vertex_normals
    res = np.zeros(len(v))
    for i, ne in enumerate(nbrs):
        if len(ne) == 0:
            continue
        d = v[i] - v[ne].mean(0)
        res[i] = abs(float(np.dot(d, n[i])))
    return res


def taubin_residual_normal(mesh: trimesh.Trimesh) -> np.ndarray:
    """Residual after low-pass (Taubin lambda/mu) smoothing, projected on normal.
    Smoothing removes low-freq curvature, so residual isolates high-freq noise."""
    sm = mesh.copy()
    try:
        trimesh.smoothing.filter_taubin(sm, lamb=0.5, nu=0.53, iterations=10)
    except Exception:
        return np.full(len(mesh.vertices), np.nan)
    d = mesh.vertices - sm.vertices
    n = mesh.vertex_normals
    return np.abs(np.einsum("ij,ij->i", d, n))


def frag_stats(verts, faces) -> dict | None:
    if len(faces) < 4 or len(verts) < 4:
        return None
    m = trimesh.Trimesh(vertices=verts.astype(np.float64), faces=faces.astype(np.int64), process=False)
    ext = m.extents
    scale = float(max(ext)) if ext is not None and len(ext) else 1.0
    if scale <= 0:
        return None
    edges = m.edges_unique_length
    dih = np.degrees(np.abs(m.face_adjacency_angles)) if len(m.face_adjacency) else np.array([0.0])
    hf = laplacian_residual_normal(m)
    hf2 = taubin_residual_normal(m)
    # solidity / shell metrics (need watertight for a meaningful signed volume)
    wt = bool(m.is_watertight)
    bbox_vol = float(np.prod(ext)) if np.all(ext > 0) else np.nan
    area = float(m.area)
    if wt:
        vol = abs(float(m.volume))
        fill_ratio = vol / bbox_vol if bbox_vol > 0 else np.nan
        # slab thickness proxy: shell of thickness t -> vol ~ (area/2)*t
        thickness_rel = (2.0 * vol / area) / scale if area > 0 else np.nan
    else:
        fill_ratio = np.nan
        thickness_rel = np.nan
    return {
        "scale": scale,
        "n_faces": len(faces),
        "n_verts": len(verts),
        "watertight": 1.0 if wt else 0.0,
        "edge_med_rel": float(np.median(edges)) / scale,
        "dihedral_mean_deg": float(np.mean(dih)),
        "dihedral_p90_deg": float(np.percentile(dih, 90)),
        "hf_rel": float(np.sqrt(np.mean(hf ** 2))) / scale,
        "hf2_rel": float(np.sqrt(np.nanmean(hf2 ** 2))) / scale,
        "fill_ratio": fill_ratio,
        "thickness_rel": thickness_rel,
    }


def collect(path: Path, split_key: str, max_samples: int) -> list[dict]:
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
            for verts, faces in iter_pieces(h[s]):
                st = frag_stats(verts, faces)
                if st:
                    out.append(st)
    return out


def summarize(name: str, rows: list[dict]) -> dict:
    keys = ["scale", "n_faces", "n_verts", "watertight", "edge_med_rel", "dihedral_mean_deg",
            "dihedral_p90_deg", "hf_rel", "hf2_rel", "fill_ratio", "thickness_rel"]
    agg = {"dataset": name, "n_frags": len(rows)}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k] = float(np.nanmean(vals))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-samples", type=int, default=8)
    args = ap.parse_args()

    datasets = [
        ("BB(everyday)", Path("/data/gpfs/projects/punim2657/TORA/dataset/breaking_bad_vol.hdf5"), "everyday"),
        ("Fractura(pig)", Path("input/Fractura/bone_synthetic.hdf5"), "pig"),
        ("Juglet", Path("input/juglet_deploy.hdf5"), "juglet_deploy"),
    ]
    aggs = []
    for name, path, key in datasets:
        rows = collect(path, key, args.max_samples)
        aggs.append(summarize(name, rows))

    cols = [
        ("n_frags", "frags", "{:.0f}"),
        ("n_faces", "faces", "{:.0f}"),
        ("watertight", "wt_frac", "{:.2f}"),
        ("edge_med_rel", "edge/scale", "{:.4f}"),
        ("dihedral_mean_deg", "dih_mean°", "{:.2f}"),
        ("dihedral_p90_deg", "dih_p90°", "{:.2f}"),
        ("hf_rel", "hf/scale", "{:.5f}"),
        ("hf2_rel", "taubin/scale", "{:.5f}"),
        ("fill_ratio", "fill_ratio", "{:.3f}"),
        ("thickness_rel", "thick/scale", "{:.4f}"),
    ]
    header = "dataset".ljust(15) + "".join(h.rjust(16) for _, h, _ in cols)
    print(header)
    print("-" * len(header))
    for a in aggs:
        line = a["dataset"].ljust(15) + "".join(fmt.format(a[k]).rjust(16) for k, _, fmt in cols)
        print(line)
    print()
    print("Bridge reference: noise strength=0.01 (= hf_noise/scale ~0.01) collapsed GARF;")
    print("strength=0.005 was the first sign of degradation.")


if __name__ == "__main__":
    main()
