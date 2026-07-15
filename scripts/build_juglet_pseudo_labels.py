#!/usr/bin/env python3
"""Exp 14 — build a pseudo-fracture-labeled Juglet training set from REAL worn geometry.

Exp 11-13 showed adapting on synthetically-eroded *bone* breaks helps only weakly:
synthetic wear is an imperfect proxy for real archaeological *ceramic* wear. The one
real weathered object we have is the Juglet itself. It has no fracture labels
(stored shared_faces are degenerate: all faces flagged), so we derive PSEUDO
fracture labels geometrically with the validated relief-band detector (Exp 8, base.
rim_face_weights: precision 0.30-0.99 / recall 0.63-0.97 on blue_pot): a face is a
fracture face if its centroid lies within band_frac*scale of a top-relief anchor.

The output HDF5 lets the FracSeg fine-tuner teach the encoder that Juglet's REAL
worn rims are fracture surfaces — injecting the real worn-texture signal the
synthetic proxy lacks. Juglet-000 is duplicated `--repeat` times in the train split
so it carries weight against the (optional) synthetic regulariser.

Usage
-----
  python scripts/build_juglet_pseudo_labels.py \
      --mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
      --out input/juglet_pseudo_labeled.hdf5 --repeat 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

RELIEF_RADIUS_FRAC = 0.03
RELIEF_SAMPLES = 4000


def relief_band_faces(mesh: trimesh.Trimesh, relief_pct: float, band_frac: float,
                      seed: int = 0) -> np.ndarray:
    """Per-face bool: True where the face is in the geometric fracture-rim band.
    Mirrors assembly/data/breaking_bad/base.rim_face_weights."""
    scale = float(max(mesh.extents))
    if scale <= 0 or len(mesh.faces) < 8:
        return np.zeros(len(mesh.faces), dtype=bool)
    pts, fid = trimesh.sample.sample_surface(mesh, RELIEF_SAMPLES, seed=seed)
    pts = np.asarray(pts)
    normals = mesh.face_normals[fid]
    r = RELIEF_RADIUS_FRAC * scale
    tree = cKDTree(pts)
    relief = np.zeros(len(pts))
    for i, nb in enumerate(tree.query_ball_point(pts, r)):
        if len(nb) < 3:
            continue
        cos = normals[nb] @ normals[i]
        relief[i] = 1.0 - float(np.clip(cos, -1.0, 1.0).mean())
    if not np.any(relief > 0):
        return np.zeros(len(mesh.faces), dtype=bool)
    anchors = pts[relief >= np.percentile(relief, relief_pct)]
    if len(anchors) == 0:
        return np.zeros(len(mesh.faces), dtype=bool)
    d, _ = cKDTree(anchors).query(mesh.triangles_center)
    return d <= band_frac * scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--relief-pct", type=float, default=80.0)
    ap.add_argument("--band-frac", type=float, default=0.05)
    ap.add_argument("--sample-name", default="artifact/Juglet-000")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    objs = sorted(p for p in args.mesh_dir.iterdir() if p.suffix == ".obj")
    print(f"{len(objs)} Juglet pieces from {args.mesh_dir}")
    pieces = []
    band_fracs = []
    for i, obj in enumerate(objs):
        m = trimesh.load(str(obj), process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        band = relief_band_faces(m, args.relief_pct, args.band_frac, seed=i)
        band_fracs.append(float(band.mean()))
        # shared_faces convention: fracture faces = 0 (a valid neighbour id),
        # non-fracture = -1 (matches (shared_faces != -1) fracture label).
        sf = np.where(band, 0, -1).astype(np.int64)
        pieces.append((np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), sf))
    print("per-piece band fraction:", [round(b, 3) for b in band_fracs])

    pname_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(args.out, "w") as f:
        grp = f.create_group(args.sample_name)
        pg = grp.create_group("pieces")
        for k, (v, fc, sf) in enumerate(pieces):
            g = pg.create_group(str(k))
            g.create_dataset("vertices", data=v)
            g.create_dataset("faces", data=fc)
            g.create_dataset("shared_faces", data=sf)
        grp.create_dataset("pieces_names",
                           data=[f"Piece{k+1:02d}".encode() for k in range(len(pieces))],
                           dtype=pname_dtype)
        grp.create_dataset("removal_masks", data=np.ones((1, len(pieces)), dtype=bool))
        grp.create_dataset("removal_order", data=np.arange(len(pieces), dtype=np.int64))

        cat = args.sample_name.split("/")[0]
        ds = f.create_group("data_split").create_group(cat)
        ref = np.array([args.sample_name.encode()] * args.repeat, dtype=object)
        for sname in ("train", "val", "test"):
            ds.create_dataset(sname, data=ref)
    mean_band = float(np.mean(band_fracs))
    print(f"wrote {args.out}: {len(pieces)} pieces, mean band frac {mean_band:.3f}, "
          f"train repeat {args.repeat}")


if __name__ == "__main__":
    main()
