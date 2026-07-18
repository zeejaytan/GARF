#!/usr/bin/env python3
"""Exp 15 — build the self-supervised Juglet-adaptation training HDF5.

Exp 14 showed geometric de-weathering re-triggers the frozen encoder's fracture
head (fired% 0.50->1.88) but does NOT restore pairwise mating: amplifying the
surviving worn texture cannot re-synthesize the fresh-break micro-structure the
pretrained representation keys on. The surviving hypothesis (a) is
representation mismatch: the worn geometry still carries a mating signal, but
only an encoder *adapted to real worn ceramic texture* can extract it.

This builds the training set for that adaptation: a copy of the labeled
synthetic replay stream (bone_synthetic) plus K replicas of the Juglet's own
nine sherds with PSEUDO fracture labels from the validated relief-band
detector (the Exp 8/10 detector — geometric, independent of the encoder, so
no circularity). FracSeg trains with a pure dice loss on
``shared_faces != -1`` (assembly/data/breaking_bad/weighted.py), so writing
the pseudo-label into ``shared_faces`` makes the existing pipeline consume it
verbatim. Per-epoch variation comes from the dataloader's random rotation +
fresh surface resampling of each replica.

Two Juglet-specific safeguards:
  - pieces are TRANSLATED APART before writing (the local02 frames overlap
    near the origin): the physically-gated worn-break erosion augmentation
    (`fracture_erosion.erode_contact_bands`) then finds no contact band on
    Juglet samples and no-ops, instead of mollifying spurious "contacts".
    FracSeg recenters each part independently, so training is unaffected.
  - band faces get the index of a PF++-adjacent piece (adjacency.json) as
    their ``shared_faces`` value, giving a sane part-graph; -1 elsewhere.

NOTE: the output is for the FracSeg (encoder) stage ONLY. Denoiser
co-adaptation needs real GT poses, which Juglet lacks — co-adapt on the
original bone_synthetic.hdf5 (Exp 13 recipe).

Usage
-----
  python scripts/build_juglet_selfsup_hdf5.py \
      --juglet-source input/juglet_deploy_local02.hdf5 \
      --sample artifact/Juglet-000 \
      --synth-source input/Fractura/bone_synthetic.hdf5 \
      --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
      --replicas 80 \
      --out input/juglet_selfsup_mix.hdf5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))


def face_band_label(verts: np.ndarray, faces: np.ndarray, *,
                    relief_pct: float = 95.0, band_frac: float = 0.015,
                    n_probe: int = 20000, seed: int = 0) -> np.ndarray:
    """Per-FACE fracture-band pseudo-label from the relief detector (same
    statistic as fracseg_introspection.relief_band_label, applied to face
    centroids). Defaults are TIGHTER than the Exp 10 scoring probe
    (pct 85 / 0.05): that setting marks 40-72% of a sherd as band — usable
    for scoring, useless as a training label. Calibrated 2026-07-19 on the
    real sherds: pct 95 / 0.015 gives mean 17% of faces (range 8-31%),
    matching the expected fracture-edge ribbon of a thin-walled vessel."""
    m = trimesh.Trimesh(vertices=np.asarray(verts, np.float64),
                        faces=np.asarray(faces, np.int64), process=False)
    scale = float(max(m.extents))
    pts, fid = trimesh.sample.sample_surface(m, n_probe, seed=seed)
    pts = np.asarray(pts, np.float64)
    fn = m.face_normals[fid]
    tree = cKDTree(pts)
    relief = np.zeros(len(pts))
    for i, nb in enumerate(tree.query_ball_point(pts, 0.03 * scale)):
        if len(nb) < 3:
            continue
        cos = fn[nb] @ fn[i]
        relief[i] = 1.0 - float(np.clip(cos, -1.0, 1.0).mean())
    if not np.any(relief > 0):
        return np.zeros(len(faces), dtype=bool)
    anchors = pts[relief >= np.percentile(relief, relief_pct)]
    if len(anchors) == 0:
        return np.zeros(len(faces), dtype=bool)
    centroids = m.triangles_center
    d, _ = cKDTree(anchors).query(centroids)
    return d <= band_frac * scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--juglet-source", type=Path,
                    default=Path("input/juglet_deploy_local02.hdf5"))
    ap.add_argument("--sample", default="artifact/Juglet-000")
    ap.add_argument("--synth-source", type=Path,
                    default=Path("input/Fractura/bone_synthetic.hdf5"))
    ap.add_argument("--adjacency", type=Path,
                    default=Path("logs/diagnostics/juglet_adjacency/adjacency.json"))
    ap.add_argument("--replicas", type=int, default=80,
                    help="Juglet copies in the train split (~1:4 vs the 332 "
                         "synthetic samples)")
    ap.add_argument("--relief-pct", type=float, default=95.0)
    ap.add_argument("--band-frac", type=float, default=0.015)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    adj = np.asarray(json.load(open(args.adjacency))["adjacency_matrix"])

    print(f"copying replay stream {args.synth_source} -> {args.out} ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.synth_source, args.out)

    with h5py.File(args.juglet_source, "r") as src:
        g = src[args.sample]["pieces"]
        keys = sorted(g.keys(), key=int)
        pieces = [(np.asarray(g[k]["vertices"][:], np.float64),
                   np.asarray(g[k]["faces"][:], np.int64)) for k in keys]
        if "pieces_names" in src[args.sample]:
            names = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                     for x in src[args.sample]["pieces_names"][:]]
        else:
            names = [f"Piece{i + 1:02d}" for i in range(len(pieces))]
    n = len(pieces)
    assert adj.shape == (n, n), f"adjacency {adj.shape} != {n} pieces"

    # Pseudo-labels (per piece, computed once — replicas share them) + spread.
    labeled = []
    spread = 2.5 * max(float(max(np.ptp(v, axis=0))) for v, _ in pieces)
    for i, (v, f) in enumerate(pieces):
        band = face_band_label(v, f, relief_pct=args.relief_pct,
                               band_frac=args.band_frac, seed=args.seed + i)
        mates = np.flatnonzero(adj[i])
        neighbor = int(mates[0]) if len(mates) else (i + 1) % n
        shared = np.where(band, neighbor, -1).astype(np.int64)
        v_spread = v + np.array([i * spread, 0.0, 0.0])
        labeled.append((v_spread, f, shared))
        print(f"  {names[i]}: band faces {band.mean()*100:.1f}% "
              f"(neighbor piece {neighbor}, mates {list(mates)})")
    band_fracs = [float((s != -1).mean()) for _, _, s in labeled]
    assert 0.02 < np.mean(band_fracs) < 0.6, (
        f"pseudo-label band fraction {np.mean(band_fracs):.3f} implausible — "
        "check relief detector params")

    pname_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(args.out, "r+") as dst:
        sample_keys = []
        for k in range(args.replicas):
            sname = f"artifact/Juglet-ss{k:03d}"
            grp = dst.create_group(sname)
            pg = grp.create_group("pieces")
            for i, (v, f, s) in enumerate(labeled):
                pc = pg.create_group(str(i))
                pc.create_dataset("vertices", data=v)
                pc.create_dataset("faces", data=f)
                pc.create_dataset("shared_faces", data=s)
            grp.create_dataset("pieces_names",
                               data=[nm.encode("utf-8") for nm in names],
                               dtype=pname_dtype)
            grp.create_dataset("removal_masks", data=np.ones((1, n), dtype=bool))
            grp.create_dataset("removal_order", data=np.arange(n, dtype=np.int64))
            sample_keys.append(sname.encode("utf-8"))

        split = dst["data_split"].create_group("artifact")
        split.create_dataset("train", data=np.array(sample_keys, dtype=object))
        for s in ("val", "test"):
            split.create_dataset(s, data=np.array(sample_keys[:1], dtype=object))

    print(f"wrote {args.out}: +{args.replicas} Juglet replicas "
          f"(mean pseudo-band {np.mean(band_fracs)*100:.1f}% of faces), "
          f"pieces spread {spread:.3f} apart (erosion aug no-ops on them)")


if __name__ == "__main__":
    main()
