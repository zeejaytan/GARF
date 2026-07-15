#!/usr/bin/env python3
"""Positive control for Exp 6 — build 2-piece subproblems for KNOWN-GOOD objects.

Exp 6 found 0/36 Juglet pairs mate, and cross-tabbing against the PF++ adjacency
showed true mates are indistinguishable from non-mates (rot dispersion ~71 vs
67 deg). But that is only decisive if the 2-piece pairwise-oracle proxy can EVER
register a mate. This builds the control: the same pairwise decomposition for
Fractura ceramics objects GARF assembles well (part_acc >= 0.92 in Exp 4c /
Exp 5), so we can check whether their TRUE mates show low cross-seed dispersion /
contact — i.e. whether the proxy works at all.

Because Fractura provides GT assembly, pieces are stored in their true relative
(assembled) positions. We copy each pair's geometry verbatim (as
build_juglet_pairs_hdf5 does) AND label the pair a TRUE MATE by contact between
the two stored (assembled) meshes.

Output HDF5 layout mirrors the Juglet pairs file: every pair becomes
``<category>/<obj>__p<ij>`` and all are listed in
``data_split/<category>/{train,val,test}`` so one eval covers them.

Exp 7 (rim-erosion bridge): ``--erode-strength S`` applies archaeological-wear
mollification (fracture_mesh_ops.erode_fracture_band) to each object's TRUE
contact band before pairing. Geometry is edited in place (GT pose preserved),
mate labels are computed on the ORIGINAL meshes so labels stay fixed across
strengths, and per-piece relief_p90 before/after is recorded in the adjacency
JSON for calibration against Juglet's worn level (~0.171).

Usage
-----
  python scripts/build_control_pairs_hdf5.py \
      --source input/Fractura/fractura_real.hdf5 \
      --objects ceramics/pink_bowl ceramics/narrow_bottle2 \
                ceramics/blue_pot ceramics/narrow_bottle4 \
      --out input/control_ceramics_pairs.hdf5 \
      --category control \
      --adjacency-out logs/diagnostics/control_ceramics_adjacency.json \
      [--erode-strength 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fracture_mesh_ops import erode_fracture_band, piece_relief_stats


def piece_mesh(pg: h5py.Group) -> trimesh.Trimesh:
    v = np.asarray(pg["vertices"][:], dtype=np.float64)
    f = np.asarray(pg["faces"][:], dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def copy_piece(src: h5py.Group, dst: h5py.Group, verts: np.ndarray | None = None) -> None:
    """Copy a piece group; ``verts`` (if given) overrides the stored vertices
    (used to write eroded geometry while keeping faces/shared_faces intact)."""
    v = (np.asarray(verts, dtype=np.float64) if verts is not None
         else np.asarray(src["vertices"][:], dtype=np.float64))
    f = np.asarray(src["faces"][:], dtype=np.int64)
    dst.create_dataset("vertices", data=v)
    dst.create_dataset("faces", data=f)
    shared = (np.asarray(src["shared_faces"][:], dtype=np.int64)
              if "shared_faces" in src else np.zeros(len(f), dtype=np.int64))
    dst.create_dataset("shared_faces", data=shared)


def is_true_mate(ma: trimesh.Trimesh, mb: trimesh.Trimesh, scale: float,
                 gap_tau: float, contact_tau: float, n: int = 6000) -> tuple[bool, float, float]:
    # Fixed seeds: mate labels must be deterministic so borderline pairs do not
    # flip between builds (e.g. across Exp 7 erosion strengths).
    pa = np.asarray(trimesh.sample.sample_surface(ma, n, seed=101)[0])
    pb = np.asarray(trimesh.sample.sample_surface(mb, n, seed=102)[0])
    ta, tb = cKDTree(pa), cKDTree(pb)
    dab, _ = tb.query(pa)
    dba, _ = ta.query(pb)
    min_gap = float(min(dab.min(), dba.min()))
    tau = gap_tau * scale
    cfrac = float(max(np.mean(dab < tau), np.mean(dba < tau)))
    mate = (min_gap / scale) < gap_tau and cfrac > contact_tau
    return mate, min_gap / scale, cfrac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--objects", nargs="+", required=True,
                    help="e.g. ceramics/blue_pot ceramics/pink_bowl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--category", default="control")
    ap.add_argument("--adjacency-out", type=Path, required=True)
    ap.add_argument("--gap-tau", type=float, default=0.03)
    ap.add_argument("--contact-tau", type=float, default=0.03)
    ap.add_argument("--erode-strength", type=float, default=0.0,
                    help="Exp 7: archaeological-wear mollification strength in [0,1] "
                         "applied to each object's true fracture contact band "
                         "(0 = verbatim copy, identical to the original behaviour).")
    ap.add_argument("--erode-kernel-frac", type=float, default=0.05,
                    help="Exp 7b: mollification kernel radius as a fraction of piece "
                         "scale (erode_fracture_band kernel_frac_max). Exp 7 showed "
                         "relief_p90 plateaus at ~0.20 with the default 0.05; larger "
                         "radii carve deeper wear.")
    ap.add_argument("--erode-knn", type=int, default=48,
                    help="Exp 7b: max surface samples averaged per vertex inside the "
                         "kernel radius. MUST grow ~quadratically with "
                         "--erode-kernel-frac, otherwise the k nearest samples "
                         "silently shrink the effective kernel back to the default.")
    args = ap.parse_args()
    if not 0.0 <= args.erode_strength <= 1.0:
        ap.error("--erode-strength must be in [0, 1]")
    if not 0.0 < args.erode_kernel_frac <= 0.5:
        ap.error("--erode-kernel-frac must be in (0, 0.5]")
    if args.erode_knn < 8:
        ap.error("--erode-knn must be >= 8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.adjacency_out.parent.mkdir(parents=True, exist_ok=True)

    pname_dtype = h5py.special_dtype(vlen=str)
    sample_keys: list[bytes] = []
    adjacency = {}
    relief_report = {}

    with h5py.File(args.source, "r") as src, h5py.File(args.out, "w") as dst:
        for obj in args.objects:
            objname = obj.split("/")[-1]
            src_pieces = src[obj]["pieces"]
            keys = sorted(src_pieces.keys(), key=int)
            n = len(keys)
            meshes = [piece_mesh(src_pieces[k]) for k in keys]
            allv = np.concatenate([m.vertices for m in meshes], axis=0)
            scale = float(np.linalg.norm(allv.max(0) - allv.min(0)))
            if "pieces_names" in src[obj]:
                names = [x.decode() if isinstance(x, bytes) else str(x)
                         for x in src[obj]["pieces_names"][:]]
            else:
                names = [f"Piece{i+1:02d}" for i in range(n)]
            print(f"{obj}: {n} pieces, assembled scale {scale:.3f}")

            # Exp 7: erode the true fracture contact band (assembled pose known).
            # Mate labels below still come from the ORIGINAL `meshes`.
            if args.erode_strength > 0.0:
                eroded_verts = erode_fracture_band(
                    [(m.vertices, m.faces) for m in meshes], args.erode_strength,
                    kernel_frac_max=args.erode_kernel_frac,
                    knn=args.erode_knn,
                )
                relief_report[objname] = []
                for pi, (m, ev) in enumerate(zip(meshes, eroded_verts)):
                    r0 = piece_relief_stats(np.asarray(m.vertices), np.asarray(m.faces))
                    r1 = piece_relief_stats(ev, np.asarray(m.faces))
                    relief_report[objname].append(
                        {"piece": pi,
                         "relief_p90_orig": r0["relief_p90"],
                         "relief_p90_eroded": r1["relief_p90"]}
                    )
                    print(f"  piece {pi}: relief_p90 {r0['relief_p90']:.3f} -> {r1['relief_p90']:.3f}")
            else:
                eroded_verts = [None] * n

            for a, b in combinations(range(n), 2):
                sample_name = f"{args.category}/{objname}__p{a+1:02d}{b+1:02d}"
                grp = dst.create_group(sample_name)
                pieces = grp.create_group("pieces")
                copy_piece(src_pieces[keys[a]], pieces.create_group("0"), verts=eroded_verts[a])
                copy_piece(src_pieces[keys[b]], pieces.create_group("1"), verts=eroded_verts[b])
                grp.create_dataset("pieces_names",
                                   data=[names[a].encode(), names[b].encode()],
                                   dtype=pname_dtype)
                grp.create_dataset("removal_masks", data=np.ones((1, 2), dtype=bool))
                grp.create_dataset("removal_order", data=np.arange(2, dtype=np.int64))
                sample_keys.append(sample_name.encode())

                mate, gap, cfrac = is_true_mate(meshes[a], meshes[b], scale,
                                                args.gap_tau, args.contact_tau)
                adjacency[f"{objname}__p{a+1:02d}{b+1:02d}"] = {
                    "object": objname, "true_mate": bool(mate),
                    "min_gap_over_scale": gap, "contact_frac": cfrac,
                }

        ds_root = dst.create_group("data_split")
        split = ds_root.create_group(args.category)
        ref = np.array(sample_keys, dtype=object)
        for sname in ("train", "val", "test"):
            split.create_dataset(sname, data=ref)

    n_mate = sum(v["true_mate"] for v in adjacency.values())
    with open(args.adjacency_out, "w") as f:
        json.dump({"gap_tau": args.gap_tau, "contact_tau": args.contact_tau,
                   "erode_strength": args.erode_strength,
                   "erode_kernel_frac": args.erode_kernel_frac,
                   "erode_knn": args.erode_knn,
                   "relief": relief_report,
                   "n_pairs": len(adjacency), "n_true_mates": n_mate,
                   "pairs": adjacency}, f, indent=2)
    print(f"\nwrote {args.out} with {len(sample_keys)} pairs "
          f"({n_mate} true mates)")
    print(f"wrote adjacency -> {args.adjacency_out}")


if __name__ == "__main__":
    main()
