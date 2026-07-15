#!/usr/bin/env python3
"""Exp 6 — build an HDF5 with every 2-piece Juglet subset as its own sample.

Pairwise oracle decomposition: if GARF cannot align ANY true mating pair in
isolation, the failure is perceptual (no usable rim signal); if pairs align
but the joint 9-piece run fails, the failure is in joint inference/search.

Pieces are copied verbatim from the canonical rebuilt deploy HDF5 (default
``input/juglet_deploy_local02.hdf5``) so pair geometry is bit-identical to the
full-object runs. Each of the C(9,2)=36 pairs becomes
``artifact/Juglet-p<i><j>`` (1-based piece numbers, e.g. Juglet-p0102) and all
pairs are listed in ``data_split/artifact/{train,val,test}`` so one eval run
covers them all.

Usage:
  python scripts/build_juglet_pairs_hdf5.py \
      --source input/juglet_deploy_local02.hdf5 \
      --sample artifact/Juglet-000 \
      --out input/juglet_pairs_local02.hdf5
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np


def copy_piece(src_piece: h5py.Group, dst_piece: h5py.Group) -> None:
    verts = np.asarray(src_piece["vertices"][:], dtype=np.float64)
    faces = np.asarray(src_piece["faces"][:], dtype=np.int64)
    dst_piece.create_dataset("vertices", data=verts)
    dst_piece.create_dataset("faces", data=faces)
    if "shared_faces" in src_piece:
        shared = np.asarray(src_piece["shared_faces"][:], dtype=np.int64)
    else:
        shared = np.zeros(len(faces), dtype=np.int64)
    dst_piece.create_dataset("shared_faces", data=shared)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("input/juglet_deploy_local02.hdf5"))
    ap.add_argument("--sample", default="artifact/Juglet-000")
    ap.add_argument("--out", type=Path, default=Path("input/juglet_pairs_local02.hdf5"))
    ap.add_argument("--category", default="artifact")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pname_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(args.source, "r") as src, h5py.File(args.out, "w") as dst:
        src_pieces = src[args.sample]["pieces"]
        keys = sorted(src_pieces.keys(), key=int)
        n = len(keys)
        if "pieces_names" in src[args.sample]:
            names = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in src[args.sample]["pieces_names"][:]
            ]
        else:
            names = [f"Piece{i + 1:02d}" for i in range(n)]
        print(f"source: {args.source} sample {args.sample} ({n} pieces: {names})")

        sample_keys: list[bytes] = []
        for a, b in combinations(range(n), 2):
            sample_name = f"{args.category}/Juglet-p{a + 1:02d}{b + 1:02d}"
            grp = dst.create_group(sample_name)
            pieces = grp.create_group("pieces")
            copy_piece(src_pieces[keys[a]], pieces.create_group("0"))
            copy_piece(src_pieces[keys[b]], pieces.create_group("1"))
            grp.create_dataset(
                "pieces_names",
                data=[names[a].encode("utf-8"), names[b].encode("utf-8")],
                dtype=pname_dtype,
            )
            grp.create_dataset("removal_masks", data=np.ones((1, 2), dtype=bool))
            grp.create_dataset("removal_order", data=np.arange(2, dtype=np.int64))
            sample_keys.append(sample_name.encode("utf-8"))

        ds_root = dst.create_group("data_split")
        split = ds_root.create_group(args.category)
        ref = np.array(sample_keys, dtype=object)
        for split_name in ("train", "val", "test"):
            split.create_dataset(split_name, data=ref)

    print(f"wrote {args.out} with {len(sample_keys)} pair samples")
    with h5py.File(args.out, "r") as f:
        val = [x.decode("utf-8") for x in f["data_split"][args.category]["val"][:]]
        print(f"data_split[{args.category}] val ({len(val)}):")
        for s in val[:3]:
            v0 = f[s]["pieces"]["0"]["vertices"].shape
            v1 = f[s]["pieces"]["1"]["vertices"].shape
            print(f"  {s}: piece0 verts {v0}, piece1 verts {v1}")
        print("  ...")


if __name__ == "__main__":
    main()
