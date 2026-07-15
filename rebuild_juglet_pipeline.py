#!/usr/bin/env python3
"""Rebuild the full Juglet deploy pipeline from raw OBJ fragments.

Steps (all from one entry point so the chain stays consistent):
  1. Load each piece, merge duplicate vertices (raw scans are triangle-soup OBJ;
     merging makes them near-closed and shrinks storage). Report watertightness.
  2. Anchor-center: subtract the largest-extent piece's centroid from all pieces
     (removes table/scanner offset; relative layout preserved). -> Juglet_anchor_centered
  3. Local-cluster compress (alpha): move each piece centroid to
     ``gmean + alpha*(centroid - gmean)`` (global mean fixed, geometry/orientation
     preserved). alpha=0.2 reproduces the previous local02 layout. -> *_local02
  4. Build GARF/TORA HDF5s for both layouts with the original data_split keys.

Usage:
  python rebuild_juglet_pipeline.py            # full rebuild with defaults
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh

from create_juglet_hdf5 import create_juglet_hdf5, verify_hdf5

ROOT = Path("/data/gpfs/projects/punim2657")
DEFAULT_INPUT = ROOT / "Dataset/Juglet"
DEFAULT_AC = ROOT / "Dataset/Juglet_anchor_centered"
DEFAULT_LO = ROOT / "Dataset/Juglet_anchor_centered_local02"
GARF_INPUT = ROOT / "GARF/input"
TORA_DATASET = ROOT / "TORA/dataset"


def load_clean(obj_path: Path) -> trimesh.Trimesh:
    m = trimesh.load(obj_path, process=True)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    # Scans leave tiny floater shells (2-6 faces) joined at a non-manifold edge,
    # which breaks watertightness. Keep only the largest connected body.
    bodies = m.split(only_watertight=False)
    if len(bodies) > 1:
        m = max(bodies, key=lambda b: len(b.faces))
        m.merge_vertices()
        m.remove_unreferenced_vertices()
    return m


def max_pairwise(centroids: np.ndarray) -> float:
    d = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = max(d, float(np.linalg.norm(centroids[i] - centroids[j])))
    return d


def write_objs(meshes: list[trimesh.Trimesh], names: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, n in zip(meshes, names):
        m.export(out_dir / n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--ac-dir", type=Path, default=DEFAULT_AC)
    ap.add_argument("--lo-dir", type=Path, default=DEFAULT_LO)
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="Local-cluster compression factor (ignored if --target-maxd set).")
    ap.add_argument("--target-maxd", type=float, default=0.935,
                    help="Target max pairwise centroid distance for the local cluster. "
                         "If set, alpha is auto-solved as target/anchor_maxd "
                         "(default 0.935 = previous local02 tightness). Pass <=0 to use --alpha.")
    ap.add_argument("--anchor-strategy", choices=("largest_extent", "largest_area"), default="largest_extent")
    args = ap.parse_args()

    obj_files = sorted(args.input_dir.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files in {args.input_dir}")
    names = [p.name for p in obj_files]

    # 1. load + clean
    print("=== 1. Load & clean (merge vertices) ===")
    meshes = []
    for p in obj_files:
        m = load_clean(p)
        meshes.append(m)
        print(f"  {p.name}: verts {len(m.vertices)} faces {len(m.faces)} "
              f"watertight {m.is_watertight} euler {m.euler_number}")

    # 2. anchor-center
    print("=== 2. Anchor-center ===")
    if args.anchor_strategy == "largest_extent":
        anchor = int(np.argmax([float(np.max(m.extents)) for m in meshes]))
    else:
        anchor = int(np.argmax([float(m.area) for m in meshes]))
    anchor_centroid = meshes[anchor].vertices.mean(axis=0)
    print(f"  anchor=[{anchor}] {names[anchor]}  subtract centroid {anchor_centroid}")
    ac_meshes = []
    for m in meshes:
        c = m.copy()
        c.vertices = c.vertices - anchor_centroid
        ac_meshes.append(c)
    write_objs(ac_meshes, names, args.ac_dir)
    ac_centroids = np.array([m.vertices.mean(0) for m in ac_meshes])
    ac_maxd = max_pairwise(ac_centroids)
    manifest = {
        "requirement": "archaeological_deploy_step1",
        "description": "Anchor-centered scan meshes (merged/cleaned). Removes table offset only.",
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(args.ac_dir.resolve()),
        "anchor_strategy": args.anchor_strategy,
        "anchor_index": anchor,
        "anchor_piece": names[anchor],
        "anchor_centroid_subtracted": anchor_centroid.tolist(),
        "num_pieces": len(names),
        "max_pairwise_centroid_distance_after_centering": ac_maxd,
        "cleaned_merge_vertices": True,
    }
    (args.ac_dir / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {len(names)} objs -> {args.ac_dir}  (max centroid dist {ac_maxd:.4f})")

    # 3. local-cluster compress
    if args.target_maxd and args.target_maxd > 0:
        alpha = args.target_maxd / ac_maxd if ac_maxd > 0 else 0.0
        print(f"=== 3. Local-cluster compress (target maxd={args.target_maxd} -> alpha={alpha:.4f}) ===")
    else:
        alpha = args.alpha
        print(f"=== 3. Local-cluster compress (alpha={alpha}) ===")
    gmean = ac_centroids.mean(axis=0)
    lo_meshes = []
    for m, c in zip(ac_meshes, ac_centroids):
        new_c = gmean + alpha * (c - gmean)
        cc = m.copy()
        cc.vertices = cc.vertices + (new_c - c)
        lo_meshes.append(cc)
    write_objs(lo_meshes, names, args.lo_dir)
    lo_centroids = np.array([m.vertices.mean(0) for m in lo_meshes])
    lo_maxd = max_pairwise(lo_centroids)
    lo_manifest = dict(manifest)
    lo_manifest.update({
        "output_dir": str(args.lo_dir.resolve()),
        "description": f"Local-cluster compressed (alpha={alpha:.4f}) around global mean centroid.",
        "local_cluster_alpha": alpha,
        "global_mean_centroid": gmean.tolist(),
        "max_pairwise_centroid_distance_after_centering": lo_maxd,
    })
    (args.lo_dir / "deploy_manifest.json").write_text(json.dumps(lo_manifest, indent=2))
    print(f"  wrote {len(names)} objs -> {args.lo_dir}  (max centroid dist {lo_maxd:.4f})")

    # 4. build HDF5s
    print("=== 4. Build HDF5s ===")
    builds = [
        (args.ac_dir, "juglet_deploy.hdf5", ["artifact", "juglet_deploy"]),
        (args.lo_dir, "juglet_deploy_local02.hdf5", ["artifact", "juglet_deploy_local02"]),
    ]
    for src_dir, fname, split_keys in builds:
        garf_path = GARF_INPUT / fname
        create_juglet_hdf5(
            input_dir=src_dir,
            output_path=garf_path,
            sample_name="Juglet-000",
            category="artifact",
            split_keys=split_keys,
        )
        verify_hdf5(garf_path, split_keys)
        tora_path = TORA_DATASET / fname
        shutil.copy2(garf_path, tora_path)
        print(f"  copied -> {tora_path}")

    print("=== DONE ===")
    print(f"  anchor-centered max centroid dist: {ac_maxd:.4f}")
    print(f"  local-cluster   max centroid dist: {lo_maxd:.4f}")


if __name__ == "__main__":
    main()
