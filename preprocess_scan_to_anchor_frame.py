#!/usr/bin/env python3
"""Step 1 of archaeological deployment: anchor-center scanned fragment meshes.

Deploy requirement (see ``ARCHAEOLOGICAL_DEPLOYMENT.md`` — Step 1):
  Run this on raw Metashape/table OBJ exports **before** GARF/TORA/PF++ inference.
  Skipping it leaves fragments metres apart in world coordinates and breaks
  denoiser scale assumptions, even though no true assembly GT exists.

Real-world scans (Metashape, etc.) store fragments in table/scanner coordinates,
often metres apart. Training data uses fragments in a local broken layout. This
script subtracts the anchor fragment centroid from every piece so all meshes
share a common origin without assuming a true assembly is known.

Writes:
  <output_dir>/*.obj          — anchor-centered meshes (same filenames)
  <output_dir>/deploy_manifest.json — anchor index, paths, bbox stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def pick_anchor_index(meshes: list[trimesh.Trimesh], strategy: str) -> int:
    if strategy == "largest_extent":
        extents = [float(np.max(m.extents)) for m in meshes]
        return int(np.argmax(extents))
    if strategy == "largest_area":
        areas = [float(m.area) for m in meshes]
        return int(np.argmax(areas))
    if strategy == "largest_vertex_count":
        counts = [len(m.vertices) for m in meshes]
        return int(np.argmax(counts))
    raise ValueError(f"Unknown anchor strategy: {strategy}")


def anchor_center_meshes(
    input_dir: Path,
    output_dir: Path,
    anchor_strategy: str = "largest_extent",
    anchor_index: int | None = None,
) -> dict:
    obj_files = sorted(input_dir.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files in {input_dir}")

    meshes: list[trimesh.Trimesh] = []
    for p in obj_files:
        m = trimesh.load(p, process=False)
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            raise ValueError(f"{p}: need a mesh with faces")
        meshes.append(m)

    if anchor_index is None:
        anchor_index = pick_anchor_index(meshes, anchor_strategy)

    anchor_centroid = meshes[anchor_index].vertices.mean(axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = []
    max_pairwise = 0.0
    centroids_raw = []
    centroids_centered = []

    for i, (obj_path, mesh) in enumerate(zip(obj_files, meshes)):
        raw_c = mesh.vertices.mean(axis=0)
        centered = mesh.copy()
        centered.vertices = centered.vertices - anchor_centroid
        out_path = output_dir / obj_path.name
        centered.export(out_path)

        c = centered.vertices.mean(axis=0)
        centroids_raw.append(raw_c.tolist())
        centroids_centered.append(c.tolist())
        pieces.append(
            {
                "index": i,
                "filename": obj_path.name,
                "is_anchor": i == anchor_index,
                "output_path": str(out_path),
                "extent_max": float(np.max(mesh.extents)),
            }
        )

    for i in range(len(centroids_centered)):
        for j in range(i + 1, len(centroids_centered)):
            d = float(
                np.linalg.norm(
                    np.array(centroids_centered[i]) - np.array(centroids_centered[j])
                )
            )
            max_pairwise = max(max_pairwise, d)

    manifest = {
        "requirement": "archaeological_deploy_step1",
        "description": (
            "Anchor-centered scan meshes for inference without true assembly GT. "
            "Does not assert fragments belong together; only removes table/scanner offset."
        ),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "anchor_strategy": anchor_strategy,
        "anchor_index": anchor_index,
        "anchor_piece": obj_files[anchor_index].name,
        "anchor_centroid_subtracted": anchor_centroid.tolist(),
        "num_pieces": len(obj_files),
        "max_pairwise_centroid_distance_after_centering": max_pairwise,
        "pieces": pieces,
        "centroids_raw_scan": centroids_raw,
        "centroids_after_anchor_centering": centroids_centered,
    }

    manifest_path = output_dir / "deploy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Anchor: [{anchor_index}] {obj_files[anchor_index].name}")
    print(f"Subtracted centroid: {anchor_centroid}")
    print(f"Max pairwise distance after centering: {max_pairwise:.4f}")
    print(f"Wrote {len(obj_files)} OBJ files -> {output_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/gpfs/projects/punim2657/Dataset/Juglet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gpfs/projects/punim2657/Dataset/Juglet_anchor_centered"),
    )
    parser.add_argument(
        "--anchor-strategy",
        choices=("largest_extent", "largest_area", "largest_vertex_count"),
        default="largest_extent",
    )
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=None,
        help="Force anchor piece index (0-based, sorted OBJ order). Overrides --anchor-strategy.",
    )
    args = parser.parse_args()
    anchor_center_meshes(
        args.input_dir,
        args.output_dir,
        anchor_strategy=args.anchor_strategy,
        anchor_index=args.anchor_index,
    )


if __name__ == "__main__":
    main()
