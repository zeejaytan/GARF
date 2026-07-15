#!/usr/bin/env python3
"""Create GARF-compatible HDF5 from scanned archaeological OBJ fragments.

Follows the tray_archaeological layout (see GARF_ARCHITECTURE.md):
  artifact/<sample>/pieces/<i>/{vertices, faces, shared_faces}
  data_split/<key>/val -> [b'artifact/<sample>']  (object-dtype array, full path)

Use ``--split-keys artifact,juglet_deploy`` when one HDF5 must satisfy both GARF
(``data.categories=['artifact']``) and TORA (filename stem ``juglet_deploy`` →
``data_split/juglet_deploy``). The mesh group path stays ``category/sample_name``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import trimesh


def load_obj_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(obj_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{obj_path}: expected Trimesh, got {type(mesh)}")
    if len(mesh.faces) == 0:
        raise ValueError(f"{obj_path}: no faces (need watertight/surface mesh)")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(
        mesh.faces, dtype=np.int64
    )


def create_juglet_hdf5(
    input_dir: Path,
    output_path: Path,
    sample_name: str = "Juglet-000",
    category: str = "artifact",
    split_keys: list[str] | None = None,
) -> None:
    obj_files = sorted(input_dir.glob("*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No .obj files in {input_dir}")

    sample_key = f"{category}/{sample_name}"
    keys = split_keys if split_keys else [category]
    print(f"Creating {output_path}")
    print(f"  input:  {input_dir} ({len(obj_files)} pieces)")
    print(f"  sample: {sample_key}")
    print(f"  data_split keys: {keys}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        sample_group = f.create_group(sample_key)
        pieces_group = sample_group.create_group("pieces")
        piece_names: list[bytes] = []

        for i, obj_path in enumerate(obj_files):
            print(f"  [{i + 1}/{len(obj_files)}] {obj_path.name}")
            vertices, faces = load_obj_mesh(obj_path)
            piece_group = pieces_group.create_group(str(i))
            piece_group.create_dataset("vertices", data=vertices)
            piece_group.create_dataset("faces", data=faces)
            piece_group.create_dataset(
                "shared_faces", data=np.zeros(len(faces), dtype=np.int64)
            )
            piece_names.append(obj_path.stem.encode("utf-8"))

        pieces_names_dtype = h5py.special_dtype(vlen=str)
        sample_group.create_dataset(
            "pieces_names", data=piece_names, dtype=pieces_names_dtype
        )

        num_pieces = len(piece_names)
        sample_group.create_dataset(
            "removal_masks", data=np.ones((1, num_pieces), dtype=bool)
        )
        sample_group.create_dataset(
            "removal_order", data=np.arange(num_pieces, dtype=np.int64)
        )

        ds_root = f.create_group("data_split")
        sample_ref = np.array([sample_key.encode("utf-8")], dtype=object)
        for key in keys:
            split_group = ds_root.create_group(key)
            for split_name in ("val", "train", "test"):
                split_group.create_dataset(split_name, data=sample_ref)

    print(f"Wrote {output_path}")


def verify_hdf5(hdf5_path: Path, split_keys: list[str]) -> None:
    with h5py.File(hdf5_path, "r") as f:
        for category in split_keys:
            val = [x.decode("utf-8") for x in f["data_split"][category]["val"][:]]
            print(f"data_split[{category}] val:", val)
            for sample in val:
                n = len(f[sample]["pieces"].keys())
                v0 = f[sample]["pieces"]["0"]["vertices"]
                print(f"  {sample}: {n} pieces, piece0 verts {v0.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/gpfs/projects/punim2657/Dataset/Juglet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("input/juglet_archaeological.hdf5"),
    )
    parser.add_argument("--sample-name", default="Juglet-000")
    parser.add_argument("--category", default="artifact")
    parser.add_argument(
        "--split-keys",
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated names under data_split/ (default: same as --category). "
            "Example: artifact,juglet_deploy for GARF + TORA from one juglet_deploy.hdf5."
        ),
    )
    parser.add_argument("--verify", action="store_true", default=True)
    args = parser.parse_args()

    sk = (
        [s.strip() for s in args.split_keys.split(",") if s.strip()]
        if args.split_keys
        else None
    )

    create_juglet_hdf5(
        input_dir=args.input_dir,
        output_path=args.output,
        sample_name=args.sample_name,
        category=args.category,
        split_keys=sk,
    )
    if args.verify:
        verify_hdf5(args.output, sk if sk else [args.category])


if __name__ == "__main__":
    main()
