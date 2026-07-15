#!/usr/bin/env python3
"""Domain-bridge generator: morph a known-good GARF object toward Juglet traits.

Writes a single-sample GARF HDF5 where each fragment's geometry is deformed in
its OWN local frame (stored assembled position/orientation preserved), so the
benchmark GT (part_acc / rmse / shape_cd) stays valid throughout. Sweeping the
strength of one transform at a time isolates which Juglet-like factor breaks
GARF.

Transforms (one factor each):
  none    : passthrough (strength-0 sanity gate; must reproduce baseline acc)
  noise   : displace vertices along normals ~ N(0, strength * frag_scale)  [H2]
  decimate: quadric decimation to `strength` fraction of faces              [H3]
  open    : drop `strength` fraction of faces -> non-watertight shell        [H3]
  erode   : round sharp fracture-rim creases toward Juglet-like wear           [H5]
  densify : subdivide fracture-rim faces (NOT a sampling remedy: GARF samples
            area-weighted, so subdivision is a no-op for point density; the
            remedy is the sampler-side data.rim_oversample_frac option)

Usage:
  python scripts/domain_bridge.py \
      --source /path/breaking_bad_vol.hdf5 --split-key everyday --pieces 6 \
      --transform noise --strength 0.02 --seed 0 \
      --out logs/diagnostics/bridge_subsets/bb_6pc_noise002.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import trimesh

from fracture_mesh_ops import densify_fracture_rim, erode_fracture_rim
from garf_matched_diagnostic import _decode, _numeric_piece_keys, pick_sample_with_parts

SHELL_VOXEL_RES = 48  # voxels across the fragment's largest extent


# --------------------------------------------------------------------------- #
# Per-fragment transforms: take (verts, faces) -> (verts, faces) [+ shared]
# --------------------------------------------------------------------------- #
def _frag_scale(verts: np.ndarray) -> float:
    ext = verts.max(0) - verts.min(0)
    return float(max(ext)) if ext.size else 1.0


def t_none(verts, faces, rng, strength):
    return verts, faces


def t_noise(verts, faces, rng, strength):
    """Displace each vertex along its (area-weighted) normal by N(0, strength*scale)."""
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy(), process=False)
    normals = np.asarray(mesh.vertex_normals)  # (V,3)
    scale = _frag_scale(verts)
    disp = rng.normal(0.0, strength * scale, size=(verts.shape[0], 1))
    new_v = verts + normals * disp
    return new_v.astype(np.float64), faces


def t_decimate(verts, faces, rng, strength):
    """Quadric decimation to `strength` fraction of faces (0<strength<=1)."""
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy(), process=False)
    target = max(4, int(len(faces) * strength))
    try:
        dec = mesh.simplify_quadric_decimation(face_count=target)
    except TypeError:
        dec = mesh.simplify_quadric_decimation(percent=float(strength))
    return np.asarray(dec.vertices, dtype=np.float64), np.asarray(dec.faces, dtype=np.int64)


def t_open(verts, faces, rng, strength):
    """Remove `strength` fraction of faces -> non-watertight, single-sided-ish."""
    n = len(faces)
    keep = rng.random(n) >= strength
    if keep.sum() < 4:
        keep[:4] = True
    return verts, faces[keep]


def t_erode(verts, faces, rng, strength):
    """Round fracture-rim creases toward archaeological wear (Exp 7 confirmation)."""
    return erode_fracture_rim(verts, faces, strength)


def t_densify(verts, faces, rng, strength):
    """Subdivide rim-band faces (resolution aid only — no effect on area-weighted sampling)."""
    return densify_fracture_rim(verts, faces, strength)


def t_shell(verts, faces, rng, strength):
    """Hollow a solid fragment into a thin shell of wall thickness ~ strength*frag_scale.

    Voxelize+fill the solid, erode the interior by `t_vox` voxels, keep the boundary
    band (solid AND NOT eroded) -> a two-walled shell, then re-mesh via marching cubes
    in the SAME world frame (GT assembled pose preserved). Mimics Juglet's thin pottery
    walls (H1). Falls back to the original mesh if voxelization fails (e.g. non-solid)."""
    try:
        import scipy.ndimage as ndi
        from trimesh.voxel import ops as vox_ops

        mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy(), process=False)
        scale = _frag_scale(verts)
        pitch = scale / SHELL_VOXEL_RES
        vox = mesh.voxelized(pitch).fill()
        mat = vox.matrix
        t_vox = max(1, int(round(strength * scale / pitch)))
        eroded = ndi.binary_erosion(mat, iterations=t_vox)
        shell = mat & ~eroded
        if shell.sum() < 8:
            return verts, faces
        sm = vox_ops.matrix_to_marching_cubes(shell)
        sm.apply_transform(vox.transform)
        if len(sm.faces) < 4:
            return verts, faces
        return np.asarray(sm.vertices, dtype=np.float64), np.asarray(sm.faces, dtype=np.int64)
    except Exception as exc:  # noqa: BLE001 - keep the sweep running
        print(f"  [t_shell] fallback to original ({exc})")
        return verts, faces


TRANSFORMS = {
    "none": t_none,
    "noise": t_noise,
    "decimate": t_decimate,
    "open": t_open,
    "shell": t_shell,
    "erode": t_erode,
    "densify": t_densify,
}


# --------------------------------------------------------------------------- #
# HDF5 writer (mirrors garf_matched_diagnostic.build_garf_subset_hdf5 schema)
# --------------------------------------------------------------------------- #
def build_bridge_hdf5(source: Path, sample: str, out: Path, transform: str, strength: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fn = TRANSFORMS[transform]
    out.parent.mkdir(parents=True, exist_ok=True)

    info = {"sample": sample, "transform": transform, "strength": strength, "parts": []}
    with h5py.File(source, "r") as src, h5py.File(out, "w") as dst:
        src_group = src[sample]
        if "pieces" in src_group:
            src_pieces = src_group["pieces"]
            keys = sorted(src_pieces.keys(), key=lambda x: int(x))
            getp = lambda k: src_pieces[k]
        else:
            keys = _numeric_piece_keys(src_group)
            getp = lambda k: src_group[k]

        dst_group = dst.create_group(sample)
        dst_pieces = dst_group.create_group("pieces")
        piece_names = []
        for i, k in enumerate(keys):
            p = getp(k)
            verts = np.asarray(p["vertices"][:], dtype=np.float64)
            faces = np.asarray(p["faces"][:], dtype=np.int64)
            nv0, nf0 = len(verts), len(faces)
            new_v, new_f = fn(verts, faces, rng, strength)
            new_f = np.asarray(new_f, dtype=np.int64)
            dp = dst_pieces.create_group(str(i))
            dp.create_dataset("vertices", data=np.asarray(new_v, dtype=np.float64))
            dp.create_dataset("faces", data=new_f)
            # shared_faces not used at inference; keep length consistent with faces.
            dp.create_dataset("shared_faces", data=-np.ones(len(new_f), dtype=np.int64))
            piece_names.append(f"Piece{i + 1:02d}".encode("utf-8"))
            info["parts"].append(
                {"idx": i, "v_in": nv0, "f_in": nf0, "v_out": len(new_v), "f_out": len(new_f)}
            )

        # prefer real piece names if present
        if "pieces_names" in src_group:
            piece_names = [_decode(n).encode("utf-8") for n in src_group["pieces_names"][:]]

        num = len(piece_names)
        dst_group.create_dataset("pieces_names", data=piece_names, dtype=h5py.special_dtype(vlen=str))
        dst_group.create_dataset("removal_masks", data=np.ones((1, num), dtype=bool))
        dst_group.create_dataset("removal_order", data=np.arange(num, dtype=np.int64))

        ds_root = dst.create_group("data_split")
        diag = ds_root.create_group("diag")
        ref = np.array([sample.encode("utf-8")], dtype=object)
        for split_name in ("train", "val", "test"):
            diag.create_dataset(split_name, data=ref)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--split-key", default=None)
    ap.add_argument("--pieces", type=int, default=None)
    ap.add_argument("--sample", default=None, help="Explicit sample name (overrides --pieces).")
    ap.add_argument("--transform", choices=list(TRANSFORMS), required=True)
    ap.add_argument("--strength", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sample = args.sample
    if sample is None:
        if args.split_key is None or args.pieces is None:
            ap.error("Provide --sample, or both --split-key and --pieces")
        with h5py.File(args.source, "r") as h5:
            sample = pick_sample_with_parts(h5, args.split_key, args.pieces)

    info = build_bridge_hdf5(args.source, sample, args.out, args.transform, args.strength, args.seed)
    print(f"sample={info['sample']}")
    print(f"transform={info['transform']} strength={info['strength']}")
    print(f"out={args.out}")
    fin = sum(p["f_in"] for p in info["parts"])
    fout = sum(p["f_out"] for p in info["parts"])
    print(f"faces total in={fin} out={fout}")


if __name__ == "__main__":
    main()
