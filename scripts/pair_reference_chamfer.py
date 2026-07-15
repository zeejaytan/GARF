#!/usr/bin/env python3
"""Symmetry-invariant per-pair correctness for the Exp 6 pairwise oracle.

Exp 6's cross-seed rotation-dispersion proxy is confounded by per-sherd
rotational symmetry (Exp 6b control: true mates ~72 deg dispersion even on
objects GARF assembles at >=0.92 part_acc). This replaces it with a
*correctness* metric that is invariant to the ambiguous symmetry DOF:

  For a pair (piece0, piece1), sample the WHOLE assembled two-piece shape from
  GARF's prediction and from a REFERENCE assembly. Normalise each to unit
  diagonal (removes the deploy scale gauge), globally register the predicted
  pair to the reference pair with multi-init ICP (the two pieces together break
  each sherd's rotational symmetry, so the global fit is well posed), and report
  the residual chamfer / diagonal.

  Low error  => GARF reproduced the correct RELATIVE placement (a real mate it
                can perceive/align); rotating a surface-of-revolution about its
                own axis leaves the assembled shape ~unchanged, so the
                unobservable symmetry DOF is not penalised.
  High error => no global rigid alignment reconciles the two -> the predicted
                relative pose is genuinely wrong.

References:
  control : real GT — the pieces stored (already assembled) in the pairs HDF5.
  juglet  : PF++ pseudo-assembly — per-part meshes posed by the reproduced
            PF++ compute_final_transformation (see derive_pfpp_adjacency.py).

Usage
-----
  python scripts/pair_reference_chamfer.py control \
      --run-dirs logs/deploy/exp6b_ctrl_20260709_144515_s41 ..._s42 ..._s43 \
      --pairs-hdf5 input/control_ceramics_pairs.hdf5 \
      --adjacency logs/diagnostics/control_ceramics_adjacency.json \
      --out logs/diagnostics/pair_chamfer_control

  python scripts/pair_reference_chamfer.py juglet \
      --run-dirs logs/deploy/exp6_pairs_20260610_162659_s41 ..._s42 ..._s43 \
      --pairs-hdf5 input/juglet_pairs_local02.hdf5 \
      --adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
      --pfpp-dir <.../inference/juglet_deploy/0> \
      --mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
      --out logs/diagnostics/pair_chamfer_juglet
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import median, mean

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def apply(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (np.c_[v, np.ones(len(v))] @ M.T)[:, :3]


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = cKDTree(a), cKDTree(b)
    dab, _ = tb.query(a)
    dba, _ = ta.query(b)
    return float(0.5 * (np.mean(dab ** 2) + np.mean(dba ** 2)) ** 0.5)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rigid transform (no scale) mapping P->Q by correspondence. 4x4."""
    muP, muQ = P.mean(0), Q.mean(0)
    H = (P - muP).T @ (Q - muQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = muQ - R @ muP
    return M


def _icp(src: np.ndarray, dst: np.ndarray, init: np.ndarray, iters: int = 30) -> tuple[float, np.ndarray]:
    tree = cKDTree(dst)
    M = init.copy()
    cur = apply(M, src)
    for _ in range(iters):
        _, idx = tree.query(cur)
        step = _kabsch(cur, dst[idx])
        M = step @ M
        new = apply(M, src)
        if np.max(np.abs(new - cur)) < 1e-6:
            cur = new
            break
        cur = new
    return chamfer(cur, dst), M


def register_chamfer(pred: np.ndarray, ref: np.ndarray) -> float:
    """Correspondence-free: normalise both to unit diagonal @ origin, then
    multi-init ICP (24 rotations) of pred onto ref; return min residual chamfer
    (already in unit-diagonal units)."""
    def norm(x):
        c = x.mean(0)
        x = x - c
        diag = np.linalg.norm(x.max(0) - x.min(0))
        return x / (diag if diag > 1e-9 else 1.0)
    P, Q = norm(pred), norm(ref)
    # 24 axis-aligned rotations (octahedral) as inits to escape symmetry minima
    inits = []
    axes = [np.eye(3)[i] for i in range(3)]
    for ax in range(3):
        for k in range(4):
            th = k * np.pi / 2
            R = _axis_rot(axes[ax], th)
            for flip_ax in range(3):
                Rf = _axis_rot(axes[flip_ax], np.pi) @ R
                m = np.eye(4); m[:3, :3] = Rf
                inits.append(m)
    m0 = np.eye(4)
    inits.append(m0)
    best = np.inf
    for init in inits:
        c, _ = _icp(P, Q, init, iters=20)
        if c < best:
            best = c
    return best


def _axis_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    a = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = a
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


def sample(v: np.ndarray, f: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return np.asarray(trimesh.sample.sample_surface(m, n)[0])


def load_glb_pieces(glb: Path):
    """Return list of (verts, faces) in scene (assembled) coords, ordered by
    descending vertex count so index 0 is the anchor (largest) piece."""
    sc = trimesh.load(str(glb), process=False)
    out = []
    for name in sc.graph.nodes_geometry:
        tf, gname = sc.graph[name]
        g = sc.geometry[gname]
        v = apply(tf, np.asarray(g.vertices, dtype=np.float64))
        out.append((v, np.asarray(g.faces)))
    out.sort(key=lambda vf: -len(vf[0]))
    return out


# --------------------------------------------------------------------------- #
# PF++ reference (juglet)
# --------------------------------------------------------------------------- #
def quat_to_matrix(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)]])


def _T(t):
    m = np.eye(4); m[:3, 3] = t; return m


def _R(q):
    m = np.eye(4); m[:3, :3] = quat_to_matrix(q); return m


def pfpp_final(init, gt, pred):
    return (_R(init[3:]) @ _T(init[:3]) @ _T(pred[:3]) @ _R(pred[3:])
            @ _R(gt[3:]).T @ _T(-gt[:3]) @ _T(-init[:3]) @ _R(init[3:]).T)


def build_pfpp_reference(pfpp_dir: Path, mesh_dir: Path):
    """Return dict piece_index(0-based) -> posed mesh (verts, faces)."""
    init = np.load(pfpp_dir / "init_pose.npy").astype(np.float64)
    gt = np.load(pfpp_dir / "gt.npy").astype(np.float64)
    traj = np.load(sorted(glob.glob(str(pfpp_dir / "predict_*.npy")))[0]).astype(np.float64)
    pred = traj[-1]
    objs = sorted(p for p in mesh_dir.iterdir() if p.suffix == ".obj")
    ref = {}
    for i, obj in enumerate(objs):
        m = trimesh.load(str(obj), process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        v = np.asarray(m.vertices, dtype=np.float64)
        # per-part recenter + unit scale, exactly as PF++ prep, then final transform
        v = v - v.mean(0)
        s = np.max(np.abs(v))
        v = v / (s if s > 0 else 1.0)
        v = apply(pfpp_final(init, gt[i], pred[i]), v)
        ref[i] = (v, np.asarray(m.faces))
    return ref


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def pair_error(pred_pieces, ref0, ref1, n=4000) -> float:
    """pred_pieces: [(v,f),(v,f)]. ref0/ref1: (v,f) reference pieces.

    Build the whole assembled 2-piece point cloud for prediction and reference,
    then correspondence-free register (multi-init ICP) pred->ref and return the
    residual chamfer in unit-diagonal units. Symmetry-invariant because ICP is
    free to rotate a surface-of-revolution about its own axis.
    """
    (pv0, pf0), (pv1, pf1) = pred_pieces
    (rv0, rf0), (rv1, rf1) = ref0, ref1
    pred_pts = np.concatenate([sample(pv0, pf0, n), sample(pv1, pf1, n)], axis=0)
    ref_pts = np.concatenate([sample(rv0, rf0, n), sample(rv1, rf1, n)], axis=0)
    return register_chamfer(pred_pts, ref_pts)


def find_pairs(run_dir: Path, subdir: str) -> dict[str, Path]:
    root = run_dir / "version_0" / "assembly_results" / subdir
    out = {}
    for glb in sorted(root.glob("*/predicted_assembly.glb")):
        out[glb.parent.name] = glb.parent
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=["control", "juglet"])
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--pairs-hdf5", type=Path, required=True)
    ap.add_argument("--adjacency", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pfpp-dir", type=Path)
    ap.add_argument("--mesh-dir", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    subdir = "control" if args.dataset == "control" else "artifact"
    per_run = {d: find_pairs(d, subdir) for d in args.run_dirs}
    samples = sorted({s for m in per_run.values() for s in m})
    if not samples:
        raise SystemExit(f"no {subdir} pair GLBs found under run dirs")
    print(f"{len(samples)} pairs across {len(args.run_dirs)} runs")

    hf = h5py.File(args.pairs_hdf5, "r")

    # adjacency label lookup
    adj_raw = json.load(open(args.adjacency))
    if args.dataset == "control":
        adj = {k: v["true_mate"] for k, v in adj_raw["pairs"].items()}
    else:
        mates = set(adj_raw["true_mates"])  # e.g. p0102
    pfpp_ref = None
    if args.dataset == "juglet":
        pfpp_ref = build_pfpp_reference(args.pfpp_dir, args.mesh_dir)

    def ref_pieces_for(sample_key: str, hdf5_key: str):
        """Return (ref0, ref1) anchor-first for this sample."""
        if args.dataset == "control":
            g = hf[hdf5_key]["pieces"]
            keys = sorted(g.keys(), key=int)
            pcs = [(np.asarray(g[k]["vertices"][:], np.float64),
                    np.asarray(g[k]["faces"][:])) for k in keys]
        else:
            # juglet sample 'Juglet-pXXYY' -> global piece indices XX,YY (1-based)
            tag = sample_key.split("-p")[-1]
            i, j = int(tag[:2]) - 1, int(tag[2:]) - 1
            pcs = [pfpp_ref[i], pfpp_ref[j]]
        pcs.sort(key=lambda vf: -len(vf[0]))
        return pcs[0], pcs[1]

    rows = []
    for s in samples:
        # hdf5 key
        if args.dataset == "control":
            # GLB parent name == '<obj>__pIJ'; hdf5 key 'control/<obj>__pIJ'
            hdf5_key = f"control/{s}"
            is_mate = adj.get(s, False)
        else:
            hdf5_key = f"artifact/{s}"
            is_mate = ("p" + s.split("-p")[-1]) in mates
        ref0, ref1 = ref_pieces_for(s, hdf5_key)
        errs = []
        for d in args.run_dirs:
            adir = per_run[d].get(s)
            if adir is None:
                continue
            pred_pieces = load_glb_pieces(adir / "predicted_assembly.glb")
            if len(pred_pieces) != 2:
                continue
            e = pair_error(pred_pieces, ref0, ref1)
            if e == e:
                errs.append(e)
        med = median(errs) if errs else float("nan")
        rows.append({"sample": s, "true_mate": bool(is_mate),
                     "n_runs": len(errs), "chamfer_over_diag": med})
        print(f"  {s}: mate={is_mate} chamfer/diag={med:.4f} (n={len(errs)})")

    def stat(sel):
        v = [r["chamfer_over_diag"] for r in rows if r["true_mate"] == sel
             and r["chamfer_over_diag"] == r["chamfer_over_diag"]]
        return (len(v), mean(v) if v else float("nan"),
                median(v) if v else float("nan"))
    nm, mm, mmed = stat(True)
    nn, nmn, nmed = stat(False)

    md = [f"# Pair correctness (symmetry-invariant chamfer) — {args.dataset}\n",
          f"Metric: chamfer(piece1 aligned by piece0, reference piece1) / pair diagonal.",
          f"Reference: {'real GT' if args.dataset=='control' else 'PF++ pseudo-assembly'}.\n",
          "| set | n | mean chamfer/diag | median |",
          "|---|---|---|---|",
          f"| TRUE MATES | {nm} | {mm:.4f} | {mmed:.4f} |",
          f"| non-mates  | {nn} | {nmn:.4f} | {nmed:.4f} |\n",
          "## Per-pair (sorted by chamfer)\n",
          "| pair | true mate | chamfer/diag | n |",
          "|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["chamfer_over_diag"]
                    if r["chamfer_over_diag"] == r["chamfer_over_diag"] else 1e9)):
        md.append(f"| {r['sample']} | {'YES' if r['true_mate'] else ''} "
                  f"| {r['chamfer_over_diag']:.4f} | {r['n_runs']} |")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    with open(args.out / "pairs.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nTRUE MATES chamfer/diag: mean {mm:.4f} median {mmed:.4f} (n={nm})")
    print(f"non-mates  chamfer/diag: mean {nmn:.4f} median {nmed:.4f} (n={nn})")
    print(f"wrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
