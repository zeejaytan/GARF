#!/usr/bin/env python3
"""No-GT assembly quality probes for GARF deploy results.

Juglet has no assembly ground truth, so accuracy metrics (part_acc, RMSE) are
undefined. These probes instead measure *geometric self-consistency* of a
proposed assembly, which is what "the shape closes and fractured edges meet"
means operationally:

  contact   — for each pair of posed parts, the nearest-surface gap and the
              fraction of one part's surface that sits within a small band of
              the other (i.e. do edges actually touch?), plus an
              interpenetration estimate (do parts overlap, which is wrong?).
  stability — across several runs (different seeds/anchors) of the SAME object,
              the dispersion of *relative* part poses. Anchor-free assembly has
              a global SE(3) gauge freedom, so only relative poses are
              meaningful. High dispersion ⇒ the model has no reliable mating
              signal; low dispersion but wrong ⇒ systematic bias.

Inputs are GARF's own authoritative exports (no transform re-derivation):
  - predicted_assembly.glb  : parts already placed by T_pred @ T_aug^-1
  - predicted_assembly.json : raw pred_transform (P,7) = [tx,ty,tz,qw,qx,qy,qz]

Usage
-----
  # contact probe on one assembly
  python scripts/no_gt_probes.py contact \
      --glb logs/deploy/<run>/version_0/assembly_results/artifact/Juglet-000/predicted_assembly.glb \
      --out logs/diagnostics/probes/juglet_raw

  # stability across several runs of the same object
  python scripts/no_gt_probes.py stability \
      --jsons run1/.../predicted_assembly.json run2/.../predicted_assembly.json ... \
      --out logs/diagnostics/probes/juglet_stability
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
# Geometry loading
# --------------------------------------------------------------------------- #
def load_posed_parts(glb_path: Path) -> list[tuple[str, trimesh.Trimesh]]:
    """Load a GLB scene and return [(name, mesh-in-scene-coordinates), ...].

    trimesh stores per-geometry transforms in the scene graph; we bake them in
    so each returned mesh is already in the common (assembled) frame.
    """
    scene = trimesh.load(str(glb_path), process=False)
    parts: list[tuple[str, trimesh.Trimesh]] = []
    if isinstance(scene, trimesh.Trimesh):
        return [("part00", scene)]
    for name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[name]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        m = geom.copy()
        m.apply_transform(transform)
        parts.append((str(name), m))
    return parts


def scene_scale(parts: list[tuple[str, trimesh.Trimesh]]) -> float:
    allv = np.concatenate([m.vertices for _, m in parts], axis=0)
    diag = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    return diag if diag > 1e-9 else 1.0


# --------------------------------------------------------------------------- #
# Contact / interpenetration probe
# --------------------------------------------------------------------------- #
def _surface_samples(mesh: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    except TypeError:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float64)


def _interpenetration_frac(points: np.ndarray, mesh: trimesh.Trimesh) -> float:
    """Fraction of `points` lying strictly inside `mesh`.

    Only meaningful for (near-)watertight meshes. Returns NaN otherwise so we
    never report a misleading 0.
    """
    if not mesh.is_watertight:
        return float("nan")
    try:
        inside = mesh.contains(points)
        return float(np.mean(inside))
    except Exception:
        return float("nan")


def contact_probe(
    glb_path: Path,
    n_points: int = 5000,
    tau_frac: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    parts = load_posed_parts(glb_path)
    P = len(parts)
    scale = scene_scale(parts)
    tau = tau_frac * scale

    names = [n for n, _ in parts]
    samples = [_surface_samples(m, n_points, seed + i) for i, (_, m) in enumerate(parts)]
    trees = [cKDTree(s) for s in samples]
    watertight = [bool(m.is_watertight) for _, m in parts]

    pair_rows: list[dict[str, Any]] = []
    # symmetric contact graph
    adj = np.zeros((P, P), dtype=bool)
    for i, j in combinations(range(P), 2):
        # nearest-neighbour distances both directions (surface-sample proxy)
        dij, _ = trees[j].query(samples[i])
        dji, _ = trees[i].query(samples[j])
        min_gap = float(min(dij.min(), dji.min()))
        contact_frac_i = float(np.mean(dij < tau))
        contact_frac_j = float(np.mean(dji < tau))
        contact_frac = max(contact_frac_i, contact_frac_j)
        pen_ij = _interpenetration_frac(samples[i], parts[j][1])
        pen_ji = _interpenetration_frac(samples[j], parts[i][1])
        pen_vals = [v for v in (pen_ij, pen_ji) if v == v]
        pen = max(pen_vals) if pen_vals else float("nan")
        is_contact = contact_frac > 0.01 and min_gap < tau
        if is_contact:
            adj[i, j] = adj[j, i] = True
        pair_rows.append(
            {
                "part_i": names[i],
                "part_j": names[j],
                "min_gap": min_gap,
                "min_gap_over_scale": min_gap / scale,
                "contact_frac": contact_frac,
                "interpenetration_frac": float(pen) if pen == pen else float("nan"),
                "is_contact": bool(is_contact),
            }
        )

    # connected components over the contact graph
    comp = -np.ones(P, dtype=int)
    c = 0
    for s in range(P):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = c
        while stack:
            u = stack.pop()
            for v in range(P):
                if adj[u, v] and comp[v] == -1:
                    comp[v] = c
                    stack.append(v)
        c += 1
    n_components = int(comp.max() + 1)

    gaps = np.array([r["min_gap_over_scale"] for r in pair_rows])
    contact_pairs = [r for r in pair_rows if r["is_contact"]]
    # the (P-1) smallest gaps ≈ the spanning contacts a closed object needs
    k = max(P - 1, 1)
    closest_k_gap = float(np.sort(gaps)[:k].mean()) if len(gaps) else float("nan")
    valid_pen = [r["interpenetration_frac"] for r in pair_rows if r["interpenetration_frac"] == r["interpenetration_frac"]]

    summary = {
        "glb": str(glb_path),
        "num_parts": P,
        "scene_scale": scale,
        "tau_frac": tau_frac,
        "tau_abs": tau,
        "n_contact_pairs": len(contact_pairs),
        "n_components": n_components,
        "single_object": bool(n_components == 1),
        "mean_min_gap_over_scale": float(gaps.mean()) if len(gaps) else float("nan"),
        "closest_k_gap_over_scale": closest_k_gap,
        "max_interpenetration_frac": float(np.max(valid_pen)) if valid_pen else float("nan"),
        "any_mesh_watertight": bool(any(watertight)),
        "all_meshes_watertight": bool(all(watertight)),
    }
    return {"summary": summary, "pairs": pair_rows}


# --------------------------------------------------------------------------- #
# Seed / anchor stability probe (relative poses, gauge-invariant)
# --------------------------------------------------------------------------- #
def _parse_pred_transform(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (translations (P,3), rotations (P,3,3)) from a predicted_assembly.json.

    JSON quaternion is scalar-first [w,x,y,z] (PyTorch3D/GARF convention).
    """
    with open(json_path) as f:
        j = json.load(f)
    arr = np.asarray(j["pred_transform"], dtype=np.float64)  # (P,7)
    t = arr[:, :3]
    q_wxyz = arr[:, 3:]
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    rot = R.from_quat(q_xyzw).as_matrix()  # (P,3,3)
    return t, rot


def _relative_poses(t: np.ndarray, rot: np.ndarray) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Relative pose of part j expressed in part i's frame, for all i<j.

    rel_R = R_i^T R_j ; rel_t = R_i^T (t_j - t_i). Invariant to global SE(3).
    """
    P = t.shape[0]
    out = {}
    for i, j in combinations(range(P), 2):
        rel_R = rot[i].T @ rot[j]
        rel_t = rot[i].T @ (t[j] - t[i])
        out[(i, j)] = (rel_R, rel_t)
    return out


def stability_probe(json_paths: list[Path]) -> dict[str, Any]:
    runs = [_parse_pred_transform(p) for p in json_paths]
    P = runs[0][0].shape[0]
    for t, _ in runs:
        if t.shape[0] != P:
            raise ValueError("All runs must have the same number of parts")

    rels = [_relative_poses(t, rot) for t, rot in runs]

    pair_rows: list[dict[str, Any]] = []
    rot_disps: list[float] = []
    trans_disps: list[float] = []
    for i, j in combinations(range(P), 2):
        Rs = [rel[(i, j)][0] for rel in rels]
        ts = np.stack([rel[(i, j)][1] for rel in rels], axis=0)  # (S,3)
        # rotation dispersion: mean geodesic angle to the chordal-mean rotation
        Rmean = R.from_matrix(np.stack(Rs)).mean().as_matrix()
        angles = [
            float(np.degrees(R.from_matrix(Rmean.T @ Rk).magnitude())) for Rk in Rs
        ]
        rot_disp = float(np.mean(angles))  # deg, spread across runs
        trans_disp = float(np.linalg.norm(ts - ts.mean(0), axis=1).mean())
        rot_disps.append(rot_disp)
        trans_disps.append(trans_disp)
        pair_rows.append(
            {
                "part_i": i,
                "part_j": j,
                "rel_rot_dispersion_deg": rot_disp,
                "rel_trans_dispersion": trans_disp,
            }
        )

    summary = {
        "n_runs": len(json_paths),
        "num_parts": P,
        "mean_rel_rot_dispersion_deg": float(np.mean(rot_disps)),
        "max_rel_rot_dispersion_deg": float(np.max(rot_disps)),
        "mean_rel_trans_dispersion": float(np.mean(trans_disps)),
        "max_rel_trans_dispersion": float(np.max(trans_disps)),
        "runs": [str(p) for p in json_paths],
    }
    return {"summary": summary, "pairs": pair_rows}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_outputs(result: dict[str, Any], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    pairs = result.get("pairs", [])
    if pairs:
        with open(out_dir / f"{tag}_pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
    with open(out_dir / f"{tag}_summary.md", "w") as f:
        f.write(f"# {tag} summary\n\n")
        for k, v in result["summary"].items():
            f.write(f"- **{k}**: {v}\n")
    print(f"[{tag}] wrote {out_dir}/{tag}.json (+ csv, md)")
    for k, v in result["summary"].items():
        print(f"    {k}: {v}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("contact", help="contact/interpenetration probe on one assembly")
    c.add_argument("--glb", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--tag", default="contact")
    c.add_argument("--n-points", type=int, default=5000)
    c.add_argument("--tau-frac", type=float, default=0.01)
    c.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("stability", help="relative-pose dispersion across runs")
    s.add_argument("--jsons", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--tag", default="stability")

    args = ap.parse_args()
    if args.cmd == "contact":
        res = contact_probe(args.glb, n_points=args.n_points, tau_frac=args.tau_frac, seed=args.seed)
        write_outputs(res, args.out, args.tag)
    elif args.cmd == "stability":
        res = stability_probe(args.jsons)
        write_outputs(res, args.out, args.tag)


if __name__ == "__main__":
    main()
