#!/usr/bin/env python3
"""T0 — like-for-like no-GT layout quality panel (PF++ success-factor plan).

Scores a proposed 9-piece assembly on five no-GT metrics and compares arms:

  pfpp     — PF++ final layout (denoiser trajectory last step, mapped through
             the verified compute_final_transformation chain onto the scan-
             frame clouds/meshes)
  garf     — GARF deploy layout (predicted_assembly.glb, fixed export)
  random   — null: parts piled compactly at random poses (N draws)
  permuted — null: PF++ slot transforms with part identities deranged (N draws)

Metrics (per layout):
  compactness       max pairwise centroid distance / mean part diagonal
  coarse_adjacency  touching pairs at derive_pfpp_adjacency thresholds
                    (min_gap/diag < 0.03 AND contact_frac@0.03·diag > 0.03)
  fine_contact      per-pair contact fraction at tau = 0.005·diag; report
                    pairs with cfrac > 0.01 (rim-level contact, Exp 9 spirit)
  interpenetration  fraction of surface samples of part i strictly inside
                    part j (watertight meshes required; skipped for
                    points-only arms)
  vesselness        surface-of-revolution fit: optimise axis (dir + point),
                    bin height, residual = mean per-bin radius std / diag;
                    also fraction of points within 5% of per-bin median radius

Usage (typical, from GARF repo root on Spartan):
  python scripts/pfpp_layout_probes.py \
      --pfpp-dir  /data/.../Puzzlefusion/output/denoiser/everyday_epoch2000_bs64/inference/juglet_deploy/0 \
      --pc-npz    /data/.../Puzzlefusion/data/pc_data/juglet_deploy/val/00000.npz \
      --mesh-dir  /data/.../Dataset/Juglet_anchor_centered \
      --garf-glb  logs/deploy/<run>/version_0/assembly_results/artifact/Juglet-000/predicted_assembly.glb \
      --out       logs/diagnostics/pfpp_t0_<ts> \
      [--n-null 5] [--seed 0] [--interp-points 300] [--no-glb-export]

Later experiments (T4/T1/T2 arms) rescore single PF++ runs:
  python scripts/pfpp_layout_probes.py --pfpp-dir <run>/0 --pc-npz <npz> \
      --mesh-dir <dir> --tag t4_shuffle --out <out>
(--mesh-dir optional; without it interpenetration is skipped.)
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

try:
    import trimesh
except ImportError:  # points-only fallback
    trimesh = None

# ---------------------------------------------------------------- transforms
# (identical chain to derive_pfpp_adjacency.py, kept in sync)


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def Tmat(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def Rmat(q):
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q)
    return m


def final_transformation(init_pose, gt, pred):
    return (Rmat(init_pose[3:]) @ Tmat(init_pose[:3]) @ Tmat(pred[:3]) @ Rmat(pred[3:])
            @ Rmat(gt[3:]).T @ Tmat(-gt[:3]) @ Tmat(-init_pose[:3]) @ Rmat(init_pose[3:]).T)


def apply(mat, v):
    return (np.c_[v, np.ones(len(v))] @ mat.T)[:, :3]


# ---------------------------------------------------------------- loaders


def load_pfpp_layout(pfpp_dir: Path, pc_npz: Path):
    """Return (scan_clouds [list P x (N,3)], slot_mats [list P x 4x4])."""
    init_pose = np.load(pfpp_dir / "init_pose.npy").astype(np.float64)
    gt = np.load(pfpp_dir / "gt.npy").astype(np.float64)
    preds = sorted(glob.glob(str(pfpp_dir / "predict_*.npy")))
    if not preds:
        raise FileNotFoundError(f"no predict_*.npy in {pfpp_dir}")
    traj = np.load(preds[0]).astype(np.float64)
    pred_final = traj[-1]
    pcs = np.load(pc_npz, allow_pickle=True)["part_pcs_gt"].astype(np.float64)
    P = pcs.shape[0]
    ident = final_transformation(init_pose, gt[0], gt[0])
    assert np.abs(ident - np.eye(4)).max() < 1e-6, "chain identity check failed"
    mats = [final_transformation(init_pose, gt[i], pred_final[i]) for i in range(P)]
    return [pcs[i] for i in range(P)], mats


def load_garf_glb(glb_path: Path, n_points: int, seed: int):
    """Return (posed_clouds, posed_meshes, names) from a GARF deploy GLB.

    Same traversal as no_gt_probes.load_posed_parts: bake scene-graph
    transforms so every mesh is in the common assembled frame.
    """
    scene = trimesh.load(str(glb_path), process=False)
    clouds, meshes, names = [], [], []
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    for name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[name]
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
        except TypeError:
            pts, _ = trimesh.sample.sample_surface(mesh, n_points)
        clouds.append(np.asarray(pts, dtype=np.float64))
        meshes.append(mesh)
        names.append(str(name))
    return clouds, meshes, names


def match_meshes_to_parts(scan_clouds, mesh_dir: Path):
    """Match npz part order to Piece*.obj by nearest-surface distance."""
    files = sorted(mesh_dir.glob("Piece*.obj"))
    if not files:
        return None, None
    meshes = [trimesh.load(f, force="mesh") for f in files]
    trees = [cKDTree(m.vertices) for m in meshes]
    assign = []
    for pc in scan_clouds:
        sub = pc[:: max(1, len(pc) // 200)]
        dists = [t.query(sub)[0].mean() for t in trees]
        assign.append(int(np.argmin(dists)))
    if len(set(assign)) != len(scan_clouds):
        print(f"WARNING: mesh matching not a bijection: {assign}")
        return None, None
    print("mesh match:", {i: files[a].name for i, a in enumerate(assign)})
    return [meshes[a] for a in assign], [files[a].name for a in assign]


# ---------------------------------------------------------------- metrics


def part_diag(pc):
    return float(np.linalg.norm(pc.max(0) - pc.min(0)))


def compactness(clouds):
    cents = np.stack([c.mean(0) for c in clouds])
    maxd = max(np.linalg.norm(cents[i] - cents[j])
               for i, j in combinations(range(len(clouds)), 2))
    return float(maxd / np.mean([part_diag(c) for c in clouds]))


def pair_metrics(clouds, gap_tau=0.03, contact_tau=0.03, fine_tau=0.005):
    allp = np.concatenate(clouds, 0)
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    trees = [cKDTree(c) for c in clouds]
    coarse, fine, rows = 0, 0, []
    for i, j in combinations(range(len(clouds)), 2):
        dij = trees[j].query(clouds[i])[0]
        dji = trees[i].query(clouds[j])[0]
        min_gap = float(min(dij.min(), dji.min())) / diag
        cfrac = float(max(np.mean(dij < contact_tau * diag),
                          np.mean(dji < contact_tau * diag)))
        ffrac = float(max(np.mean(dij < fine_tau * diag),
                          np.mean(dji < fine_tau * diag)))
        is_coarse = min_gap < gap_tau and cfrac > contact_tau
        is_fine = ffrac > 0.01
        coarse += is_coarse
        fine += is_fine
        rows.append({"pair": f"p{i+1:02d}{j+1:02d}", "min_gap_over_diag": min_gap,
                     "contact_frac": cfrac, "fine_frac": ffrac,
                     "coarse_touch": bool(is_coarse), "fine_touch": bool(is_fine)})
    return {"diag": diag, "coarse_pairs": coarse, "fine_pairs": fine,
            "n_pairs": len(rows), "pairs": rows}


def interpenetration(meshes, mats, n_points, seed):
    """Mean fraction of part-i surface samples strictly inside any part j."""
    if trimesh is None or meshes is None:
        return None
    posed = []
    for m, mat in zip(meshes, mats):
        pm = m.copy()
        pm.apply_transform(mat)
        posed.append(pm)
    n_watertight = sum(1 for p in posed if p.is_watertight)
    fracs = []
    for i, pi in enumerate(posed):
        try:
            pts, _ = trimesh.sample.sample_surface(pi, n_points, seed=seed)
        except TypeError:
            pts, _ = trimesh.sample.sample_surface(pi, n_points)
        pts = np.asarray(pts)
        inside = np.zeros(len(pts), dtype=bool)
        for j, pj in enumerate(posed):
            if i == j or not pj.is_watertight:
                continue
            # cheap bbox reject
            in_bbox = np.all((pts > pj.bounds[0]) & (pts < pj.bounds[1]), axis=1)
            if not in_bbox.any():
                continue
            try:
                inside[in_bbox] |= pj.contains(pts[in_bbox])
            except Exception:
                pass
        fracs.append(float(inside.mean()))
    return {"per_part": fracs, "mean": float(np.mean(fracs)),
            "max": float(np.max(fracs)), "n_watertight": n_watertight}


def vesselness(clouds, n_bins=24, seed=0):
    """Fit a surface of revolution; return residual and profile fraction."""
    allp = np.concatenate(clouds, 0)
    if len(allp) > 20000:
        rng = np.random.default_rng(seed)
        allp = allp[rng.choice(len(allp), 20000, replace=False)]
    diag = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    centroid = allp.mean(0)

    def residual(params):
        th, ph, cx, cy, cz = params
        d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        c = centroid + np.array([cx, cy, cz]) * diag
        rel = allp - c
        h = rel @ d
        r = np.linalg.norm(rel - np.outer(h, d), axis=1)
        edges = np.linspace(h.min(), h.max() + 1e-9, n_bins + 1)
        idx = np.digitize(h, edges) - 1
        tot, cnt = 0.0, 0
        for b in range(n_bins):
            rb = r[idx == b]
            if len(rb) >= 20:
                tot += rb.std()
                cnt += 1
        return (tot / max(cnt, 1)) / diag

    # multi-init: PCA axes + z
    _, _, vt = np.linalg.svd(allp - centroid, full_matrices=False)
    inits = [vt[0], vt[1], vt[2], np.array([0.0, 0.0, 1.0])]
    best = None
    for d0 in inits:
        d0 = d0 / np.linalg.norm(d0)
        th0, ph0 = np.arccos(np.clip(d0[2], -1, 1)), np.arctan2(d0[1], d0[0])
        res = minimize(residual, [th0, ph0, 0, 0, 0], method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    th, ph, cx, cy, cz = best.x
    d = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    c = centroid + np.array([cx, cy, cz]) * diag
    rel = allp - c
    h = rel @ d
    r = np.linalg.norm(rel - np.outer(h, d), axis=1)
    edges = np.linspace(h.min(), h.max() + 1e-9, n_bins + 1)
    idx = np.digitize(h, edges) - 1
    med = np.full(n_bins, np.nan)
    for b in range(n_bins):
        rb = r[idx == b]
        if len(rb):
            med[b] = np.median(rb)
    ok = ~np.isnan(med[idx])
    within = np.abs(r[ok] - med[idx][ok]) < 0.05 * diag
    return {"sor_residual": float(best.fun), "profile_frac": float(within.mean()),
            "axis": d.tolist(), "median_radius_over_diag": float(np.nanmedian(med) / diag)}


def score_layout(clouds, meshes=None, mats=None, interp_points=300, seed=0):
    out = {"compactness": compactness(clouds)}
    pm = pair_metrics(clouds)
    out["diag"] = pm["diag"]
    out["coarse_pairs"] = pm["coarse_pairs"]
    out["fine_pairs"] = pm["fine_pairs"]
    out["pairs"] = pm["pairs"]
    out["vesselness"] = vesselness(clouds, seed=seed)
    if meshes is not None and mats is not None:
        out["interpenetration"] = interpenetration(meshes, mats, interp_points, seed)
    else:
        out["interpenetration"] = None
    return out


# ---------------------------------------------------------------- baselines


def random_layout(scan_clouds, rng):
    """Pile the (recentered) parts compactly at random poses."""
    from scipy.spatial.transform import Rotation as SR
    diags = [part_diag(c) for c in scan_clouds]
    radius = 0.8 * float(np.mean(diags))
    clouds = []
    for c in scan_clouds:
        cc = c - c.mean(0)
        rot = SR.random(random_state=rng.integers(2**31)).as_matrix()
        offset = rng.normal(size=3)
        offset = offset / np.linalg.norm(offset) * radius * rng.uniform(0, 1) ** (1 / 3)
        clouds.append(cc @ rot.T + offset)
    return clouds


def permuted_layout(scan_clouds, mats, rng):
    """Apply slot sigma(i)'s transform to part i (identities deranged)."""
    P = len(scan_clouds)
    while True:
        perm = rng.permutation(P)
        if not np.any(perm == np.arange(P)):
            break
    clouds, pmats = [], []
    cents = [c.mean(0) for c in scan_clouds]
    for i in range(P):
        j = int(perm[i])
        moved = scan_clouds[i] - cents[i] + cents[j]
        clouds.append(apply(mats[j], moved))
        pmats.append(mats[j] @ Tmat(cents[j] - cents[i]))
    return clouds, pmats, perm.tolist()


# ---------------------------------------------------------------- reporting


def export_glb(meshes, mats, out_path: Path):
    scene = trimesh.Scene()
    for k, (m, mat) in enumerate(zip(meshes, mats)):
        pm = m.copy()
        pm.apply_transform(mat)
        scene.add_geometry(pm, node_name=f"part{k+1:02d}")
    scene.export(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pfpp-dir", type=Path, required=True)
    ap.add_argument("--pc-npz", type=Path, required=True)
    ap.add_argument("--mesh-dir", type=Path, default=None)
    ap.add_argument("--garf-glb", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="t0")
    ap.add_argument("--n-null", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interp-points", type=int, default=300)
    ap.add_argument("--no-glb-export", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scan_clouds, mats = load_pfpp_layout(args.pfpp_dir, args.pc_npz)
    P = len(scan_clouds)
    meshes = names = None
    if args.mesh_dir is not None and trimesh is not None:
        meshes, names = match_meshes_to_parts(scan_clouds, args.mesh_dir)

    results = {}

    # --- PF++ arm
    pf_clouds = [apply(mats[i], scan_clouds[i]) for i in range(P)]
    results["pfpp"] = score_layout(pf_clouds, meshes, mats,
                                   args.interp_points, args.seed)
    if meshes is not None and not args.no_glb_export:
        export_glb(meshes, mats, args.out / f"{args.tag}_pfpp_posed.glb")

    # --- GARF arm
    if args.garf_glb is not None and trimesh is not None:
        g_clouds, g_meshes, g_names = load_garf_glb(args.garf_glb, 2000, args.seed)
        ident = [np.eye(4)] * len(g_meshes)
        results["garf"] = score_layout(g_clouds, g_meshes, ident,
                                       args.interp_points, args.seed)

    # --- nulls
    for null_name in ("random", "permuted"):
        runs = []
        for k in range(args.n_null):
            if null_name == "random":
                clouds = random_layout(scan_clouds, rng)
                s = score_layout(clouds, None, None, seed=args.seed + k)
            else:
                clouds, pmats, perm = permuted_layout(scan_clouds, mats, rng)
                s = score_layout(clouds, meshes, pmats,
                                 args.interp_points, seed=args.seed + k)
                s["perm"] = perm
            runs.append(s)
        agg = {}
        for key in ("compactness", "coarse_pairs", "fine_pairs"):
            vals = [r[key] for r in runs]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg["sor_residual"] = {
            "mean": float(np.mean([r["vesselness"]["sor_residual"] for r in runs])),
            "std": float(np.std([r["vesselness"]["sor_residual"] for r in runs]))}
        agg["profile_frac"] = {
            "mean": float(np.mean([r["vesselness"]["profile_frac"] for r in runs])),
            "std": float(np.std([r["vesselness"]["profile_frac"] for r in runs]))}
        interp_runs = [r["interpenetration"]["mean"] for r in runs
                       if r.get("interpenetration")]
        agg["interpenetration"] = ({"mean": float(np.mean(interp_runs)),
                                    "std": float(np.std(interp_runs))}
                                   if interp_runs else None)
        results[null_name] = {"aggregate": agg, "runs": runs}

    with open(args.out / f"{args.tag}_panel.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- markdown summary
    md = [f"# T0 layout panel — {args.tag}\n",
          "| arm | compactness | coarse pairs | fine pairs | interp mean | SoR resid | profile frac |",
          "|---|---|---|---|---|---|---|"]

    def row(tag, s):
        it = s.get("interpenetration")
        v = s["vesselness"]
        md.append(f"| {tag} | {s['compactness']:.3f} | {s['coarse_pairs']}/{len(s['pairs'])} | "
                  f"{s['fine_pairs']} | {it['mean']:.4f}" if it else
                  f"| {tag} | {s['compactness']:.3f} | {s['coarse_pairs']}/{len(s['pairs'])} | "
                  f"{s['fine_pairs']} | n/a")
        md[-1] += f" | {v['sor_residual']:.4f} | {v['profile_frac']:.3f} |"

    for tag in ("pfpp", "garf"):
        if tag in results:
            row(tag, results[tag])
    for tag in ("random", "permuted"):
        a = results[tag]["aggregate"]
        it = a["interpenetration"]
        md.append(f"| {tag} (n={args.n_null}) | {a['compactness']['mean']:.3f}"
                  f"±{a['compactness']['std']:.3f} | "
                  f"{a['coarse_pairs']['mean']:.1f}±{a['coarse_pairs']['std']:.1f} | "
                  f"{a['fine_pairs']['mean']:.1f} | "
                  + (f"{it['mean']:.4f} | " if it else "n/a | ")
                  + f"{a['sor_residual']['mean']:.4f}±{a['sor_residual']['std']:.4f} | "
                  f"{a['profile_frac']['mean']:.3f} |")
    (args.out / f"{args.tag}_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {args.out}/{args.tag}_panel.json and {args.tag}_summary.md")


if __name__ == "__main__":
    main()
