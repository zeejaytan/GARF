"""
generate_viz.py — Build a self-contained GARF assembly viewer HTML.

Takes GARF test JSON results + the HDF5 dataset and writes a single HTML file
with Three.js inlined (no internet required).

Usage
-----
    # Visualise a specific sample by JSON index:
    python generate_viz.py \\
        --results_dir logs/experiment/version_0/json_results \\
        --hdf5 input/Fractura/bone_synthetic.hdf5 \\
        --sample_id 42 \\
        --output bone_viz.html

    # Auto-pick the best sample:
    python generate_viz.py \\
        --results_dir logs/experiment/version_0/json_results \\
        --hdf5 input/Fractura/bone_synthetic.hdf5 \\
        --pick best \\
        --output bone_viz.html

    # List all evaluated samples:
    python generate_viz.py \\
        --results_dir logs/experiment/version_0/json_results \\
        --list

    # Build from a single JSON file directly:
    python generate_viz.py \\
        --json_file logs/experiment/version_0/json_results/42.json \\
        --hdf5 input/Fractura/bone_synthetic.hdf5 \\
        --output bone_viz.html
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from assembly.data.transform import recenter_pc, rotate_pc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PROJDIR = os.path.dirname(os.path.abspath(__file__))
THREEJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
ORBIT_CDN = "https://unpkg.com/three@0.134.0/examples/js/controls/OrbitControls.js"

PTS_PER_FRAG = 320  # downsampled points per fragment for web rendering

COLORS = [
    "#FE8A18", "#C91A09", "#237841", "#0055BF", "#F2705E",
    "#FC97AC", "#4B9F4A", "#008F9B", "#F5CD2F", "#4354A3",
    "#B3D7D1", "#C7D23C", "#FF800D", "#E53935", "#1E88E5",
    "#43A047", "#FB8C00", "#8E24AA", "#00ACC1", "#E040FB",
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def fps_downsample(pts: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sampling for roughly uniform coverage."""
    if pts.shape[0] <= k:
        return pts
    selected = [0]
    dists = np.full(pts.shape[0], np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(pts - pts[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return pts[selected]


def quat_angle_deg(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> float:
    """Angular error in degrees between two scipy [x,y,z,w] quaternions."""
    r1 = Rot.from_quat(q1_xyzw)
    r2 = Rot.from_quat(q2_xyzw)
    deg = float((r1 * r2.inv()).magnitude() * 180.0 / np.pi)
    return min(deg, 360.0 - deg)


def apply_se3(pts: np.ndarray, trans: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Apply SE(3) transform to points. quat is scipy [x,y,z,w]."""
    r = Rot.from_quat(quat_xyzw)
    return r.apply(pts) + trans


def se3_inverse(trans: np.ndarray, quat_xyzw: np.ndarray):
    """Compute inverse of an SE(3) transform. Returns (trans_inv, quat_inv) in scipy format."""
    r = Rot.from_quat(quat_xyzw)
    r_inv = r.inv()
    t_inv = -r_inv.apply(trans)
    return t_inv, r_inv.as_quat()


def se3_compose(t1, q1_xyzw, t2, q2_xyzw):
    """Compose two SE(3) transforms: T1 @ T2. Returns (trans, quat) in scipy format."""
    r1 = Rot.from_quat(q1_xyzw)
    r2 = Rot.from_quat(q2_xyzw)
    r_out = r1 * r2
    t_out = r1.apply(t2) + t1
    return t_out, r_out.as_quat()


def make_tf(pos: np.ndarray, quat_xyzw: np.ndarray) -> List[float]:
    """Flatten to [tx,ty,tz,qw,qx,qy,qz] for Three.js consumption.
    Input quat is scipy [x,y,z,w], output is Three.js [w,x,y,z] (scalar-first)."""
    qw, qx, qy, qz = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    return [round(float(v), 6) for v in [pos[0], pos[1], pos[2], qw, qx, qy, qz]]


Q_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0])  # scipy [x,y,z,w] identity


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """GARF JSON stores quaternions scalar-first [w, x, y, z] (PyTorch3D)."""
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def parse_trans_rot_7d(row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parse [tx, ty, tz, qw, qx, qy, qz] from GARF json_results."""
    return row[:3].astype(np.float64), quat_wxyz_to_xyzw(row[3:])


def is_scan_layout(centroids: np.ndarray, threshold: float = 2.0) -> bool:
    """True when fragment mesh centroids are far apart (table/scan coordinates)."""
    if len(centroids) < 2:
        return False
    max_dist = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            max_dist = max(max_dist, float(np.linalg.norm(centroids[i] - centroids[j])))
    return max_dist > threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def list_results(results_dir: str) -> List[Tuple[int, str, float, float]]:
    """Return [(index, name, part_acc, rmse_r), ...] sorted by index."""
    samples = []
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue
        try:
            idx = int(fname.replace(".json", ""))
        except ValueError:
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        samples.append((
            idx,
            data.get("name", ""),
            data.get("part_acc", 0.0),
            data.get("rmse_r", 999.0),
        ))
    return sorted(samples, key=lambda x: x[0])


def load_result_json(json_path: str) -> dict:
    """Load a single GARF test result JSON."""
    with open(json_path) as f:
        return json.load(f)


def load_meshes_from_hdf5(
    hdf5_path: str,
    sample_name: str,
    pieces_csv: str,
    mesh_scale: float,
) -> List[trimesh.Trimesh]:
    """Load fragment meshes from HDF5 and apply scaling."""
    h5 = h5py.File(hdf5_path, "r")
    grp = h5[sample_name]
    piece_keys = list(grp["pieces"].keys())

    meshes = []
    for pk in piece_keys:
        verts = np.array(grp["pieces"][pk]["vertices"][:])
        faces = np.array(grp["pieces"][pk]["faces"][:])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        meshes.append(mesh)

    h5.close()

    # Apply the same mesh_scale normalisation as the dataset code
    max_extent = max(max(m.extents) for m in meshes) if meshes else 1.0
    for m in meshes:
        m.apply_scale(1.0 / max_extent)

    return meshes


def recompute_dataloader_aug(
    meshes: List[trimesh.Trimesh],
    pts_per_frag: int,
    seed: int = 42,
) -> np.ndarray:
    """Reproduce per-part recenter + random-rot aug from MeshInferenceDataset."""
    rng = np.random.default_rng(seed=seed)
    rows = []
    for mesh in meshes:
        try:
            pts, _ = trimesh.sample.sample_surface_even(
                mesh, max(pts_per_frag * 3, 200)
            )
        except Exception:
            pts, _ = trimesh.sample.sample_surface(mesh, max(pts_per_frag * 3, 200))
        pts = fps_downsample(pts.astype(np.float32), pts_per_frag)
        centered, trans = recenter_pc(pts)
        _, _, quat_wxyz = rotate_pc(centered, numpy_rng=rng)
        rows.append(np.concatenate([trans.astype(np.float64), quat_wxyz.astype(np.float64)]))
    return np.stack(rows)


def sample_points_from_meshes(
    meshes: List[trimesh.Trimesh], pts_per_frag: int
) -> List[np.ndarray]:
    """Sample and FPS-downsample surface points from each mesh fragment."""
    base_points = []
    for mesh in meshes:
        try:
            pts, _ = trimesh.sample.sample_surface_even(mesh, max(pts_per_frag * 3, 200))
        except Exception:
            pts, _ = trimesh.sample.sample_surface(mesh, max(pts_per_frag * 3, 200))
        pts = fps_downsample(pts, pts_per_frag)
        base_points.append(pts)
    return base_points


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_payload(
    result: dict,
    hdf5_path: Optional[str],
    pts_per_frag: int = PTS_PER_FRAG,
    scan_layout: Optional[bool] = None,
    proposed_only: bool = False,
    aug_seed: int = 42,
) -> dict:
    name = result["name"]
    num_parts = result["num_parts"]
    deploy_mode = bool(result.get("deploy_mode", False))
    part_acc = float(result.get("part_acc", 0.0))
    rmse_r = float(result.get("rmse_r", 0.0))
    rmse_t = float(result.get("rmse_t", 0.0))
    shape_cd = float(result.get("shape_cd", 0.0))
    mesh_scale = result.get("mesh_scale", 1.0)
    pieces_csv = result.get("pieces", "")

    # JSON stores [tx, ty, tz, qw, qx, qy, qz] — GARF / PyTorch3D scalar-first quaternion
    gt_trans_rots_raw = result.get("gt_trans_rots", None)
    if gt_trans_rots_raw is None:
        gt_trans_rots_raw = result.get("gt_transform", None)
    gt_trans_rots = (
        np.array(gt_trans_rots_raw) if gt_trans_rots_raw is not None else None
    )  # (P, 7) scatter aug; None in older deploy JSON
    pred_trajectory = np.array(result["pred_trans_rots"])      # (T, P, 7)

    T = pred_trajectory.shape[0]
    pred_final = pred_trajectory[-1]                           # (P, 7)

    print(f"\n── Sample: {name} ──────────────────────────────────────────")
    print(f"  {num_parts} fragments, {T} denoising steps")
    if deploy_mode or gt_trans_rots is None:
        print("  deploy_mode=True: no true assembly GT; metrics are placeholders")
    else:
        print(
            f"  part_acc={part_acc:.4f}, rmse_r={rmse_r:.2f}°, rmse_t={rmse_t:.4f}, cd={shape_cd:.4f}"
        )

    # ── Load or synthesize fragment geometry ─────────────────────────────
    if hdf5_path and os.path.isfile(hdf5_path):
        print(f"  Loading meshes from {hdf5_path}")
        meshes = load_meshes_from_hdf5(hdf5_path, name, pieces_csv, mesh_scale)
        if len(meshes) != num_parts:
            print(f"  WARNING: HDF5 has {len(meshes)} pieces, JSON has {num_parts}. Truncating.")
            meshes = meshes[:num_parts]
        base_points_raw = sample_points_from_meshes(meshes, pts_per_frag)
    else:
        print("  No HDF5 provided — generating placeholder sphere fragments")
        base_points_raw = []
        for i in range(num_parts):
            pts = np.random.randn(pts_per_frag, 3).astype(np.float32)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * 0.05
            base_points_raw.append(pts)

    # Per-fragment geometry for Three.js (local origin at mesh centroid).
    centroids = np.array([pts.mean(axis=0) for pts in base_points_raw])
    base_points = [pts - centroids[i] for i, pts in enumerate(base_points_raw)]

    all_pts = np.concatenate(base_points_raw)
    scene_center = all_pts.mean(axis=0)

    if deploy_mode and gt_trans_rots is None and hdf5_path and os.path.isfile(hdf5_path):
        gt_trans_rots = recompute_dataloader_aug(meshes, pts_per_frag, seed=aug_seed)
        print(
            f"  Recomputed dataloader scatter aug (seed={aug_seed}) for mesh-relative proposal"
        )

    if scan_layout is None:
        # Deploy: table ref + mesh-relative pred; not pure scan_layout pred-at-origin.
        scan_layout = True if (deploy_mode or gt_trans_rots is None) else is_scan_layout(centroids)
    if scan_layout:
        print("  scan_layout=True: archaeological/scan coordinates (not true assembly GT)")

    use_mesh_relative_pred = gt_trans_rots is not None

    if gt_trans_rots is not None:
        # Anchor = part pinned to GT during denoising (pred ≈ gt)
        pred_gt_diff = np.array(
            [
                np.linalg.norm(pred_final[i, :3] - gt_trans_rots[i, :3])
                + 0.01
                * quat_angle_deg(
                    parse_trans_rot_7d(pred_final[i])[1],
                    parse_trans_rot_7d(gt_trans_rots[i])[1],
                )
                for i in range(num_parts)
            ]
        )
        anchor_idx = int(np.argmin(pred_gt_diff))
    else:
        anchor_idx = 0

    gt_tf = []
    pred_tf = []
    for i in range(num_parts):
        if scan_layout and not use_mesh_relative_pred:
            # Legacy: raw pose-space translation as world origin (misleading on deploy).
            pos = centroids[i] - scene_center
            gt_tf.append(make_tf(pos, Q_IDENTITY_XYZW))
            pr_t, pr_q = parse_trans_rot_7d(pred_final[i])
            pred_tf.append(make_tf(pr_t, pr_q))
        elif scan_layout and use_mesh_relative_pred:
            pos = centroids[i] - scene_center
            gt_tf.append(make_tf(pos, Q_IDENTITY_XYZW))
            gt_t, gt_q = parse_trans_rot_7d(gt_trans_rots[i])
            pr_t, pr_q = parse_trans_rot_7d(pred_final[i])
            gt_t_inv, gt_q_inv = se3_inverse(gt_t, gt_q)
            rel_t, rel_q = se3_compose(pr_t, pr_q, gt_t_inv, gt_q_inv)
            pos = apply_se3(centroids[i:i + 1], rel_t, rel_q)[0] - scene_center
            pred_tf.append(make_tf(pos, rel_q))
        else:
            if gt_trans_rots is None:
                raise ValueError("assembled_layout viz requires gt_trans_rots; use --scan_layout for deploy results")
            # Fractura / Breaking Bad: meshes are pre-broken near assembled layout.
            pos = centroids[i] - scene_center
            gt_tf.append(make_tf(pos, Q_IDENTITY_XYZW))
            gt_t, gt_q = parse_trans_rot_7d(gt_trans_rots[i])
            pr_t, pr_q = parse_trans_rot_7d(pred_final[i])
            gt_t_inv, gt_q_inv = se3_inverse(gt_t, gt_q)
            rel_t, rel_q = se3_compose(pr_t, pr_q, gt_t_inv, gt_q_inv)
            pos = apply_se3(centroids[i:i+1], rel_t, rel_q)[0] - scene_center
            pred_tf.append(make_tf(pos, rel_q))

    # ── Scattered grid layout ────────────────────────────────────────────
    scene_scale = float(np.max(np.abs(all_pts - scene_center))) or 1.0
    grid_spacing = scene_scale * 2.2
    cols = int(np.ceil(np.sqrt(num_parts)))
    scattered_tf = []
    for i in range(num_parts):
        row, col = divmod(i, cols)
        offset = np.array([
            col * grid_spacing - (cols - 1) * grid_spacing / 2,
            0.0,
            row * grid_spacing - (cols - 1) * grid_spacing / 2,
        ])
        scattered_tf.append(make_tf(offset, Q_IDENTITY_XYZW))

    # ── Per-fragment errors ──────────────────────────────────────────────
    if gt_trans_rots is not None:
        rot_errors = [
            quat_angle_deg(
                parse_trans_rot_7d(pred_final[i])[1],
                parse_trans_rot_7d(gt_trans_rots[i])[1],
            )
            for i in range(num_parts)
        ]
        trans_errors = [
            float(np.linalg.norm(pred_final[i, :3] - gt_trans_rots[i, :3]))
            for i in range(num_parts)
        ]
    else:
        rot_errors = [0.0 for _ in range(num_parts)]
        trans_errors = [0.0 for _ in range(num_parts)]

    # ── Trajectory keyframes ─────────────────────────────────────────────
    stride = max(1, T // 16)
    traj_steps = list(range(0, T, stride))
    if T - 1 not in traj_steps:
        traj_steps.append(T - 1)

    traj_tf = []
    traj_mean_rot_err = []
    for t_idx in traj_steps:
        step_pred = pred_trajectory[t_idx]
        step_tf = []
        step_errors = []
        for i in range(num_parts):
            sp_t, sp_q = parse_trans_rot_7d(step_pred[i])
            if scan_layout and not use_mesh_relative_pred:
                step_tf.append(make_tf(sp_t, sp_q))
            elif scan_layout and use_mesh_relative_pred:
                gt_t, gt_q = parse_trans_rot_7d(gt_trans_rots[i])
                gt_t_inv, gt_q_inv = se3_inverse(gt_t, gt_q)
                rel_t, rel_q = se3_compose(sp_t, sp_q, gt_t_inv, gt_q_inv)
                pos = apply_se3(centroids[i:i + 1], rel_t, rel_q)[0] - scene_center
                step_tf.append(make_tf(pos, rel_q))
            else:
                if gt_trans_rots is None:
                    raise ValueError("assembled_layout trajectory requires gt_trans_rots")
                gt_t, gt_q = parse_trans_rot_7d(gt_trans_rots[i])
                gt_t_inv, gt_q_inv = se3_inverse(gt_t, gt_q)
                rel_t, rel_q = se3_compose(sp_t, sp_q, gt_t_inv, gt_q_inv)
                pos = apply_se3(centroids[i:i+1], rel_t, rel_q)[0] - scene_center
                step_tf.append(make_tf(pos, rel_q))
            if gt_trans_rots is not None:
                gt_q = parse_trans_rot_7d(gt_trans_rots[i])[1]
                step_errors.append(quat_angle_deg(sp_q, gt_q))
        traj_tf.append(step_tf)
        traj_mean_rot_err.append(float(np.mean(step_errors)) if step_errors else 0.0)

    # ── Summary ──────────────────────────────────────────────────────────
    if gt_trans_rots is not None:
        print(
            f"  rot_error  min/mean/max: "
            f"{min(rot_errors):.1f}° / {np.mean(rot_errors):.1f}° / {max(rot_errors):.1f}°"
        )
        print(
            f"  trans_err  min/mean/max: "
            f"{min(trans_errors):.4f} / {np.mean(trans_errors):.4f} / {max(trans_errors):.4f}"
        )
    if len(traj_mean_rot_err) > 1:
        print(f"  traj mean rot: {traj_mean_rot_err[0]:.1f}° → {traj_mean_rot_err[-1]:.1f}°")

    # Determine label
    path_parts = name.replace("\\", "/").split("/")
    category = path_parts[0] if path_parts else "unknown"
    label = f"{category} · {num_parts} fragments"
    if scan_layout:
        label += " · scan layout"
    if proposed_only or deploy_mode:
        if use_mesh_relative_pred:
            label = f"{category} · {num_parts} fragments · model proposal (mesh-relative)"
        else:
            label = f"{category} · {num_parts} fragments · model proposal (pose frame)"

    return {
        "n_parts": num_parts,
        "anchor_idx": anchor_idx,
        "scan_layout": scan_layout,
        "deploy_mode": deploy_mode,
        "proposed_only": proposed_only or deploy_mode,
        "name": name,
        "label": label,
        "part_acc": float(part_acc),
        "rmse_r": float(rmse_r),
        "rmse_t": float(rmse_t),
        "shape_cd": float(shape_cd),
        "colors": COLORS[:num_parts],
        "rot_errors": rot_errors,
        "trans_errors": trans_errors,
        "traj_mean_rot_err": traj_mean_rot_err,
        "traj_steps": traj_steps,
        "T": int(T),
        "base_points": [p.tolist() for p in base_points],
        "scattered_tf": scattered_tf,
        "gt_tf": gt_tf,
        "pred_tf": pred_tf,
        "traj_tf": traj_tf,
    }


# ---------------------------------------------------------------------------
# Three.js asset loading (local first, CDN fallback)
# ---------------------------------------------------------------------------

def load_asset(local_dir: str, filename: str, cdn_url: str) -> str:
    local_path = os.path.join(local_dir, filename)
    if os.path.isfile(local_path):
        with open(local_path) as f:
            return f.read()
    print(f"  {filename} not found locally, downloading from CDN...")
    try:
        import urllib.request
        with urllib.request.urlopen(cdn_url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        os.makedirs(local_dir, exist_ok=True)
        with open(local_path, "w") as f:
            f.write(content)
        print(f"  Saved to {local_path}")
        return content
    except Exception as e:
        sys.exit(f"ERROR: Could not load {filename}: {e}")


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def generate_html(payload: dict, threejs_dir: str) -> str:
    three_js = load_asset(threejs_dir, "three.min.js", THREEJS_CDN)
    orbit_js = load_asset(threejs_dir, "OrbitControls.js", ORBIT_CDN)

    data_json = json.dumps(payload, separators=(",", ":"))

    label = payload["label"]
    part_acc = payload["part_acc"] * 100
    rmse_r = payload["rmse_r"]
    rmse_t = payload["rmse_t"]
    shape_cd = payload["shape_cd"]
    n_parts = payload["n_parts"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>GARF — {label}</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0a0e1a;color:#e0e0e0;font-family:system-ui,-apple-system,sans-serif;overflow:hidden}}
    #c{{position:absolute;inset:0}}

    #ui{{
      position:absolute;top:16px;right:16px;width:290px;
      max-height:calc(100vh - 32px);overflow-y:auto;
      padding:16px;border-radius:10px;
      background:rgba(10,14,26,0.95);border:1px solid #1e293b;
      box-shadow:0 20px 50px rgba(0,0,0,.7);
      display:flex;flex-direction:column;gap:12px;z-index:10
    }}
    #ui::-webkit-scrollbar{{width:4px}}
    #ui::-webkit-scrollbar-thumb{{background:#334155;border-radius:2px}}

    .brand{{display:flex;align-items:center;gap:8px;border-bottom:1px solid #1e293b;padding-bottom:8px}}
    .brand h1{{font-size:14px;letter-spacing:3px;text-transform:uppercase;color:#7dd3fc;font-weight:700}}
    .brand .sub{{font-size:10px;color:#64748b;letter-spacing:1px}}

    .metric-grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
    .metric-card{{
      padding:8px;border-radius:6px;background:#0f172a;border:1px solid #1e293b;
      text-align:center
    }}
    .metric-card .val{{font-size:18px;font-weight:700;color:#e2e8f0}}
    .metric-card .lbl{{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-top:2px}}
    .metric-card.good .val{{color:#4ade80}}
    .metric-card.warn .val{{color:#fbbf24}}
    .metric-card.bad  .val{{color:#f87171}}

    .section-label{{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-top:4px}}

    #mode-buttons{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}}
    .mode-btn{{
      padding:7px 4px;border-radius:5px;border:1px solid #1e293b;
      background:#020617;color:#94a3b8;font-size:10px;text-transform:uppercase;
      letter-spacing:1px;cursor:pointer;transition:all .15s ease-out;text-align:center
    }}
    .mode-btn:hover{{border-color:#0ea5e9;color:#bae6fd}}
    .mode-btn.active{{border-color:#0ea5e9;background:#0c2744;color:#e0f2fe}}

    #frag-table{{font-size:10px;color:#94a3b8;border-collapse:collapse;width:100%}}
    #frag-table td{{padding:3px 4px}}
    #frag-table tr:hover{{background:#0f172a}}
    .ok{{color:#4ade80}}.bad{{color:#f87171}}.anchor{{color:#fbbf24}}

    #traj-section{{display:flex;flex-direction:column;gap:6px}}
    #traj-bar-bg{{height:5px;background:#1e293b;border-radius:3px;overflow:hidden}}
    #traj-bar{{height:100%;width:0%;background:linear-gradient(90deg,#0ea5e9,#7dd3fc);border-radius:3px;transition:width .08s}}
    #traj-controls{{display:flex;gap:6px;align-items:center}}
    #traj-play{{
      padding:5px 12px;border-radius:5px;border:1px solid #1e293b;background:#020617;
      color:#94a3b8;font-size:10px;cursor:pointer;flex-shrink:0
    }}
    #traj-play:hover{{border-color:#0ea5e9;color:#bae6fd}}
    #traj-label{{font-size:10px;color:#64748b;flex:1}}

    #mode-label{{
      position:absolute;top:16px;left:16px;padding:5px 12px;border-radius:999px;
      border:1px solid #1e293b;background:rgba(10,14,26,.95);font-size:10px;
      letter-spacing:2px;text-transform:uppercase;color:#7dd3fc;pointer-events:none;z-index:10
    }}
    #hint{{font-size:10px;color:#475569;text-align:center}}
  </style>
</head>
<body>
  <div id="c"></div>
  <div id="mode-label">Scattered</div>
  <div id="ui">
    <div class="brand">
      <div>
        <h1>GARF</h1>
        <div class="sub">3D Fragment Reassembly</div>
      </div>
    </div>

    <div class="metric-card" style="text-align:left;padding:10px">
      <div style="font-size:11px;color:#94a3b8">Sample</div>
      <div style="font-size:13px;color:#e2e8f0;font-weight:600;margin-top:2px">{label}</div>
      <div style="font-size:10px;color:#64748b;margin-top:1px;word-break:break-all" id="sample-name"></div>
    </div>

    <div class="metric-grid">
      <div class="metric-card" id="mc-acc"><div class="val">{part_acc:.1f}%</div><div class="lbl">Part Accuracy</div></div>
      <div class="metric-card" id="mc-rot"><div class="val">{rmse_r:.1f}&deg;</div><div class="lbl">RMSE Rotation</div></div>
      <div class="metric-card" id="mc-trans"><div class="val">{rmse_t:.3f}</div><div class="lbl">RMSE Translation</div></div>
      <div class="metric-card" id="mc-cd"><div class="val">{shape_cd:.4f}</div><div class="lbl">Shape CD</div></div>
    </div>

    <div class="section-label">View Mode</div>
    <div id="mode-buttons">
      <button class="mode-btn active" data-mode="scattered">Scattered</button>
      <button class="mode-btn"        data-mode="pred">Predicted</button>
      <button class="mode-btn"        data-mode="gt" id="gt-mode-btn">Ground Truth</button>
    </div>

    <div class="section-label">Fragments ({n_parts})</div>
    <div style="max-height:180px;overflow-y:auto">
      <table id="frag-table"><tbody id="frag-rows"></tbody></table>
    </div>

    <div class="section-label">Denoising Trajectory</div>
    <div id="traj-section">
      <div id="traj-bar-bg"><div id="traj-bar"></div></div>
      <div id="traj-controls">
        <button id="traj-play">&#9654; Play</button>
        <span id="traj-label">Step 0 / 0</span>
      </div>
    </div>

    <div id="hint">Drag to orbit &middot; scroll to zoom &middot; right-drag to pan</div>
  </div>

  <script>{three_js}</script>
  <script>{orbit_js}</script>
  <script>
    const DATA = {data_json};

    // Colour metric cards
    (function(){{
      const a=DATA.part_acc*100;
      document.getElementById('mc-acc').classList.add(a>80?'good':a>40?'warn':'bad');
      const r=DATA.rmse_r;
      document.getElementById('mc-rot').classList.add(r<15?'good':r<45?'warn':'bad');
      const t=DATA.rmse_t;
      document.getElementById('mc-trans').classList.add(t<0.5?'good':t<2?'warn':'bad');
      document.getElementById('sample-name').textContent=DATA.name;
    }})();

    const MODE_SCAT='scattered', MODE_PRED='pred', MODE_GT='gt', MODE_TRAJ='traj';
    let scene, camera, renderer, controls;
    let mode = DATA.proposed_only ? MODE_PRED : MODE_SCAT;
    let trajFrame = 0, trajPlaying = false, trajTimer = null;
    const fragmentPoints = [];

    // ── Build fragment status table ──────────────────────────────────────
    (function(){{
      const tbody = document.getElementById('frag-rows');
      for (let i = 0; i < DATA.n_parts; i++) {{
        const err  = DATA.rot_errors[i];
        const terr = DATA.trans_errors[i];
        const isAnc = i === DATA.anchor_idx;
        const isOk  = err < 30;
        const cls   = isAnc ? 'anchor' : (isOk ? 'ok' : 'bad');
        const icon  = isAnc ? '&#9875;' : (isOk ? '&#10003;' : '&#10007;');
        const tr = document.createElement('tr');
        tr.innerHTML =
          '<td><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
          + 'vertical-align:middle;background:' + DATA.colors[i] + ';margin-right:4px"></span>F' + i + '</td>'
          + '<td class="'+cls+'">'+icon+'</td>'
          + '<td>' + err.toFixed(1) + '&deg;</td>'
          + '<td>' + terr.toFixed(3) + '</td>';
        tbody.appendChild(tr);
      }}
    }})();

    // ── Transform helpers ────────────────────────────────────────────────
    function tfForMode() {{
      if (mode === MODE_SCAT) return DATA.scattered_tf;
      if (mode === MODE_PRED) return DATA.pred_tf;
      if (mode === MODE_GT)   return DATA.gt_tf;
      return DATA.traj_tf[trajFrame];
    }}

    function applyTransforms(tfs) {{
      for (let i = 0; i < fragmentPoints.length; i++) {{
        const tf = tfs[i];
        const [tx,ty,tz,qw,qx,qy,qz] = tf;
        fragmentPoints[i].position.set(tx, ty, tz);
        fragmentPoints[i].setRotationFromQuaternion(new THREE.Quaternion(qx,qy,qz,qw));
      }}
    }}

    function updateModeLabel() {{
      const labels = {{
        scattered:'Scattered (dataloader aug)',
        pred: DATA.proposed_only ? 'Model proposal (pred vs scatter aug)' : 'Predicted (model pose frame)',
        gt: DATA.scan_layout ? 'Scan layout (ref)' : 'Ground Truth',
        traj:'Trajectory'
      }};
      if (DATA.scan_layout) {{
        document.getElementById('gt-mode-btn').title =
          'Table/scanner positions — not a true assembly';
      }}
      document.getElementById('mode-label').textContent = labels[mode] || mode;
    }}

    // ── Scene setup ──────────────────────────────────────────────────────
    function buildScene() {{
      const container = document.getElementById('c');
      const w = window.innerWidth, h = window.innerHeight;

      renderer = new THREE.WebGLRenderer({{antialias:true,alpha:false}});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(w, h);
      renderer.setClearColor(0x0a0e1a, 1);
      container.appendChild(renderer.domElement);

      scene  = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(45, w/h, 0.001, 50);
      camera.position.set(0.6, 0.5, 1.2);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;

      // Lighting
      scene.add(new THREE.AmbientLight(0x404050, 0.4));
      const hemi = new THREE.HemisphereLight(0xddeeff, 0x111122, 0.8);
      scene.add(hemi);
      const dir = new THREE.DirectionalLight(0xffffff, 0.5);
      dir.position.set(2, 3, 1);
      scene.add(dir);

      // Grid
      const grid = new THREE.GridHelper(4, 20, 0x1e293b, 0x0f172a);
      grid.position.y = -0.5;
      scene.add(grid);
      const axes = new THREE.AxesHelper(0.2);
      axes.position.y = -0.5;
      scene.add(axes);

      // Build fragment point clouds
      for (let i = 0; i < DATA.n_parts; i++) {{
        const pts  = DATA.base_points[i];
        const geom = new THREE.BufferGeometry();
        const pos  = new Float32Array(pts.length * 3);
        for (let j = 0; j < pts.length; j++) {{
          pos[3*j]=pts[j][0]; pos[3*j+1]=pts[j][1]; pos[3*j+2]=pts[j][2];
        }}
        geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat   = new THREE.PointsMaterial({{
          size: 0.008,
          color: new THREE.Color(DATA.colors[i]),
          sizeAttenuation: true,
        }});
        const cloud = new THREE.Points(geom, mat);
        scene.add(cloud);
        fragmentPoints.push(cloud);
      }}

      applyTransforms(tfForMode());
      updateModeLabel();

      window.addEventListener('resize', () => {{
        renderer.setSize(window.innerWidth, window.innerHeight);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
      }});

      animate();
    }}

    function animate() {{
      requestAnimationFrame(animate);
      if (controls) controls.update();
      if (renderer && scene && camera) renderer.render(scene, camera);
    }}

    // ── Mode switching ───────────────────────────────────────────────────
    function setMode(next) {{
      stopTraj();
      mode = next;
      document.querySelectorAll('.mode-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === mode));
      applyTransforms(tfForMode());
      updateModeLabel();
    }}

    document.querySelectorAll('.mode-btn').forEach(b =>
      b.addEventListener('click', () => setMode(b.dataset.mode)));

    // ── Trajectory playback ──────────────────────────────────────────────
    function setTrajFrame(f) {{
      trajFrame = Math.max(0, Math.min(f, DATA.traj_tf.length - 1));
      const step = DATA.traj_steps[trajFrame];
      const pct  = step / Math.max(DATA.T - 1, 1) * 100;
      const err  = DATA.traj_mean_rot_err[trajFrame];
      document.getElementById('traj-bar').style.width = pct.toFixed(1) + '%';
      document.getElementById('traj-label').textContent =
        'Step ' + step + ' / ' + (DATA.T-1) + ' · ' + err.toFixed(1) + '°';
      if (mode === MODE_TRAJ) applyTransforms(DATA.traj_tf[trajFrame]);
    }}

    function startTraj() {{
      trajPlaying = true;
      document.getElementById('traj-play').innerHTML = '&#9632; Stop';
      mode = MODE_TRAJ;
      updateModeLabel();
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      setTrajFrame(0);
      trajTimer = setInterval(() => {{
        if (trajFrame >= DATA.traj_tf.length - 1) {{ stopTraj(); return; }}
        setTrajFrame(trajFrame + 1);
      }}, 120);
    }}

    function stopTraj() {{
      if (trajTimer) {{ clearInterval(trajTimer); trajTimer = null; }}
      trajPlaying = false;
      document.getElementById('traj-play').innerHTML = '&#9654; Play';
    }}

    document.getElementById('traj-play').addEventListener('click', () => {{
      if (trajPlaying) stopTraj(); else startTraj();
    }});

    setTrajFrame(0);
    buildScene();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Multi-sample index page
# ---------------------------------------------------------------------------

def resolve_scan_layout_flag(
    scan_layout: bool,
    assembled_layout: bool,
) -> Optional[bool]:
    if scan_layout and assembled_layout:
        raise ValueError("Use only one of --scan_layout or --assembled_layout")
    if scan_layout:
        return True
    if assembled_layout:
        return False
    return None


def generate_index_html(
    results_dir: str,
    hdf5_path: Optional[str],
    threejs_dir: str,
    output_dir: str,
    pts_per_frag: int,
    max_samples: int = 50,
    scan_layout: Optional[bool] = None,
    proposed_only: bool = False,
    aug_seed: int = 42,
) -> str:
    """Generate individual viz pages + an index HTML linking them all."""
    samples = list_results(results_dir)
    if not samples:
        sys.exit(f"ERROR: No JSON results found in {results_dir}")

    # Sort by part_acc descending
    samples.sort(key=lambda x: x[2], reverse=True)
    samples = samples[:max_samples]

    os.makedirs(output_dir, exist_ok=True)

    entries = []
    for idx, name, acc, rmse in samples:
        json_path = os.path.join(results_dir, f"{idx}.json")
        result = load_result_json(json_path)
        try:
            payload = build_payload(
                result,
                hdf5_path,
                pts_per_frag,
                scan_layout=scan_layout,
                proposed_only=proposed_only,
                aug_seed=aug_seed,
            )
        except Exception as e:
            print(f"  SKIP {idx} ({name}): {e}")
            continue

        viz_file = f"sample_{idx}.html"
        viz_path = os.path.join(output_dir, viz_file)
        html = generate_html(payload, threejs_dir)
        with open(viz_path, "w") as f:
            f.write(html)

        entries.append({
            "idx": idx,
            "name": name,
            "file": viz_file,
            "part_acc": acc,
            "rmse_r": result.get("rmse_r", 0),
            "rmse_t": result.get("rmse_t", 0),
            "shape_cd": result.get("shape_cd", 0),
            "num_parts": result.get("num_parts", 0),
        })

    # Build index page
    mean_acc = np.mean([e["part_acc"] for e in entries]) if entries else 0
    mean_rot = np.mean([e["rmse_r"] for e in entries]) if entries else 0

    rows_html = ""
    for e in entries:
        acc_cls = "good" if e["part_acc"] > 0.8 else ("warn" if e["part_acc"] > 0.4 else "bad")
        rows_html += f"""
        <tr onclick="window.open('{e['file']}','_blank')" style="cursor:pointer">
          <td>{e['idx']}</td>
          <td style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{e['name']}">{e['name']}</td>
          <td>{e['num_parts']}</td>
          <td class="{acc_cls}">{e['part_acc']*100:.1f}%</td>
          <td>{e['rmse_r']:.1f}&deg;</td>
          <td>{e['rmse_t']:.3f}</td>
          <td>{e['shape_cd']:.4f}</td>
        </tr>"""

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>GARF — Evaluation Results</title>
  <style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{background:#0a0e1a;color:#e0e0e0;font-family:system-ui,sans-serif;padding:32px}}
    h1{{font-size:22px;letter-spacing:3px;text-transform:uppercase;color:#7dd3fc;margin-bottom:4px}}
    .sub{{color:#64748b;font-size:12px;margin-bottom:24px}}
    .summary{{display:flex;gap:16px;margin-bottom:24px}}
    .summary-card{{padding:16px 24px;border-radius:8px;background:#0f172a;border:1px solid #1e293b;text-align:center}}
    .summary-card .val{{font-size:24px;font-weight:700;color:#e2e8f0}}
    .summary-card .lbl{{font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-top:4px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{text-align:left;padding:10px 12px;border-bottom:2px solid #1e293b;color:#64748b;font-size:10px;
        text-transform:uppercase;letter-spacing:1px}}
    td{{padding:10px 12px;border-bottom:1px solid #1e293b}}
    tr:hover{{background:#0f172a}}
    .good{{color:#4ade80}}.warn{{color:#fbbf24}}.bad{{color:#f87171}}
    a{{color:#7dd3fc;text-decoration:none}}
    a:hover{{text-decoration:underline}}
  </style>
</head>
<body>
  <h1>GARF Evaluation Results</h1>
  <div class="sub">{len(entries)} samples · click any row to open 3D viewer</div>

  <div class="summary">
    <div class="summary-card"><div class="val">{mean_acc*100:.1f}%</div><div class="lbl">Mean Part Accuracy</div></div>
    <div class="summary-card"><div class="val">{mean_rot:.1f}&deg;</div><div class="lbl">Mean RMSE Rotation</div></div>
    <div class="summary-card"><div class="val">{len(entries)}</div><div class="lbl">Samples</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th><th>Name</th><th>Parts</th>
        <th>Part Acc</th><th>RMSE(R)</th><th>RMSE(T)</th><th>Shape CD</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</body>
</html>"""

    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(index_html)

    return index_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate self-contained GARF assembly viewer HTML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir", "-r",
        help="Path to json_results directory from GARF evaluation.",
    )
    parser.add_argument(
        "--json_file", "-j",
        help="Path to a single result JSON file (alternative to --results_dir).",
    )
    parser.add_argument(
        "--hdf5", "-H",
        help="Path to the HDF5 dataset file for loading fragment meshes.",
    )
    parser.add_argument(
        "--sample_id", "-s", type=int, default=None,
        help="Which sample to visualise (integer index from json_results).",
    )
    parser.add_argument(
        "--pick", choices=["best", "worst", "first", "last"], default=None,
        help="Auto-pick a sample by part accuracy.",
    )
    parser.add_argument(
        "--output", "-o", default=os.path.join(PROJDIR, "assembly_viz.html"),
        help="Output HTML file path (single sample mode).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate individual pages for ALL samples + an index page.",
    )
    parser.add_argument(
        "--output_dir", default=os.path.join(PROJDIR, "viz_output"),
        help="Output directory for --all mode (default: viz_output/).",
    )
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Max samples to generate in --all mode (default: 50).",
    )
    parser.add_argument(
        "--pts_per_frag", type=int, default=PTS_PER_FRAG,
        help=f"Points per fragment to embed (default {PTS_PER_FRAG}).",
    )
    parser.add_argument(
        "--threejs_dir", default=os.path.join(PROJDIR, "renderer", "threejs"),
        help="Directory containing three.min.js and OrbitControls.js.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available samples and their metrics, then exit.",
    )
    parser.add_argument(
        "--scan_layout",
        action="store_true",
        help="Force archaeological/scan viz (pred in pose frame, GT = scan layout).",
    )
    parser.add_argument(
        "--assembled_layout",
        action="store_true",
        help="Force Fractura-style viz (T_pred @ T_gt^{-1} in mesh space).",
    )
    parser.add_argument(
        "--proposed_only",
        action="store_true",
        help="Deploy QA: open on Model proposal view (anchor-free poses, no GT snap).",
    )
    parser.add_argument(
        "--aug_seed",
        type=int,
        default=42,
        help="RNG seed to recompute dataloader scatter aug for deploy viz (match eval seed).",
    )
    args = parser.parse_args()
    scan_layout_flag = resolve_scan_layout_flag(
        args.scan_layout, args.assembled_layout
    )
    if args.proposed_only and not scan_layout_flag:
        scan_layout_flag = True

    # ── List mode ────────────────────────────────────────────────────────
    if args.list:
        if not args.results_dir:
            parser.error("--results_dir is required for --list")
        samples = list_results(args.results_dir)
        if not samples:
            sys.exit(f"No JSON results found in {args.results_dir}")
        print(f"{'idx':>6}  {'part_acc':>10}  {'rmse_r':>8}  name")
        print("-" * 60)
        for idx, name, acc, rmse in samples:
            print(f"{idx:>6}  {acc*100:>9.2f}%  {rmse:>7.1f}°  {name}")
        print(f"\nTotal: {len(samples)} samples")
        return

    # ── All-samples mode ─────────────────────────────────────────────────
    if args.all:
        if not args.results_dir:
            parser.error("--results_dir is required for --all")
        index_path = generate_index_html(
            results_dir=args.results_dir,
            hdf5_path=args.hdf5,
            threejs_dir=args.threejs_dir,
            output_dir=args.output_dir,
            pts_per_frag=args.pts_per_frag,
            max_samples=args.max_samples,
            scan_layout=scan_layout_flag,
            proposed_only=args.proposed_only,
            aug_seed=args.aug_seed,
        )
        print(f"\nIndex page: {index_path}")
        print("Open in any modern browser (no server needed).\n")
        return

    # ── Single-sample mode ───────────────────────────────────────────────
    if args.json_file:
        result = load_result_json(args.json_file)
    elif args.results_dir:
        samples = list_results(args.results_dir)
        if not samples:
            sys.exit(f"No JSON results found in {args.results_dir}")

        if args.sample_id is not None:
            json_path = os.path.join(args.results_dir, f"{args.sample_id}.json")
            if not os.path.isfile(json_path):
                sys.exit(f"Sample {args.sample_id} not found in {args.results_dir}")
            result = load_result_json(json_path)
        elif args.pick:
            if args.pick == "best":
                chosen = max(samples, key=lambda x: x[2])
            elif args.pick == "worst":
                chosen = min(samples, key=lambda x: x[2])
            elif args.pick == "first":
                chosen = samples[0]
            else:
                chosen = samples[-1]
            print(f"Auto-picked sample {chosen[0]} ({chosen[1]}, acc={chosen[2]*100:.1f}%)")
            result = load_result_json(os.path.join(args.results_dir, f"{chosen[0]}.json"))
        else:
            chosen = max(samples, key=lambda x: x[2])
            print(f"No --sample_id specified; picking best: {chosen[0]} (acc={chosen[2]*100:.1f}%)")
            result = load_result_json(os.path.join(args.results_dir, f"{chosen[0]}.json"))
    else:
        parser.error("Either --results_dir or --json_file is required")

    payload = build_payload(
        result,
        args.hdf5,
        args.pts_per_frag,
        scan_layout=scan_layout_flag,
        proposed_only=args.proposed_only,
        aug_seed=args.aug_seed,
    )

    print(f"\n── Generating HTML ─────────────────────────────────────────────")
    html = generate_html(payload, args.threejs_dir)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Written: {out_path}  ({size_kb:.0f} KB)")
    print(f"\nOpen in any modern browser (no server needed).\n")


if __name__ == "__main__":
    main()
