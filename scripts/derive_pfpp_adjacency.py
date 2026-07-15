#!/usr/bin/env python3
"""Derive Juglet sherd adjacency from the PF++ plausible assembly.

Exp 6 produced 0/36 mated pairs, but that is uninterpretable without knowing
which of the 36 pairs are TRUE physical mates. Juglet has no assembly GT, so we
take the PF++ deploy assembly (judged visually plausible in
JUGLET_DEPLOY_INFERENCE_ANALYSIS.md) as the reference configuration and read off
which sherds touch.

Method
------
PF++ stores, for the Juglet deploy sample:
  - part_pcs_gt (P,1000,3) : per-part point clouds in the scatter frame
    (data/pc_data/juglet_deploy/val/00000.npz),
  - gt.npy (P,7)           : per-part scatter pose,
  - init_pose.npy (7,)     : global reference pose,
  - predict_*.npy (T,P,7)  : denoise trajectory; last step = final pose.

The renderer maps each part into the assembled frame with
``compute_final_transformation`` (myrenderer.py L274-294). We reproduce that 4x4
chain in numpy and apply it to part_pcs_gt (verified: pred==gt -> identity, and
the predicted assembly compacts to a tight vessel). Then, for every pair, we
measure the min point-to-point gap (normalised by the assembled diagonal) and
the contact fraction; a pair is a TRUE MATE when both cross thresholds.

Usage
-----
  python scripts/derive_pfpp_adjacency.py \
      --pfpp-dir  <.../inference/juglet_deploy/0> \
      --pc-npz    <.../pc_data/juglet_deploy/val/00000.npz> \
      --out       logs/diagnostics/juglet_adjacency
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Scalar-first (w,x,y,z) unit quaternion -> 3x3 rotation matrix (Blender convention)."""
    w, x, y, z = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def Tmat(t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m


def Rmat(q_wxyz: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix(q_wxyz)
    return m


def final_transformation(init_pose, gt, pred):
    """PF++ myrenderer.compute_final_transformation as a 4x4 matrix."""
    return (Rmat(init_pose[3:]) @ Tmat(init_pose[:3]) @ Tmat(pred[:3]) @ Rmat(pred[3:])
            @ Rmat(gt[3:]).T @ Tmat(-gt[:3]) @ Tmat(-init_pose[:3]) @ Rmat(init_pose[3:]).T)


def apply(mat, v):
    return (np.c_[v, np.ones(len(v))] @ mat.T)[:, :3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pfpp-dir", type=Path, required=True)
    ap.add_argument("--pc-npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gap-tau", type=float, default=0.03,
                    help="max min-gap/scale for a mating pair")
    ap.add_argument("--contact-tau", type=float, default=0.03,
                    help="min contact fraction for a mating pair")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    init_pose = np.load(args.pfpp_dir / "init_pose.npy").astype(np.float64)
    gt = np.load(args.pfpp_dir / "gt.npy").astype(np.float64)
    traj = np.load(sorted(glob.glob(str(args.pfpp_dir / "predict_*.npy")))[0]).astype(np.float64)
    pred_final = traj[-1]
    pcs = np.load(args.pc_npz, allow_pickle=True)["part_pcs_gt"].astype(np.float64)  # (P,N,3)
    P = gt.shape[0]
    print(f"PF++ traj {traj.shape}, part_pcs_gt {pcs.shape}, {P} parts")

    # identity sanity check
    ident = final_transformation(init_pose, gt[0], gt[0])
    assert np.abs(ident - np.eye(4)).max() < 1e-6, "transform chain failed identity check"

    posed = [apply(final_transformation(init_pose, gt[i], pred_final[i]), pcs[i])
             for i in range(P)]
    allp = np.concatenate(posed, axis=0)
    scale = float(np.linalg.norm(allp.max(0) - allp.min(0)))
    tau = args.gap_tau * scale
    print(f"assembled diag scale = {scale:.4f}, contact tau_abs = {tau:.4f}")

    trees = [cKDTree(p) for p in posed]
    rows = []
    adj = np.zeros((P, P), dtype=int)
    for i, j in combinations(range(P), 2):
        dij, _ = trees[j].query(posed[i])
        dji, _ = trees[i].query(posed[j])
        min_gap = float(min(dij.min(), dji.min()))
        cfrac = float(max(np.mean(dij < tau), np.mean(dji < tau)))
        is_mate = (min_gap / scale) < args.gap_tau and cfrac > args.contact_tau
        if is_mate:
            adj[i, j] = adj[j, i] = 1
        rows.append({
            "pair": f"p{i+1:02d}{j+1:02d}",
            "piece_i": i + 1, "piece_j": j + 1,
            "min_gap_over_scale": min_gap / scale,
            "contact_frac": cfrac,
            "true_mate": bool(is_mate),
        })

    rows.sort(key=lambda r: r["min_gap_over_scale"])
    mates = [r for r in rows if r["true_mate"]]

    with open(args.out / "adjacency.json", "w") as f:
        json.dump({
            "n_parts": P, "gap_tau": args.gap_tau, "contact_tau": args.contact_tau,
            "scale": scale, "adjacency_matrix": adj.tolist(),
            "true_mates": [r["pair"] for r in mates], "pairs": rows,
        }, f, indent=2)

    md = ["# Juglet adjacency from PF++ plausible assembly\n",
          f"Parts: {P}. Mating rule: min_gap/scale < {args.gap_tau} and "
          f"contact_frac > {args.contact_tau}.\n",
          f"**True mates: {len(mates)} / {len(rows)} pairs**\n",
          "| pair | min_gap/scale | contact_frac | TRUE MATE |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pair']} | {r['min_gap_over_scale']:.4f} | "
                  f"{r['contact_frac']:.3f} | {'YES' if r['true_mate'] else ''} |")
    md.append("\n## Adjacency (piece x piece)\n")
    md.append("| |" + "".join(f" p{k+1:02d} |" for k in range(P)))
    md.append("|" + "---|" * (P + 1))
    for i in range(P):
        md.append(f"| p{i+1:02d} |" + "".join(
            f" {'X' if adj[i, k] else '.'} |" for k in range(P)))
    (args.out / "adjacency.md").write_text("\n".join(md) + "\n")

    print(f"\nTRUE MATES ({len(mates)}): {[r['pair'] for r in mates]}")
    print(f"wrote {args.out}/adjacency.md and adjacency.json")


if __name__ == "__main__":
    main()
