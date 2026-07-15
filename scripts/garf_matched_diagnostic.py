#!/usr/bin/env python3
"""Run matched-piece GARF diagnostics across Juglet/BB/Fractura.

This script:
1) Selects samples with the requested piece count from each dataset split.
2) Builds GARF-compatible one-sample subset HDF5 files.
3) Runs GARF eval with identical inference settings across cases.
4) Computes process indicators from JSON trajectories + HDF5 geometry.
5) Writes CSV + Markdown summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class CaseSpec:
    label: str
    source_hdf5: Path
    split_key: str
    fixed_sample: str | None = None


def _decode(x: Any) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)


def _numeric_piece_keys(group: h5py.Group) -> list[str]:
    return sorted([k for k in group.keys() if k.isdigit()], key=lambda x: int(x))


def count_parts(h5: h5py.File, sample_name: str) -> int:
    g = h5[sample_name]
    if "pieces" in g:
        return len(g["pieces"].keys())
    return len(_numeric_piece_keys(g))


def get_val_samples(h5: h5py.File, split_key: str) -> list[str]:
    vals = h5["data_split"][split_key]["val"][:]
    return [_decode(v) for v in vals]


def pick_sample_with_parts(h5: h5py.File, split_key: str, num_parts: int) -> str:
    for name in get_val_samples(h5, split_key):
        try:
            if count_parts(h5, name) == num_parts:
                return name
        except Exception:
            continue
    raise RuntimeError(
        f"No sample with {num_parts} parts in split data_split/{split_key}/val"
    )


def _copy_piece(src_piece: h5py.Group, dst_piece: h5py.Group) -> None:
    verts = np.asarray(src_piece["vertices"][:], dtype=np.float64)
    faces = np.asarray(src_piece["faces"][:], dtype=np.int64)
    dst_piece.create_dataset("vertices", data=verts)
    dst_piece.create_dataset("faces", data=faces)
    if "shared_faces" in src_piece:
        dst_piece.create_dataset("shared_faces", data=np.asarray(src_piece["shared_faces"][:], dtype=np.int64))
    else:
        dst_piece.create_dataset("shared_faces", data=-np.ones(len(faces), dtype=np.int64))


def build_garf_subset_hdf5(source_hdf5: Path, sample_name: str, out_hdf5: Path) -> None:
    out_hdf5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_hdf5, "r") as src, h5py.File(out_hdf5, "w") as dst:
        src_group = src[sample_name]
        dst_group = dst.create_group(sample_name)
        dst_pieces = dst_group.create_group("pieces")

        piece_names: list[bytes] = []
        if "pieces" in src_group:
            src_pieces = src_group["pieces"]
            keys = sorted(src_pieces.keys(), key=lambda x: int(x))
            for i, k in enumerate(keys):
                _copy_piece(src_pieces[k], dst_pieces.create_group(str(i)))
                piece_names.append(f"Piece{i+1:02d}".encode("utf-8"))
            if "pieces_names" in src_group:
                raw_names = src_group["pieces_names"][:]
                piece_names = [_decode(n).encode("utf-8") for n in raw_names]
        else:
            keys = _numeric_piece_keys(src_group)
            for i, k in enumerate(keys):
                _copy_piece(src_group[k], dst_pieces.create_group(str(i)))
                piece_names.append(f"Piece{i+1:02d}".encode("utf-8"))

        num_pieces = len(piece_names)
        pname_dtype = h5py.special_dtype(vlen=str)
        dst_group.create_dataset("pieces_names", data=piece_names, dtype=pname_dtype)
        dst_group.create_dataset("removal_masks", data=np.ones((1, num_pieces), dtype=bool))
        dst_group.create_dataset("removal_order", data=np.arange(num_pieces, dtype=np.int64))

        ds_root = dst.create_group("data_split")
        diag = ds_root.create_group("diag")
        ref = np.array([sample_name.encode("utf-8")], dtype=object)
        for split_name in ("train", "val", "test"):
            diag.create_dataset(split_name, data=ref)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_angle_deg(q1_wxyz: np.ndarray, q2_wxyz: np.ndarray) -> float:
    r1 = R.from_quat(quat_wxyz_to_xyzw(q1_wxyz))
    r2 = R.from_quat(quat_wxyz_to_xyzw(q2_wxyz))
    deg = float((r1 * r2.inv()).magnitude() * 180.0 / np.pi)
    return min(deg, 360.0 - deg)


def spread_stats(points: np.ndarray) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, float(np.linalg.norm(points.mean(axis=0)))
    center = points.mean(axis=0)
    rel = points - center
    d = np.linalg.norm(rel[:, None] - rel[None, :], axis=2)
    return float(d.max()), float(np.linalg.norm(rel, axis=1).mean())


def sample_mesh_centroids(hdf5_path: Path, sample_name: str) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        g = f[sample_name]
        if "pieces" in g:
            pieces = g["pieces"]
            keys = sorted(pieces.keys(), key=lambda x: int(x))
            cents = [np.asarray(pieces[k]["vertices"][:], dtype=np.float64).mean(axis=0) for k in keys]
        else:
            keys = _numeric_piece_keys(g)
            cents = [np.asarray(g[k]["vertices"][:], dtype=np.float64).mean(axis=0) for k in keys]
    return np.asarray(cents, dtype=np.float64)


def run_eval(
    garf_root: Path,
    data_root: Path,
    experiment_name: str,
    seed: int,
    ckpt: str,
) -> Path:
    cmd = [
        "python",
        "eval.py",
        f"seed={seed}",
        "experiment=denoiser_flow_matching",
        f"experiment_name={experiment_name}",
        "loggers=csv",
        "loggers.csv.save_dir=logs/diagnostics",
        "trainer.num_nodes=1",
        "trainer.devices=1",
        "trainer.precision=bf16-mixed",
        f"data.data_root={data_root}",
        "data.categories=['diag']",
        "data.min_parts=2",
        "data.max_parts=20",
        "data.batch_size=1",
        "data.num_workers=4",
        "data.multi_ref=False",
        f"ckpt_path={ckpt}",
        "++data.random_anchor=false",
        "++model.inference_config.one_step_init=true",
        "++model.inference_config.write_to_json=true",
        "++model.inference_config.anchor_free=true",
        "++model.inference_config.deploy_mode=true",
        "++model.inference_config.save_assembly=false",
    ]
    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    subprocess.run(cmd, cwd=garf_root, check=True, env=env)
    return garf_root / "logs" / "diagnostics" / experiment_name / "version_0" / "json_results" / "0.json"


def summarize_case(
    case: str,
    sample_name: str,
    subset_hdf5: Path,
    result_json: Path,
    seed: int,
) -> dict[str, Any]:
    with open(result_json) as f:
        j = json.load(f)

    traj = np.asarray(j["pred_trans_rots"], dtype=np.float64)  # (T,P,7)
    first = traj[0]
    final = traj[-1]
    gt_tf = np.asarray(j.get("gt_transform", j.get("gt_trans_rots", final)), dtype=np.float64)

    mesh_centroids = sample_mesh_centroids(subset_hdf5, sample_name)
    mesh_maxd, mesh_meanr = spread_stats(mesh_centroids)
    pred_maxd, pred_meanr = spread_stats(final[:, :3])
    gt_maxd, gt_meanr = spread_stats(gt_tf[:, :3])

    traj_t_mean_start = float(np.linalg.norm(first[:, :3], axis=1).mean())
    traj_t_mean_final = float(np.linalg.norm(final[:, :3], axis=1).mean())
    rot_change = [quat_angle_deg(first[i, 3:], final[i, 3:]) for i in range(final.shape[0])]

    return {
        "case": case,
        "seed": seed,
        "sample": sample_name,
        "num_parts": int(final.shape[0]),
        "mesh_centroid_maxdist": mesh_maxd,
        "mesh_centroid_mean_radius": mesh_meanr,
        "target_t_maxdist": gt_maxd,
        "target_t_mean_radius": gt_meanr,
        "pred_t_maxdist": pred_maxd,
        "pred_t_mean_radius": pred_meanr,
        "pred_target_spread_ratio": (pred_maxd / gt_maxd) if gt_maxd > 1e-8 else math.nan,
        "traj_t_mean_start": traj_t_mean_start,
        "traj_t_mean_final": traj_t_mean_final,
        "traj_rot_change_mean_deg": float(np.mean(rot_change)),
        "traj_rot_change_max_deg": float(np.max(rot_change)),
        "part_acc": float(j.get("part_acc", math.nan)),
        "rmse_r": float(j.get("rmse_r", math.nan)),
        "rmse_t": float(j.get("rmse_t", math.nan)),
        "shape_cd": float(j.get("shape_cd", math.nan)),
        "result_json": str(result_json),
    }


def write_reports(rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No diagnostic rows to report.")

    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(out_md, "w") as f:
        f.write("# GARF matched-piece diagnostic summary\n\n")
        f.write("| case | seed | sample | parts | mesh_maxd | target_maxd | pred_maxd | pred/target | t_start | t_final | rot_drift_mean |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['case']} | {r['seed']} | `{r['sample']}` | {r['num_parts']} | "
                f"{r['mesh_centroid_maxdist']:.4f} | {r['target_t_maxdist']:.4f} | {r['pred_t_maxdist']:.4f} | "
                f"{r['pred_target_spread_ratio']:.4f} | {r['traj_t_mean_start']:.4f} | {r['traj_t_mean_final']:.4f} | "
                f"{r['traj_rot_change_mean_deg']:.2f} |\n"
            )
        f.write("\n")
        f.write(f"CSV: `{out_csv}`\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--garf-root", type=Path, default=Path("/data/gpfs/projects/punim2657/GARF"))
    parser.add_argument("--pieces", type=int, default=9, help="Match samples with this number of parts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt", default="output/GARF.ckpt")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/data/gpfs/projects/punim2657/GARF/logs/diagnostics/matched_runs"),
    )
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    garf_root = args.garf_root
    work_dir = args.work_dir / f"diag_{args.timestamp}"
    subset_dir = work_dir / "subset_hdf5"
    subset_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        CaseSpec(
            label="juglet_raw",
            source_hdf5=garf_root / "input" / "juglet_deploy.hdf5",
            split_key="artifact",
            fixed_sample="artifact/Juglet-000",
        ),
        CaseSpec(
            label="juglet_local02",
            source_hdf5=garf_root / "input" / "juglet_deploy_local02.hdf5",
            split_key="artifact",
            fixed_sample="artifact/Juglet-000",
        ),
        CaseSpec(
            label="bb_everyday",
            source_hdf5=Path("/data/gpfs/projects/punim2657/TORA/dataset/breaking_bad_vol.hdf5"),
            split_key="everyday",
            fixed_sample=None,
        ),
        CaseSpec(
            label="fractura_pig",
            source_hdf5=garf_root / "input" / "Fractura" / "bone_synthetic.hdf5",
            split_key="pig",
            fixed_sample=None,
        ),
    ]

    rows: list[dict[str, Any]] = []
    selected_json = work_dir / "selected_samples.json"
    selected: dict[str, str] = {}

    for case in cases:
        if not case.source_hdf5.exists():
            raise FileNotFoundError(f"{case.label}: missing source file {case.source_hdf5}")
        with h5py.File(case.source_hdf5, "r") as h5:
            sample = case.fixed_sample or pick_sample_with_parts(h5, case.split_key, args.pieces)
        selected[case.label] = sample

        subset_h5 = subset_dir / f"{case.label}.hdf5"
        build_garf_subset_hdf5(case.source_hdf5, sample, subset_h5)

        exp = f"diag_{case.label}_{args.timestamp}_s{args.seed}"
        result_json = run_eval(
            garf_root=garf_root,
            data_root=subset_h5,
            experiment_name=exp,
            seed=args.seed,
            ckpt=args.ckpt,
        )
        row = summarize_case(
            case=case.label,
            sample_name=sample,
            subset_hdf5=subset_h5,
            result_json=result_json,
            seed=args.seed,
        )
        rows.append(row)
        print(
            f"[{case.label}] sample={sample} parts={row['num_parts']} "
            f"pred_maxd={row['pred_t_maxdist']:.4f} target_maxd={row['target_t_maxdist']:.4f}"
        )

    with open(selected_json, "w") as f:
        json.dump(selected, f, indent=2)

    out_csv = work_dir / "diagnostic_summary.csv"
    out_md = work_dir / "diagnostic_summary.md"
    write_reports(rows, out_csv, out_md)
    print(f"\nSelected samples: {selected_json}")
    print(f"Summary CSV: {out_csv}")
    print(f"Summary MD:  {out_md}")


if __name__ == "__main__":
    main()
