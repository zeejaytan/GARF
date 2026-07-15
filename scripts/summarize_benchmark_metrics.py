#!/usr/bin/env python3
"""Summarize GARF benchmark metrics across reference/config/seed runs.

Scans logs/<save_dir>/<prefix>_<label>_<config>_s<seed>/version_0/json_results/0.json
and aggregates part_acc / rmse_r / rmse_t / shape_cd by (label, config), reporting
mean and std across seeds. Used to test whether one_step_init vs full-schedule
changes correctness on objects that HAVE ground truth (BB / Fractura).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

NAME_RE = re.compile(r"^(?P<label>.+)_(?P<config>init[01])_s(?P<seed>\d+)$")


def collect(root: Path, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in sorted(root.glob(f"{prefix}_*")):
        if not d.is_dir():
            continue
        rest = d.name[len(prefix) + 1 :]  # strip "<prefix>_"
        m = NAME_RE.match(rest)
        if not m:
            continue
        jpath = d / "version_0" / "json_results" / "0.json"
        if not jpath.exists():
            print(f"  WARN missing {jpath}")
            continue
        with open(jpath) as f:
            j = json.load(f)
        rows.append(
            {
                "label": m.group("label"),
                "config": m.group("config"),
                "seed": int(m.group("seed")),
                "num_parts": j.get("num_parts"),
                "part_acc": j.get("part_acc", float("nan")),
                "rmse_r": j.get("rmse_r", float("nan")),
                "rmse_t": j.get("rmse_t", float("nan")),
                "shape_cd": j.get("shape_cd", float("nan")),
                "name": j.get("name", ""),
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["config"])].append(r)
    out = []
    for (label, config), rs in sorted(groups.items()):
        def ms(key: str) -> tuple[float, float]:
            vals = [r[key] for r in rs if isinstance(r[key], (int, float)) and r[key] == r[key]]
            if not vals:
                return float("nan"), float("nan")
            return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)
        pa_m, pa_s = ms("part_acc")
        rr_m, rr_s = ms("rmse_r")
        rt_m, rt_s = ms("rmse_t")
        cd_m, cd_s = ms("shape_cd")
        out.append(
            {
                "label": label,
                "config": config,
                "n_seeds": len(rs),
                "num_parts": rs[0]["num_parts"],
                "part_acc_mean": pa_m, "part_acc_std": pa_s,
                "rmse_r_mean": rr_m, "rmse_r_std": rr_s,
                "rmse_t_mean": rt_m, "rmse_t_std": rt_s,
                "shape_cd_mean": cd_m, "shape_cd_std": cd_s,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("logs/diagnostics"))
    ap.add_argument("--prefix", required=True, help="experiment_name prefix, e.g. bench_cmp_20260529_142500")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = collect(args.root, args.prefix)
    if not rows:
        raise SystemExit(f"No runs found under {args.root} with prefix {args.prefix}")
    agg = aggregate(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(args.out / "aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)

    lines = ["# one_step_init vs full-schedule — correctness on GT objects\n",
             "(init1 = one_step_init=true [warmup + 20-step schedule]; "
             "init0 = full schedule only)\n",
             "\n| label | config | seeds | parts | part_acc | rmse_r(deg) | rmse_t | shape_cd |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for a in agg:
        lines.append(
            f"| {a['label']} | {a['config']} | {a['n_seeds']} | {a['num_parts']} | "
            f"{a['part_acc_mean']:.3f}±{a['part_acc_std']:.3f} | "
            f"{a['rmse_r_mean']:.1f}±{a['rmse_r_std']:.1f} | "
            f"{a['rmse_t_mean']:.3f}±{a['rmse_t_std']:.3f} | "
            f"{a['shape_cd_mean']:.4f}±{a['shape_cd_std']:.4f} |"
        )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out}/summary.md (+ per_run.csv, aggregate.csv)")


if __name__ == "__main__":
    main()
