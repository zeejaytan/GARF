#!/usr/bin/env python3
"""Aggregate GARF benchmark metrics across domain-bridge runs.

Scans <root>/<prefix>_<src>_<tag>_s<seed>/version_0/json_results/0.json and
aggregates part_acc / rmse_r / rmse_t / shape_cd by (src, tag), mean±std over
seeds. `tag` encodes the transform+strength (e.g. base, noise002, dec25, open25).
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

NAME_RE = re.compile(r"^(?P<src>[A-Za-z0-9]+)_(?P<tag>[A-Za-z0-9]+)_s(?P<seed>\d+)$")

# order tags for readable tables
TAG_ORDER = ["base",
             "noise0005", "noise001", "noise002", "noise005",
             "dec50", "dec25", "dec10",
             "open10", "open25"]


def collect(root: Path, prefix: str) -> list[dict[str, Any]]:
    rows = []
    for d in sorted(root.glob(f"{prefix}_*")):
        if not d.is_dir():
            continue
        rest = d.name[len(prefix) + 1:]
        m = NAME_RE.match(rest)
        if not m:
            continue
        jp = d / "version_0" / "json_results" / "0.json"
        if not jp.exists():
            print(f"  WARN missing {jp}")
            continue
        j = json.load(open(jp))
        rows.append({
            "src": m.group("src"), "tag": m.group("tag"), "seed": int(m.group("seed")),
            "num_parts": j.get("num_parts"),
            "part_acc": j.get("part_acc", float("nan")),
            "rmse_r": j.get("rmse_r", float("nan")),
            "rmse_t": j.get("rmse_t", float("nan")),
            "shape_cd": j.get("shape_cd", float("nan")),
        })
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["src"], r["tag"])].append(r)

    def ms(rs, key):
        vals = [r[key] for r in rs if isinstance(r[key], (int, float)) and r[key] == r[key]]
        if not vals:
            return float("nan"), float("nan")
        return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0)

    out = []
    for (src, tag), rs in groups.items():
        pa = ms(rs, "part_acc"); rr = ms(rs, "rmse_r"); rt = ms(rs, "rmse_t"); cd = ms(rs, "shape_cd")
        out.append({"src": src, "tag": tag, "n_seeds": len(rs), "num_parts": rs[0]["num_parts"],
                    "part_acc_mean": pa[0], "part_acc_std": pa[1],
                    "rmse_r_mean": rr[0], "rmse_r_std": rr[1],
                    "rmse_t_mean": rt[0], "rmse_t_std": rt[1],
                    "shape_cd_mean": cd[0], "shape_cd_std": cd[1]})

    def keyfn(a):
        ti = TAG_ORDER.index(a["tag"]) if a["tag"] in TAG_ORDER else 99
        return (a["src"], ti, a["tag"])
    return sorted(out, key=keyfn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("logs/diagnostics"))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = collect(args.root, args.prefix)
    if not rows:
        raise SystemExit(f"No runs under {args.root} with prefix {args.prefix}")
    agg = aggregate(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.out / "aggregate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)

    lines = ["# Domain bridge — correctness vs Juglet-like degradation\n",
             "Each transform deforms fragments in their local frame (GT pose preserved).",
             "base = untouched (sanity gate; should match the known-good baseline).\n",
             "| src | tag | seeds | parts | part_acc | rmse_r | rmse_t | shape_cd |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for a in agg:
        lines.append(f"| {a['src']} | {a['tag']} | {a['n_seeds']} | {a['num_parts']} | "
                     f"{a['part_acc_mean']:.3f}±{a['part_acc_std']:.3f} | "
                     f"{a['rmse_r_mean']:.1f}±{a['rmse_r_std']:.1f} | "
                     f"{a['rmse_t_mean']:.3f}±{a['rmse_t_std']:.3f} | "
                     f"{a['shape_cd_mean']:.4f}±{a['shape_cd_std']:.4f} |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
