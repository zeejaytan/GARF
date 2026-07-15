#!/usr/bin/env python3
"""Summarize the Exp 5 inference-lever sweep.

Reads per-sample json_results written by eval.py for each
<prefix>_<config>_<category>_s<seed> run, aggregates per object across seeds,
and writes a markdown report describing whether each lever made the failing
thin-wall objects better or worse vs the baseline config.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

# Objects we care about (failing thin-wall) + controls (already work).
FAIL = [
    "ceramics/narrow_bottle1",
    "ceramics/narrow_bottle3",
    "ceramics/plate",
    "egg/egg1",
    "egg/egg2",
    "egg/egg3",
]
CTRL = [
    "ceramics/pink_bowl",
    "ceramics/narrow_bottle2",
    "ceramics/blue_pot",
    "ceramics/narrow_bottle4",
    "ceramics/galli_pot",
]

CONFIG_ORDER = [
    "baseline", "init0", "steps50", "steps100", "iters2", "iters3",
    "anchor_on", "sde", "sde_hi", "sigma_exp", "sigma_pwl",
]

DIR_RE = re.compile(r"_(?P<cfg>[a-z0-9_]+)_(?P<cat>ceramics|egg)_s(?P<seed>\d+)$")


def collect(root: Path, prefix: str):
    # data[cfg][object] = list of (seed, part_acc, rmse_r, num_parts)
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(glob.glob(str(root / f"{prefix}_*"))):
        name = Path(d).name
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        m = DIR_RE.search(rest)
        if not m:
            continue
        cfg = m.group("cfg")
        seed = int(m.group("seed"))
        for jf in glob.glob(str(Path(d) / "version_0" / "json_results" / "*.json")):
            try:
                j = json.load(open(jf))
            except Exception:
                continue
            obj = j["name"]
            data[cfg][obj].append(
                (seed, float(j["part_acc"]), float(j["rmse_r"]), int(j["num_parts"]))
            )
    return data


def mean_pacc(rows):
    return mean(r[1] for r in rows) if rows else float("nan")


def best_pacc(rows):
    return max(r[1] for r in rows) if rows else float("nan")


def fmt(x):
    return "  -  " if x != x else f"{x:5.2f}"


def md_table(data, objects, configs, reducer):
    lines = []
    header = "| object | " + " | ".join(configs) + " |"
    sep = "|---|" + "---|" * len(configs)
    lines.append(header)
    lines.append(sep)
    for obj in objects:
        cells = []
        for cfg in configs:
            rows = data.get(cfg, {}).get(obj, [])
            cells.append(fmt(reducer(rows)))
        lines.append(f"| {obj} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(data, objects, configs, baseline="baseline"):
    """part_acc delta vs baseline (mean over seeds)."""
    lines = ["| object | base | " + " | ".join(c for c in configs if c != baseline) + " |"]
    lines.append("|---|---|" + "---|" * (len(configs) - 1))
    for obj in objects:
        base_rows = data.get(baseline, {}).get(obj, [])
        base = mean_pacc(base_rows)
        cells = []
        for cfg in configs:
            if cfg == baseline:
                continue
            rows = data.get(cfg, {}).get(obj, [])
            v = mean_pacc(rows)
            if v != v or base != base:
                cells.append("  -  ")
            else:
                d = v - base
                sign = "+" if d >= 0 else ""
                cells.append(f"{sign}{d:4.2f}")
        lines.append(f"| {obj} | {fmt(base)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def category_means(data, configs):
    lines = ["| config | FAIL mean | CTRL mean | FAIL best-of-3 |"]
    lines.append("|---|---|---|---|")
    for cfg in configs:
        fail_vals = [mean_pacc(data.get(cfg, {}).get(o, [])) for o in FAIL]
        ctrl_vals = [mean_pacc(data.get(cfg, {}).get(o, [])) for o in CTRL]
        fail_best = [best_pacc(data.get(cfg, {}).get(o, [])) for o in FAIL]
        fail_vals = [v for v in fail_vals if v == v]
        ctrl_vals = [v for v in ctrl_vals if v == v]
        fail_best = [v for v in fail_best if v == v]
        lines.append(
            f"| {cfg} | {fmt(mean(fail_vals) if fail_vals else float('nan'))} "
            f"| {fmt(mean(ctrl_vals) if ctrl_vals else float('nan'))} "
            f"| {fmt(mean(fail_best) if fail_best else float('nan'))} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = collect(args.root, args.prefix)
    present = [c for c in CONFIG_ORDER if c in data]
    extra = [c for c in data if c not in CONFIG_ORDER]
    configs = present + sorted(extra)

    out = []
    out.append(f"# Exp 5 - Inference lever sweep ({args.prefix})\n")
    out.append(
        "Per-object `part_acc` (mean over seeds 41/42/43). One lever changed at a "
        "time vs `baseline` (init1, anchor_free, 20 steps, ODE, sigma=linear).\n"
    )
    out.append("## Summary: effect on failing vs control objects\n")
    out.append(category_means(data, configs))
    out.append("\n\n## Failing thin-wall objects - mean part_acc\n")
    out.append(md_table(data, FAIL, configs, mean_pacc))
    out.append("\n\n## Failing thin-wall objects - delta vs baseline\n")
    out.append(delta_table(data, FAIL, configs))
    out.append("\n\n## Failing thin-wall objects - best-of-3 seeds part_acc\n")
    out.append(md_table(data, FAIL, configs, best_pacc))
    out.append("\n\n## Control objects (should stay high) - mean part_acc\n")
    out.append(md_table(data, CTRL, configs, mean_pacc))
    out.append("\n")

    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out / "summary.md"
    report.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
