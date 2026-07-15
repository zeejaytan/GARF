#!/usr/bin/env python3
"""Measure the fracture-mating surface between assembled pieces (GARF's signal).

GARF assembles by matching fracture surfaces. For each Exp4c object we already
have the GT-assembled pieces as separate geometries in view_gt.glb. For every
piece we compute the fraction of its surface that lies in contact with some other
piece (the mating/fracture band) and how 'thick' vs 'thin' that band is. The
hypothesis: objects whose pieces share little / very thin mating surface give
GARF almost nothing to latch onto -> failure, independent of part count.

Correlate per-object mean contact fraction with part_acc.
"""

from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def part_clouds(glb_path: Path, n_per: int = 2500) -> list[np.ndarray]:
    scene = trimesh.load(glb_path, force="scene")
    meshes = [scene] if isinstance(scene, trimesh.Trimesh) else scene.dump(concatenate=False)
    clouds = []
    for m in meshes:
        if len(getattr(m, "faces", [])) == 0:
            continue
        try:
            p, _ = trimesh.sample.sample_surface_even(m, n_per * 2)
        except Exception:
            p, _ = trimesh.sample.sample_surface(m, n_per * 2)
        p = np.asarray(p)
        if len(p) > n_per:
            p = p[np.random.default_rng(0).choice(len(p), n_per, replace=False)]
        clouds.append(p)
    return clouds


def object_scale(clouds: list[np.ndarray]) -> float:
    allp = np.vstack(clouds)
    return float(np.linalg.norm(allp.max(0) - allp.min(0)))


def contact_stats(clouds: list[np.ndarray], tau_frac: float = 0.02) -> dict:
    P = len(clouds)
    scale = object_scale(clouds)
    tau = tau_frac * scale
    trees = [cKDTree(c) for c in clouds]
    # per-piece fraction of points in contact with ANY other piece
    in_contact_any = [np.zeros(len(clouds[i]), dtype=bool) for i in range(P)]
    pair_contacts = []
    for i, j in combinations(range(P), 2):
        dij, _ = trees[j].query(clouds[i])
        dji, _ = trees[i].query(clouds[j])
        ci = dij < tau
        cj = dji < tau
        in_contact_any[i] |= ci
        in_contact_any[j] |= cj
        frac = max(ci.mean(), cj.mean())
        if frac > 0.005:
            pair_contacts.append(frac)
    per_piece_frac = [float(m.mean()) for m in in_contact_any]
    return {
        "mean_piece_contact_frac": float(np.mean(per_piece_frac)),
        "min_piece_contact_frac": float(np.min(per_piece_frac)),
        "n_contact_pairs": len(pair_contacts),
        "mean_pair_contact_frac": float(np.mean(pair_contacts)) if pair_contacts else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thinviz-stamp", default="20260605_163400")
    ap.add_argument("--thinwall-stamp", default="20260605_162150")
    ap.add_argument("--tau-frac", type=float, default=0.02)
    args = ap.parse_args()

    metrics = {}
    for d in glob.glob(f"logs/diagnostics/thinwall_{args.thinwall_stamp}_*_init1_s41/version_0/json_results/*.json"):
        j = json.load(open(d))
        metrics[j["name"]] = j

    rows = []
    deploy = ROOT / "logs" / "deploy"
    for cat_dir in sorted(deploy.glob(f"thinviz_{args.thinviz_stamp}_*")):
        cat = cat_dir.name.split(f"thinviz_{args.thinviz_stamp}_", 1)[-1]
        asm = cat_dir / "version_0" / "assembly_results" / cat
        if not asm.is_dir():
            continue
        for sd in sorted(asm.iterdir()):
            gt = sd / "view_gt.glb"
            if not gt.exists():
                continue
            key = f"{cat}/{sd.name}"
            m = metrics.get(key, {})
            clouds = part_clouds(gt)
            if len(clouds) < 2:
                continue
            s = contact_stats(clouds, args.tau_frac)
            rows.append({
                "key": key, "cat": cat, "parts": m.get("num_parts"),
                "part_acc": m.get("part_acc", float("nan")),
                **s,
            })
            print(f"  {key:30s} P={m.get('num_parts'):>2} acc={m.get('part_acc',0):.2f} "
                  f"meanContact={s['mean_piece_contact_frac']:.3f} minContact={s['min_piece_contact_frac']:.3f}")

    print("\n=== ceramics + egg, sorted by mean piece contact fraction ===")
    sub = [r for r in rows if r["cat"] in ("ceramics", "egg")]
    sub.sort(key=lambda r: r["mean_piece_contact_frac"])
    print(f"{'object':28s} {'P':>2} {'acc':>5} {'meanC':>6} {'minC':>6} {'pairC':>6}")
    for r in sub:
        flag = " <- FAILS" if r["part_acc"] < 0.6 else ""
        print(f"{r['key']:28s} {r['parts']:>2} {r['part_acc']:>5.2f} "
              f"{r['mean_piece_contact_frac']:>6.3f} {r['min_piece_contact_frac']:>6.3f} {r['mean_pair_contact_frac']:>6.3f}{flag}")

    acc = np.array([r["part_acc"] for r in rows])
    mc = np.array([r["mean_piece_contact_frac"] for r in rows])
    minc = np.array([r["min_piece_contact_frac"] for r in rows])
    print(f"\nAll objects: r(meanContact, acc)={np.corrcoef(mc,acc)[0,1]:.3f}  r(minContact, acc)={np.corrcoef(minc,acc)[0,1]:.3f}")
    cer = [r for r in rows if r["cat"] == "ceramics"]
    if len(cer) > 2:
        a = np.array([r["part_acc"] for r in cer]); c = np.array([r["min_piece_contact_frac"] for r in cer])
        print(f"Ceramics only: r(minContact, acc)={np.corrcoef(c,a)[0,1]:.3f}")
    out = ROOT / "logs" / "diagnostics" / "fracture_contact_analysis.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
