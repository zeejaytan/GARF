#!/usr/bin/env python3
"""Exp 14 — de-weathering sufficiency test ("un-wear the Juglet").

The mechanism chain so far proves NECESSITY only in the forward direction:
Exp 10 (encoder blind to Juglet's worn rims: fired% 0.57% vs 3.4% control) and
Exp 10b (eroding fresh breaks toward worn *causes* the blindness, 9.8%→2.7%).
The missing, decisive proof that the blindness is THE cause — not a correlate —
is SUFFICIENCY on the failing object itself:

    de-wear the Juglet's fracture bands  →  encoder fires again (arm A)
                                         →  pairwise mating returns (arm B)

`fracture_mesh_ops.sharpen_fracture_band_solo` inverts the causally-validated
mollifier wear model (surface unsharp mask on the relief-detected band,
pose-free, complementarity-preserving). If arm B moves Juglet's true-mate
chamfer (baseline 0.070, == non-mate 0.073) toward the control's 0.024 with
mate/non separation, the encoder-blindness cause is proven sufficient AND the
transform is itself remedy path 1: an inference-time de-weathering
preprocessor (no pose, no retraining needed).

Controls:
  arm C (specificity) : sharpen only OFF-band (original vessel surface) —
                        must NOT raise fired% / restore mating, ruling out
                        "any added high-frequency detail helps".
  arm D (regression)  : same transform on the 4 fresh control ceramics —
                        their 0.024 true-mate chamfer must survive.

Pre-registered gates (also in JUGLET_ROOTCAUSE_EXPERIMENT_PLAN.md):
  A  PERCEPTION RESTORED : juglet fired% >= 1.5% at some strength
                           (>=2.5x the 0.57% baseline); STRONG >= 3%.
  C  SPECIFICITY HOLDS   : offband fired% < 1.5x the 0.57% baseline.
  B  SUFFICIENCY         : true-mate median chamfer/diag <= 0.045 AND
                           mate/non separation >= 1.25x (PARTIAL: <=0.055
                           or separation >= 1.15x).
  D  NO REGRESSION       : control true-mate median <= 0.030.

Subcommands
-----------
probe : sweep sharpen strengths, run the frozen FracSeg fired% readout
        (Exp 10 probe) on de-weathered Juglet + control pieces.
build : copy an HDF5 (deploy or pairs layout) and de-weather every piece's
        vertices IN PLACE (shapes/faces/poses unchanged, so all downstream
        tooling — build_juglet_pairs_hdf5, eval.py, pair_reference_chamfer —
        works verbatim).

Usage
-----
  python scripts/exp14_dewear_sufficiency.py probe \
      --ckpt output/feature_extractor.ckpt \
      --juglet-mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
      --control-source input/Fractura/fractura_real.hdf5 \
      --control-objects ceramics/pink_bowl ceramics/blue_pot \
      --strengths 0.0 0.5 1.0 2.0 3.0 --region band \
      --out logs/diagnostics/exp14_probe_band_<stamp>

  python scripts/exp14_dewear_sufficiency.py build \
      --source input/juglet_deploy_local02.hdf5 \
      --strength 2.0 --out input/juglet_deploy_dewear.hdf5
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from statistics import mean

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fracture_mesh_ops import piece_relief_stats, sharpen_fracture_band_solo


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def load_juglet_pieces(mesh_dir: Path):
    import trimesh

    pieces = []
    for obj in sorted(mesh_dir.glob("*.obj")):
        m = trimesh.load(str(obj), process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        pieces.append((np.asarray(m.vertices, np.float64), np.asarray(m.faces)))
    return pieces


def fired_frac(model, pieces, device, n_per_piece, seed):
    """Fraction of sampled points with P(fracture) > 0.5, pooled over pieces
    (the Exp 10 validated label-free readout)."""
    from fracseg_introspection import frac_prob, sample_piece

    probs = []
    for k, (v, f) in enumerate(pieces):
        pts, nrm, _, _ = sample_piece(v, f, n_per_piece, seed + k)
        probs.append(frac_prob(model, pts, nrm, device))
    prob = np.concatenate(probs)
    return float((prob > 0.5).mean()), float(prob.mean())


def sharpen_pieces(pieces, strength, region, seed, kwargs):
    return [
        (sharpen_fracture_band_solo(v, f, strength, region=region,
                                    seed=seed + k, **kwargs), f)
        for k, (v, f) in enumerate(pieces)
    ]


def cmd_probe(args) -> None:
    import torch

    from fracseg_introspection import load_fracseg

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_fracseg(args.ckpt, device)
    print("frozen FracSeg loaded")

    sharpen_kwargs = dict(relief_pct=args.relief_pct, band_frac=args.band_frac,
                          kernel_frac=args.kernel_frac)

    juglet = load_juglet_pieces(args.juglet_mesh_dir)
    print(f"juglet pieces: {len(juglet)}")
    controls = {}
    if args.control_source and args.control_objects:
        from fracseg_introspection import load_control_pieces

        for obj in args.control_objects:
            controls[obj], _ = load_control_pieces(args.control_source, obj)

    rows = []
    for s in args.strengths:
        jp = sharpen_pieces(juglet, s, args.region, args.seed, sharpen_kwargs)
        j_fired, j_meanp = fired_frac(model, jp, device, args.n_per_piece, args.seed)
        relief = mean(piece_relief_stats(v, f)["relief_p90"] for v, f in jp)
        ctrl_fired = float("nan")
        if controls:
            cf = []
            for obj, pieces in controls.items():
                cp = sharpen_pieces(pieces, s, args.region, args.seed, sharpen_kwargs)
                cf.append(fired_frac(model, cp, device, args.n_per_piece, args.seed)[0])
            ctrl_fired = mean(cf)
        rows.append({"strength": s, "region": args.region,
                     "juglet_fired": j_fired, "juglet_mean_prob": j_meanp,
                     "juglet_band_relief_p90": relief, "control_fired": ctrl_fired})
        print(f"[s={s:.2f} {args.region}] juglet fired%={j_fired*100:.2f} "
              f"meanP={j_meanp:.3f} relief_p90={relief:.3f} "
              f"control fired%={ctrl_fired*100:.2f}")

    best = max(rows, key=lambda r: r["juglet_fired"])
    if args.region == "band":
        perception_restored = best["juglet_fired"] >= 0.015
        strong = best["juglet_fired"] >= 0.03
        verdict = ("STRONG" if strong else
                   "PERCEPTION RESTORED" if perception_restored else "NOT RESTORED")
    else:
        specificity_holds = best["juglet_fired"] < 1.5 * 0.0057
        verdict = "SPECIFICITY HOLDS" if specificity_holds else "SPECIFICITY VIOLATED"

    summary = {"params": {"region": args.region, "strengths": args.strengths,
                          **sharpen_kwargs, "n_per_piece": args.n_per_piece,
                          "seed": args.seed},
               "rows": rows, "best_strength": best["strength"],
               "best_juglet_fired": best["juglet_fired"], "verdict": verdict,
               "baselines": {"juglet_fired_exp10": 0.0057,
                             "control_fired_exp10": 0.034}}
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = [f"# Exp 14 probe — de-weathering perception sweep ({args.region})\n",
          "Frozen-encoder fracture response (Exp 10 fired% readout) on "
          "de-weathered pieces. Baselines: Juglet 0.57%, control 3.4%.\n",
          "| strength | juglet fired% | juglet mean P | juglet band relief_p90 | control fired% |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['strength']:.2f} | {r['juglet_fired']*100:.2f} | "
                  f"{r['juglet_mean_prob']:.3f} | {r['juglet_band_relief_p90']:.3f} | "
                  f"{r['control_fired']*100:.2f} |")
    md += [f"\n## Verdict: **{verdict}** (best strength {best['strength']:.2f}, "
           f"juglet fired% {best['juglet_fired']*100:.2f})\n"]
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"\nverdict: {verdict} | best strength {best['strength']:.2f} "
          f"({best['juglet_fired']*100:.2f}%)")
    print(f"wrote {args.out}/summary.md")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, args.out)
    sharpen_kwargs = dict(relief_pct=args.relief_pct, band_frac=args.band_frac,
                          kernel_frac=args.kernel_frac)
    n_pieces, disp_sum, disp_n = 0, 0.0, 0
    with h5py.File(args.out, "r+") as f:
        samples = [k2 for k1 in f if k1 != "data_split" and isinstance(f[k1], h5py.Group)
                   for k2 in (f"{k1}/{s}" for s in f[k1]) if "pieces" in f[k2]]
        for sname in samples:
            g = f[sname]["pieces"]
            for k in sorted(g.keys(), key=int):
                v = np.asarray(g[k]["vertices"][:], np.float64)
                fc = np.asarray(g[k]["faces"][:], np.int64)
                nv = sharpen_fracture_band_solo(v, fc, args.strength,
                                                seed=args.seed, region=args.region,
                                                **sharpen_kwargs)
                g[k]["vertices"][...] = nv
                d = np.linalg.norm(nv - v, axis=1)
                disp_sum += float(d.sum()); disp_n += len(d); n_pieces += 1
            print(f"  {sname}: {len(g)} pieces de-weathered")
    print(f"wrote {args.out}: {n_pieces} pieces, strength {args.strength}, "
          f"region {args.region}, mean |disp| = {disp_sum / max(disp_n, 1):.5f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_sharpen_args(p):
        p.add_argument("--relief-pct", type=float, default=85.0)
        p.add_argument("--band-frac", type=float, default=0.05)
        p.add_argument("--kernel-frac", type=float, default=0.03)
        p.add_argument("--region", choices=["band", "offband"], default="band")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("probe", help="perception sweep (arms A/C)")
    p.add_argument("--ckpt", default="output/feature_extractor.ckpt")
    p.add_argument("--juglet-mesh-dir", type=Path, required=True)
    p.add_argument("--control-source", type=Path)
    p.add_argument("--control-objects", nargs="*", default=[])
    p.add_argument("--strengths", type=float, nargs="+",
                   default=[0.0, 0.5, 1.0, 2.0, 3.0])
    p.add_argument("--n-per-piece", type=int, default=5000)
    p.add_argument("--out", type=Path, required=True)
    add_sharpen_args(p)
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("build", help="write de-weathered copy of an HDF5 (arms B/D)")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--strength", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    add_sharpen_args(p)
    p.set_defaults(func=cmd_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
