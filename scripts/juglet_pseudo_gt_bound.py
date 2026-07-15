#!/usr/bin/env python3
"""Exp 9 — bound the PF++ pseudo-GT label error in the Juglet pairwise oracle.

Motivation (JUGLET_ROOTCAUSE_FINDINGS.md, 2026-07-10 addendum): the headline
Juglet failure number — true-mate symmetry-invariant chamfer/diag 0.070 vs
control 0.024 — scores GARF's predictions against the PF++ pseudo-assembly.
If PF++'s relative poses are themselves loose, part of that 0.070 is LABEL
error, not GARF error. This probe bounds that contribution.

Method, per TRUE-MATE pair (piece0 = anchor, piece1 = mate):

  1. REFINE the reference: band-constrained ICP snaps piece1's fracture-band
     points onto piece0's fracture-band points, initialised at the reference
     pose. For a true mating pair the two break faces are two copies of the
     same fracture surface, so at the correct pose they coincide; ICP restricted
     to the mutual contact band is a local mate refinement that cannot slide far
     (it only sees band points).
  2. DRIFT = pair_error(original reference pair, refined reference pair) — the
     label shift expressed in the SAME chamfer/diag units as the headline
     metric (global re-registration included, so the unobservable symmetry DOF
     is not charged).
  3. RE-SCORE the existing GARF predictions (per-seed GLBs from the original
     pairwise-oracle runs) against the REFINED reference; compare with the
     original scores loaded from the stored pairs.json.

Built-in validation: the identical procedure runs on the CONTROL ceramics,
whose reference is real GT. There the refinement must barely move (drift ~0)
and re-scored medians must match the originals — any systematic shift is the
procedure's own bias, subtracted before interpreting Juglet.

Reading the result:
  - Juglet drift is the label looseness; (orig median - rescored median),
    net of control bias, bounds the pseudo-GT contribution to the 0.070.
  - If the re-scored Juglet true-mate median stays near 0.070 -> labels are
    fine; the perceptual-failure conclusion stands at full strength.
  - If it collapses toward the control's 0.024 -> the oracle overstated
    GARF's failure and the PF++ pseudo-GT must be replaced before further
    remedy work.

Usage
-----
  python scripts/juglet_pseudo_gt_bound.py \
      --control-hdf5 input/control_ceramics_pairs.hdf5 \
      --control-adjacency logs/diagnostics/control_ceramics_adjacency.json \
      --control-run-dirs logs/deploy/exp6b_ctrl_20260709_144515_s41 ..._s42 ..._s43 \
      --control-orig logs/diagnostics/pair_chamfer_control/pairs.json \
      --juglet-adjacency logs/diagnostics/juglet_adjacency/adjacency.json \
      --juglet-run-dirs logs/deploy/exp6_pairs_20260610_162659_s41 ..._s42 ..._s43 \
      --juglet-orig logs/diagnostics/pair_chamfer_juglet/pairs.json \
      --pfpp-dir <.../inference/juglet_deploy/0> \
      --mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
      --out logs/diagnostics/exp9_pseudogt_<stamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_reference_chamfer import (  # noqa: E402
    _kabsch,
    apply,
    build_pfpp_reference,
    find_pairs,
    load_glb_pieces,
    pair_error,
    sample,
)


def _icp_trimmed(src: np.ndarray, dst: np.ndarray, init: np.ndarray,
                 iters: int = 50, keep: float = 0.7) -> np.ndarray:
    """Trimmed point-to-point ICP: each iteration fits the rigid step on only
    the best ``keep`` fraction of correspondences. Plain ICP over-pulls a
    fracture band toward full coincidence (band-edge points where the mating
    surface curves away drag the fit), which showed up as ~0.009 chamfer/diag
    drift even on real-GT control pairs; trimming suppresses exactly those
    edge correspondences. Returns the 4x4 refinement transform."""
    tree = cKDTree(dst)
    M = init.copy()
    cur = apply(M, src)
    n_keep = max(int(keep * len(src)), 10)
    for _ in range(iters):
        d, idx = tree.query(cur)
        sel = np.argsort(d)[:n_keep]
        step = _kabsch(cur[sel], dst[idx[sel]])
        M = step @ M
        new = apply(M, src)
        if np.max(np.abs(new - cur)) < 1e-6:
            cur = new
            break
        cur = new
    return M


# --------------------------------------------------------------------------- #
# band-constrained mate refinement
# --------------------------------------------------------------------------- #
def contact_stats(pa: np.ndarray, pb: np.ndarray, scale: float,
                  gap_tau: float = 0.03) -> tuple[float, float]:
    """(min_gap/scale, contact fraction at gap_tau*scale) between two point sets.
    Same definition as build_control_pairs_hdf5.is_true_mate."""
    ta, tb = cKDTree(pa), cKDTree(pb)
    dab, _ = tb.query(pa)
    dba, _ = ta.query(pb)
    min_gap = float(min(dab.min(), dba.min()))
    tau = gap_tau * scale
    cfrac = float(max(np.mean(dab < tau), np.mean(dba < tau)))
    return min_gap / scale, cfrac


def refine_mate_pose(
    ref0: tuple[np.ndarray, np.ndarray],
    ref1: tuple[np.ndarray, np.ndarray],
    *,
    band_tau_frac: float = 0.02,
    feather_mult: float = 3.0,
    n_samples: int = 20000,
    icp_iters: int = 50,
    min_band_pts: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray | None, dict]:
    """Refine piece1's pose by ICP restricted to the mutual fracture band.

    Returns (M, info): M is the 4x4 refinement applied to piece1's vertices in
    the reference frame (None if no usable contact band exists), and info holds
    the band sizes plus before/after contact statistics.
    """
    (v0, f0), (v1, f1) = ref0, ref1
    rng = np.random.default_rng(seed)
    p0 = sample(v0, f0, n_samples, seed=int(rng.integers(2**31)))
    p1 = sample(v1, f1, n_samples, seed=int(rng.integers(2**31)))

    allv = np.concatenate([v0, v1], axis=0)
    scale = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    band_r = feather_mult * band_tau_frac * scale

    d1, _ = cKDTree(p0).query(p1)
    d0, _ = cKDTree(p1).query(p0)
    src = p1[d1 < band_r]   # piece1 points near piece0 (the mate's band)
    dst = p0[d0 < band_r]   # piece0 points near piece1 (the anchor's band)

    gap0, cfrac0 = contact_stats(p0, p1, scale)
    info = {
        "scale": scale,
        "n_band_src": int(len(src)),
        "n_band_dst": int(len(dst)),
        "min_gap_over_scale_ref": gap0,
        "contact_frac_ref": cfrac0,
    }
    if len(src) < min_band_pts or len(dst) < min_band_pts:
        info["min_gap_over_scale_refined"] = float("nan")
        info["contact_frac_refined"] = float("nan")
        return None, info

    M = _icp_trimmed(src, dst, np.eye(4), iters=icp_iters)
    gap1, cfrac1 = contact_stats(p0, apply(M, p1), scale)
    info["min_gap_over_scale_refined"] = gap1
    info["contact_frac_refined"] = cfrac1
    return M, info


# --------------------------------------------------------------------------- #
# reference loading (mirrors pair_reference_chamfer conventions)
# --------------------------------------------------------------------------- #
def control_ref_pieces(hf: h5py.File, sample_key: str):
    g = hf[f"control/{sample_key}"]["pieces"]
    keys = sorted(g.keys(), key=int)
    pcs = [(np.asarray(g[k]["vertices"][:], np.float64),
            np.asarray(g[k]["faces"][:])) for k in keys]
    pcs.sort(key=lambda vf: -len(vf[0]))
    return pcs[0], pcs[1]


def juglet_ref_pieces(pfpp_ref: dict, sample_key: str):
    tag = sample_key.split("-p")[-1]
    i, j = int(tag[:2]) - 1, int(tag[2:]) - 1
    pcs = [pfpp_ref[i], pfpp_ref[j]]
    pcs.sort(key=lambda vf: -len(vf[0]))
    return pcs[0], pcs[1]


def load_orig_scores(pairs_json: Path) -> dict[str, float]:
    rows = json.load(open(pairs_json))
    return {r["sample"]: r["chamfer_over_diag"] for r in rows}


# --------------------------------------------------------------------------- #
# per-dataset processing
# --------------------------------------------------------------------------- #
def process_dataset(
    name: str,
    true_mates: list[str],
    ref_lookup,
    run_dirs: list[Path],
    glb_subdir: str,
    orig_scores: dict[str, float],
    icp_seed: int,
) -> list[dict]:
    per_run = {d: find_pairs(d, glb_subdir) for d in run_dirs}
    rows = []
    for s in true_mates:
        ref0, ref1 = ref_lookup(s)
        M, info = refine_mate_pose(ref0, ref1, seed=icp_seed)
        if M is None:
            drift = float("nan")
            ref1_refined = ref1
            print(f"  [{name}] {s}: no usable contact band "
                  f"(src={info['n_band_src']}, dst={info['n_band_dst']}) — skipped refinement")
        else:
            ref1_refined = (apply(M, ref1[0]), ref1[1])
            drift = pair_error([ref0, ref1_refined], ref0, ref1)

        rescored_errs = []
        for d in run_dirs:
            adir = per_run[d].get(s)
            if adir is None:
                continue
            pred_pieces = load_glb_pieces(adir / "predicted_assembly.glb")
            if len(pred_pieces) != 2:
                continue
            e = pair_error(pred_pieces, ref0, ref1_refined)
            if e == e:
                rescored_errs.append(e)
        rescored = median(rescored_errs) if rescored_errs else float("nan")
        orig = orig_scores.get(s, float("nan"))
        rows.append({
            "sample": s,
            "drift_chamfer_over_diag": drift,
            "orig_chamfer_over_diag": orig,
            "rescored_chamfer_over_diag": rescored,
            "n_runs": len(rescored_errs),
            **info,
        })
        print(f"  [{name}] {s}: drift={drift:.4f} orig={orig:.4f} "
              f"rescored={rescored:.4f} "
              f"contact_frac {info['contact_frac_ref']:.3f}->"
              f"{info['contact_frac_refined']:.3f} (n={len(rescored_errs)})")
    return rows


def identity_floor(true_mates: list[str], ref_lookup, n_pairs: int = 5) -> float:
    """Median pair_error of a reference pair scored against ITSELF. Because
    every pair_error call resamples both surfaces independently, this is the
    metric's sampling-noise floor (~0.005 at n=4000): drift or rescored-vs-orig
    deltas at or below this value are indistinguishable from zero."""
    vals = []
    for s in true_mates[:n_pairs]:
        ref0, ref1 = ref_lookup(s)
        e = pair_error([ref0, ref1], ref0, ref1)
        if e == e:
            vals.append(e)
    return median(vals) if vals else float("nan")


def med_of(rows: list[dict], key: str) -> float:
    v = [r[key] for r in rows if r[key] == r[key]]
    return median(v) if v else float("nan")


def mean_of(rows: list[dict], key: str) -> float:
    v = [r[key] for r in rows if r[key] == r[key]]
    return mean(v) if v else float("nan")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control-hdf5", type=Path, required=True)
    ap.add_argument("--control-adjacency", type=Path, required=True)
    ap.add_argument("--control-run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--control-orig", type=Path, required=True,
                    help="pairs.json of the original control chamfer scoring")
    ap.add_argument("--juglet-adjacency", type=Path, required=True)
    ap.add_argument("--juglet-run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--juglet-orig", type=Path, required=True,
                    help="pairs.json of the original Juglet chamfer scoring")
    ap.add_argument("--pfpp-dir", type=Path, required=True)
    ap.add_argument("--mesh-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--icp-seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # --- control (real GT reference; validates the refinement procedure) ----
    ctrl_adj = json.load(open(args.control_adjacency))
    ctrl_mates = sorted(k for k, v in ctrl_adj["pairs"].items() if v["true_mate"])
    print(f"control: {len(ctrl_mates)} true-mate pairs")
    hf = h5py.File(args.control_hdf5, "r")
    ctrl_floor = identity_floor(ctrl_mates, lambda s: control_ref_pieces(hf, s))
    print(f"control identity (sampling-noise) floor: {ctrl_floor:.4f}")
    ctrl_rows = process_dataset(
        "control", ctrl_mates,
        lambda s: control_ref_pieces(hf, s),
        args.control_run_dirs, "control",
        load_orig_scores(args.control_orig), args.icp_seed,
    )

    # --- juglet (PF++ pseudo-GT reference; the quantity under test) ---------
    jug_adj = json.load(open(args.juglet_adjacency))
    jug_mate_tags = set(jug_adj["true_mates"])  # e.g. {"p0102", ...}
    jug_orig = load_orig_scores(args.juglet_orig)
    jug_mates = sorted(s for s in jug_orig
                       if ("p" + s.split("-p")[-1]) in jug_mate_tags)
    print(f"juglet: {len(jug_mates)} true-mate pairs")
    pfpp_ref = build_pfpp_reference(args.pfpp_dir, args.mesh_dir)
    jug_floor = identity_floor(jug_mates, lambda s: juglet_ref_pieces(pfpp_ref, s))
    print(f"juglet identity (sampling-noise) floor: {jug_floor:.4f}")
    jug_rows = process_dataset(
        "juglet", jug_mates,
        lambda s: juglet_ref_pieces(pfpp_ref, s),
        args.juglet_run_dirs, "artifact",
        jug_orig, args.icp_seed,
    )

    # --- aggregate + decision -----------------------------------------------
    c_drift, c_orig, c_resc = (med_of(ctrl_rows, "drift_chamfer_over_diag"),
                               med_of(ctrl_rows, "orig_chamfer_over_diag"),
                               med_of(ctrl_rows, "rescored_chamfer_over_diag"))
    j_drift, j_orig, j_resc = (med_of(jug_rows, "drift_chamfer_over_diag"),
                               med_of(jug_rows, "orig_chamfer_over_diag"),
                               med_of(jug_rows, "rescored_chamfer_over_diag"))
    bias = c_resc - c_orig
    label_contrib = (j_orig - j_resc) - bias

    md = ["# Exp 9 — PF++ pseudo-GT label-error bound (Juglet pairwise oracle)\n",
          "Band-constrained ICP refines each TRUE-MATE reference pose; drift is the",
          "label shift and the re-scored chamfer is GARF's error against the refined",
          "reference. All values are symmetry-invariant chamfer / pair diagonal,",
          "medians over true-mate pairs (per-pair values are seed medians).\n",
          "| dataset | n mates | ref drift | orig true-mate chamfer | rescored | delta |",
          "|---|---|---|---|---|---|",
          f"| control (real GT) | {len(ctrl_rows)} | {c_drift:.4f} | {c_orig:.4f} "
          f"| {c_resc:.4f} | {c_resc - c_orig:+.4f} |",
          f"| juglet (PF++ pseudo-GT) | {len(jug_rows)} | {j_drift:.4f} | {j_orig:.4f} "
          f"| {j_resc:.4f} | {j_resc - j_orig:+.4f} |\n",
          f"Metric sampling-noise (identity) floor: control {ctrl_floor:.4f}, "
          f"juglet {jug_floor:.4f} — drift/deltas at or below this are "
          f"indistinguishable from zero.",
          f"Procedure bias (control rescored - orig): {bias:+.4f}",
          f"**Pseudo-GT label-error contribution (bias-corrected): {label_contrib:+.4f} "
          f"of the {j_orig:.4f} Juglet deficit**\n"]

    checks = []
    checks.append(("procedure validated: control drift small (< 0.015)",
                   c_drift == c_drift and c_drift < 0.015))
    checks.append(("procedure validated: control rescored ~ orig (|delta| < 0.008)",
                   bias == bias and abs(bias) < 0.008))
    juglet_labels_fine = (j_resc == j_resc and j_resc > 0.050)
    checks.append(("juglet rescored stays > 0.050 (labels do NOT explain the deficit)",
                   juglet_labels_fine))
    md.append("## Decision\n")
    for label, ok in checks:
        md.append(f"- [{'PASS' if ok else 'FAIL'}] {label}")
    md.append("")
    valid = all(ok for _, ok in checks[:2])
    if not valid:
        md.append("**VERDICT: refinement procedure is biased on the control — do not "
                  "interpret the Juglet bound; fix the band/ICP parameters first.**")
    elif juglet_labels_fine:
        md.append("**VERDICT: PF++ pseudo-GT error does not explain the Juglet deficit — "
                  "the pairwise perception-failure conclusion stands at full strength.**")
    else:
        md.append("**VERDICT: a substantial share of the 0.070 Juglet deficit is PF++ "
                  "label error — replace the pseudo-GT (better reference assembly) "
                  "before further remedy experiments.**")

    for name, rows in (("control", ctrl_rows), ("juglet", jug_rows)):
        md.append(f"\n## Per-pair — {name}\n")
        md.append("| pair | drift | orig | rescored | contact_frac ref->refined | band pts | n |")
        md.append("|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: r["sample"]):
            md.append(
                f"| {r['sample']} | {r['drift_chamfer_over_diag']:.4f} "
                f"| {r['orig_chamfer_over_diag']:.4f} "
                f"| {r['rescored_chamfer_over_diag']:.4f} "
                f"| {r['contact_frac_ref']:.3f} -> {r['contact_frac_refined']:.3f} "
                f"| {r['n_band_src']}/{r['n_band_dst']} | {r['n_runs']} |")

    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    with open(args.out / "pairs.json", "w") as f:
        json.dump({"control": ctrl_rows, "juglet": jug_rows,
                   "identity_floor_control": ctrl_floor,
                   "identity_floor_juglet": jug_floor,
                   "bias": bias, "label_error_contribution": label_contrib}, f, indent=2)
    print("\n".join(md[5:]))
    print(f"\nwrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
