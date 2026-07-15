#!/usr/bin/env python3
"""Exp 10 — look INSIDE GARF's frozen fracture-aware encoder (never done before).

Every prior Juglet experiment measured GARF as a black box via output chamfer and
concluded a pairwise *perception* failure: the encoder extracts no usable mating
signal from worn archaeological rims. But nobody has inspected what the encoder
actually perceives. GARF's frozen feature extractor is a ``FracSeg`` module whose
``coarse_segmenter`` head predicts, per point, P(fracture-surface). That same
backbone feeds the denoiser. So the head is a direct, interpretable readout of
"does GARF recognize this surface as a mating/fracture surface?".

Hypothesis P-B: on fresh control ceramics the segmenter fires cleanly on the true
mating band; on Juglet's worn rims it does not — the pretrained representation
never learned worn breaks look like fractures. If confirmed, the remedy is to
adapt the FrozenExtractor/FracSeg backbone on worn breaks (with geometrically
derived fracture labels), not to keep tuning the denoiser (Exp 5/8 showed those
levers do nothing).

Method (per object, per piece)
------------------------------
1. Pose each piece into the ASSEMBLED frame:
     - control : pieces are stored already assembled (real GT).
     - juglet  : PF++ pseudo-assembly (reuses pair_reference_chamfer.build_pfpp_reference).
2. Sample the surface (area count), feed each piece to the frozen FracSeg exactly
   as the denoiser does (recenter + unit-scale; FracSeg is rotation-augmented so
   no random rotation is applied — deterministic, interpretable).
3. Read per-point P(fracture) from ``coarse_seg_pred``.
4. Label each sampled point TRUE-BAND if, in the assembled pose, it lies within
   ``band_tau_frac * object_scale`` of ANOTHER piece's surface (identical criterion
   to fracture_mesh_ops.erode_fracture_band and derive_pfpp_adjacency).
5. Score how well P(fracture) separates band vs off-band points (ROC AUC, mean
   probabilities, fired fraction). Control validates the probe (segmenter should
   fire on true fractures); Juglet is the test.

For control, also report AUC against the STORED fracture label (shared_faces != -1)
— the segmenter's own training target — as an independent probe validation.

Usage
-----
  python scripts/fracseg_introspection.py \
      --ckpt output/feature_extractor.ckpt \
      --control-source input/Fractura/fractura_real.hdf5 \
      --control-objects ceramics/pink_bowl ceramics/narrow_bottle2 \
                        ceramics/narrow_bottle4 ceramics/blue_pot \
      --juglet-pfpp-dir <.../inference/juglet_deploy/0> \
      --juglet-mesh-dir /data/gpfs/projects/punim2657/Dataset/artifact/Juglet-000 \
      --out logs/diagnostics/exp10_fracseg_<stamp>
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `assembly`

import torch


# --------------------------------------------------------------------------- #
# frozen FracSeg encoder
# --------------------------------------------------------------------------- #
def load_fracseg(ckpt_path: str, device: torch.device):
    """Instantiate the FracSeg feature extractor exactly as the denoiser does and
    load the frozen weights (encoder + batch_norm + coarse_segmenter head)."""
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cfg_dir = str((Path(__file__).resolve().parent.parent / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(config_name="model/frac_seg")
    # frac_seg.yaml has no `# @package`, so its content lands under the `model` key.
    model = instantiate(cfg.model)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Only optimizer/metric buffers may be absent; encoder+head must all load.
    bad = [k for k in missing if k.startswith(("encoder", "batch_norm", "coarse_segmenter"))]
    assert not bad, f"missing critical FracSeg weights: {bad[:5]}"
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def frac_prob(model, pts: np.ndarray, normals: np.ndarray, device: torch.device) -> np.ndarray:
    """Per-point P(fracture) for a single piece, fed like the denoiser input:
    recenter + unit-max-abs scale (weighted.py transform), no random rotation."""
    p = pts.astype(np.float64)
    p = p - p.mean(0)
    s = float(np.max(np.abs(p)))
    p = p / (s if s > 0 else 1.0)
    n = pts.shape[0]
    batch = {
        "pointclouds": torch.tensor(p[None], dtype=torch.float32, device=device),
        "pointclouds_normals": torch.tensor(normals[None], dtype=torch.float32, device=device),
        "points_per_part": torch.tensor([[n]], dtype=torch.int64, device=device),
        "graph": torch.zeros((1, 1, 1), dtype=torch.float32, device=device),
    }
    out = model(batch)
    return out["coarse_seg_pred"].detach().float().cpu().numpy().reshape(-1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney ROC AUC. labels: bool (positive = true-band)."""
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[labels].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def sample_piece(v: np.ndarray, f: np.ndarray, n: int, seed: int):
    m = trimesh.Trimesh(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.int64),
                        process=False)
    pts, fid = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fid]
    return np.asarray(pts, np.float64), np.asarray(nrm, np.float64), np.asarray(fid), m


def point_relief(pts: np.ndarray, normals: np.ndarray, scale: float,
                 radius_frac: float = 0.03) -> np.ndarray:
    """Per-point surface relief = local normal variation within a fixed physical
    radius (radius_frac * piece scale). Same statistic as base.rim_face_weights /
    fracture_sharpness_analysis: elevated on fracture break faces (even worn,
    rounded fillets), low on smooth original vessel walls. Assembly-free."""
    r = radius_frac * scale
    tree = cKDTree(pts)
    relief = np.zeros(len(pts))
    for i, nb in enumerate(tree.query_ball_point(pts, r)):
        if len(nb) < 3:
            continue
        cos = normals[nb] @ normals[i]
        relief[i] = 1.0 - float(np.clip(cos, -1.0, 1.0).mean())
    return relief


def relief_band_label(pts: np.ndarray, relief: np.ndarray, scale: float,
                      relief_pct: float = 85.0, band_frac: float = 0.05) -> np.ndarray:
    """Geometric fracture-band label per point: top-(100-relief_pct)% relief points
    anchor the band; every point within band_frac * scale of an anchor is in the
    band. Mirrors base.rim_face_weights but at point (not face) resolution."""
    if not np.any(relief > 0):
        return np.zeros(len(pts), dtype=bool)
    anchors = pts[relief >= np.percentile(relief, relief_pct)]
    if len(anchors) == 0:
        return np.zeros(len(pts), dtype=bool)
    d, _ = cKDTree(anchors).query(pts)
    return d <= band_frac * scale


def process_object(model, pieces, device, n_per_piece, relief_pct, band_frac, seed,
                   stored_frac=None):
    """pieces: list of (verts, faces). Per piece: sample, run frozen segmenter,
    label each point via geometric relief-band (assembly-free) and optionally via
    the stored fracture GT (control). Returns pooled-over-pieces metrics.

    stored_frac: optional list of per-FACE bool arrays (control training label)."""
    probs, band_all, stored_all, relief_all = [], [], [], []
    for k, (v, f) in enumerate(pieces):
        pts, nrm, fid, m = sample_piece(v, f, n_per_piece, seed + k)
        pscale = float(max(m.extents)) if max(m.extents) > 0 else 1.0
        relief = point_relief(pts, nrm, pscale)
        band = relief_band_label(pts, relief, pscale, relief_pct, band_frac)
        p = frac_prob(model, pts, nrm, device)
        probs.append(p); band_all.append(band); relief_all.append(relief)
        if stored_frac is not None:
            stored_all.append(stored_frac[k][fid])

    prob = np.concatenate(probs)
    band = np.concatenate(band_all)
    relief = np.concatenate(relief_all)
    # rank correlation of P(fracture) with geometric break-relief (Spearman via ranks)
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        ra = ra - ra.mean(); rb = rb - rb.mean()
        d = np.sqrt((ra**2).sum() * (rb**2).sum())
        return float((ra * rb).sum() / d) if d > 0 else float("nan")

    res = {
        "n_points": int(len(prob)),
        "band_frac": float(band.mean()),
        "auc_band": roc_auc(prob, band),
        "spearman_prob_relief": spearman(prob, relief),
        "mean_prob_band": float(prob[band].mean()) if band.any() else float("nan"),
        "mean_prob_offband": float(prob[~band].mean()) if (~band).any() else float("nan"),
        "fired_frac_overall": float((prob > 0.5).mean()),
        "fired_frac_band": float((prob[band] > 0.5).mean()) if band.any() else float("nan"),
        "fired_frac_offband": float((prob[~band] > 0.5).mean()) if (~band).any() else float("nan"),
    }
    if stored_frac is not None:
        stored = np.concatenate(stored_all)
        res["stored_frac_frac"] = float(stored.mean())
        res["auc_stored"] = roc_auc(prob, stored)
        res["mean_prob_stored"] = float(prob[stored].mean()) if stored.any() else float("nan")
        res["mean_prob_nonstored"] = float(prob[~stored].mean()) if (~stored).any() else float("nan")
    return res


# --------------------------------------------------------------------------- #
def load_control_pieces(source: Path, obj: str):
    with h5py.File(source, "r") as f:
        g = f[obj]["pieces"]
        keys = sorted(g.keys(), key=int)
        pieces, stored = [], []
        for k in keys:
            v = np.asarray(g[k]["vertices"][:], np.float64)
            fc = np.asarray(g[k]["faces"][:], np.int64)
            pieces.append((v, fc))
            sf = (np.asarray(g[k]["shared_faces"][:], np.int64) if "shared_faces" in g[k]
                  else np.full(len(fc), -1, np.int64))
            stored.append(sf != -1)   # per-FACE fracture label
    return pieces, stored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="output/feature_extractor.ckpt")
    ap.add_argument("--control-source", type=Path, required=True)
    ap.add_argument("--control-objects", nargs="+", required=True)
    ap.add_argument("--synth-source", type=Path, required=True,
                    help="labeled synthetic hdf5 (valid shared_faces) to validate the probe")
    ap.add_argument("--synth-objects", nargs="+", required=True)
    ap.add_argument("--juglet-mesh-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-per-piece", type=int, default=5000)
    ap.add_argument("--relief-pct", type=float, default=85.0,
                    help="top (100-pct)%% relief points anchor the fracture band")
    ap.add_argument("--band-frac", type=float, default=0.05,
                    help="band radius as fraction of piece scale (matches base.rim_band_frac)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synth-erode-strength", type=float, default=0.0,
                    help="Exp 10b: mollify synthetic fracture bands toward Juglet-like "
                         "wear before scoring (labels preserved). Tests whether erosion "
                         "alone drives the encoder's fracture response toward silence.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_fracseg(args.ckpt, device)
    print("FracSeg frozen extractor loaded (encoder + coarse_segmenter head).")

    results = {"synthetic": {}, "control": {}, "juglet": {}}

    # ---- synthetic (labeled fractures): validates the frozen segmenter + probe ----
    from fracture_mesh_ops import erode_fracture_band
    for obj in args.synth_objects:
        pieces, stored = load_control_pieces(args.synth_source, obj)
        if args.synth_erode_strength > 0.0:
            # Exp 10b: mollify the true fracture bands toward Juglet-like wear
            # (assembled synthetic pieces -> contact band found physically), labels
            # unchanged. Tests whether erosion alone silences the encoder.
            eroded = erode_fracture_band([(v, f) for v, f in pieces],
                                         args.synth_erode_strength)
            pieces = [(ev, f) for (v, f), ev in zip(pieces, eroded)]
        r = process_object(model, pieces, device, args.n_per_piece, args.relief_pct,
                           args.band_frac, args.seed, stored_frac=stored)
        results["synthetic"][obj.replace("/", "_")] = r
        print(f"[synth] {obj}: AUC_stored={r.get('auc_stored', float('nan')):.3f} "
              f"AUC_relief={r['auc_band']:.3f} P(frac)={r.get('mean_prob_stored', float('nan')):.3f} "
              f"P(non)={r.get('mean_prob_nonstored', float('nan')):.3f}")

    # ---- control (fresh real ceramics, unlabeled: relief-band proxy) ----
    for obj in args.control_objects:
        pieces, _ = load_control_pieces(args.control_source, obj)
        r = process_object(model, pieces, device, args.n_per_piece, args.relief_pct,
                           args.band_frac, args.seed)
        results["control"][obj.split("/")[-1]] = r
        print(f"[control] {obj}: AUC_relief={r['auc_band']:.3f} "
              f"P(band)={r['mean_prob_band']:.3f} P(off)={r['mean_prob_offband']:.3f}")

    # ---- juglet (worn archaeological rims; per-piece geometry from source OBJs) ----
    juglet_pieces = []
    for obj in sorted(args.juglet_mesh_dir.glob("*.obj")):
        m = trimesh.load(str(obj), process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        juglet_pieces.append((np.asarray(m.vertices, np.float64), np.asarray(m.faces)))
    print(f"juglet pieces loaded: {len(juglet_pieces)}")
    rj = process_object(model, juglet_pieces, device, args.n_per_piece, args.relief_pct,
                        args.band_frac, args.seed)
    results["juglet"]["Juglet-000"] = rj
    print(f"[juglet] Juglet-000: AUC_relief={rj['auc_band']:.3f} "
          f"P(band)={rj['mean_prob_band']:.3f} P(off)={rj['mean_prob_offband']:.3f}")

    # ---- aggregate + decision ----
    # PRIMARY metric = label-free fracture-response strength (fired fraction).
    # The relief-band AUC proved an unreliable proxy (synthetic arm: P(fracture) is
    # NEGATIVELY correlated with relief, because on these objects relief anchors the
    # rough ORIGINAL surface, not the smooth fresh break). But the synthetic arm's
    # AUC-vs-TRUE-labels validates that the segmenter fires on the correct points,
    # so how OFTEN / how STRONGLY it fires per object is the trustworthy readout.
    synth = list(results["synthetic"].values())
    synth_auc_stored = mean([s["auc_stored"] for s in synth])   # probe validity
    synth_fired = mean([s["fired_frac_overall"] for s in synth])
    ctrl = list(results["control"].values())
    ctrl_fired = mean([c["fired_frac_overall"] for c in ctrl])
    ctrl_auc = mean([c["auc_band"] for c in ctrl])
    jug_fired = rj["fired_frac_overall"]
    jug_auc = rj["auc_band"]
    response_ratio = jug_fired / ctrl_fired if ctrl_fired > 0 else float("nan")

    probe_valid = synth_auc_stored >= 0.65
    # P-B: encoder fires on fresh-ceramic breaks but is (near-)silent on worn Juglet.
    p_b_confirmed = (probe_valid and ctrl_fired >= 0.015
                     and response_ratio <= 0.5)
    detection_intact = probe_valid and response_ratio >= 0.8

    summary = {
        "params": {"n_per_piece": args.n_per_piece, "relief_pct": args.relief_pct,
                   "band_frac": args.band_frac, "seed": args.seed},
        "synth_mean_auc_stored": synth_auc_stored,
        "synth_mean_fired_frac": synth_fired,
        "control_mean_fired_frac": ctrl_fired,
        "control_mean_auc_relief": ctrl_auc,
        "juglet_fired_frac": jug_fired,
        "juglet_auc_relief": jug_auc,
        "response_ratio_juglet_over_control": response_ratio,
        "probe_valid": bool(probe_valid),
        "P_B_confirmed": bool(p_b_confirmed),
        "detection_intact": bool(detection_intact),
        "results": results,
    }
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    def row(name, cls, r, auc_stored="n/a"):
        return (f"| {name} | {cls} | {r['auc_band']:.3f} | {auc_stored} | "
                f"{r['spearman_prob_relief']:+.3f} | {r['mean_prob_band']:.3f} | "
                f"{r['mean_prob_offband']:.3f} | {r['fired_frac_band']:.2f} | "
                f"{r['fired_frac_offband']:.2f} |")

    md = ["# Exp 10 — FracSeg fracture-perception introspection\n",
          "Reads GARF's frozen encoder head P(fracture) per point and asks whether it",
          "fires on each object's break/mating band. Same frozen backbone feeds the denoiser.\n",
          f"Params: {args.n_per_piece} pts/piece; geometric fracture band = top "
          f"{100-args.relief_pct:.0f}% relief anchors dilated by {args.band_frac} x piece scale "
          f"(assembly-free; same detector as Exp 8 rim sampler).\n",
          "Arms: **synthetic** (labeled fractures -> validates the probe via AUC vs stored GT), ",
          "**control** (fresh real ceramics GARF assembles well), **juglet** (worn, fails).\n",
          "## ROC AUC of P(fracture) vs the fracture band\n",
          "| object | class | AUC(relief band) | AUC(stored GT) | Spearman(P,relief) | P(band) | P(off) | fired% band | fired% off |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in results["synthetic"].items():
        md.append(row(name, "synth", r, f"{r.get('auc_stored', float('nan')):.3f}"))
    for name, r in results["control"].items():
        md.append(row(name, "control", r))
    md.append(row("Juglet-000", "JUGLET", rj))
    md += [
        "\n## Aggregate (PRIMARY metric = fracture-response strength, fired%)\n",
        f"- probe validity: synthetic mean AUC vs TRUE fracture GT = **{synth_auc_stored:.3f}** "
        f"(the frozen segmenter fires on the correct points on labeled data).",
        f"- NOTE: relief-band AUC is an unreliable proxy (synthetic P(fracture) anti-correlates "
        f"with relief — relief anchors the rough ORIGINAL surface, not the smooth fresh break), "
        f"so fired% is the trustworthy readout.\n",
        f"- synthetic fresh breaks: mean fired% = **{synth_fired*100:.2f}%**",
        f"- control fresh real ceramics (GARF works): mean fired% = **{ctrl_fired*100:.2f}%**",
        f"- Juglet worn rims (GARF fails): fired% = **{jug_fired*100:.2f}%** "
        f"(**{response_ratio:.2f}x** the control response)\n",
        "## Verdict\n",
    ]
    if not probe_valid:
        md.append(f"**INCONCLUSIVE:** probe did not validate (synth AUC {synth_auc_stored:.3f} "
                  f"< 0.65); P(fracture) cannot be trusted as a fracture readout here.")
    elif p_b_confirmed:
        md.append(f"**P-B CONFIRMED — the encoder is blind to worn breaks:** probe valid "
                  f"(synth AUC {synth_auc_stored:.3f}). GARF's frozen fracture-aware encoder fires "
                  f"on {ctrl_fired*100:.1f}% of points on fresh real ceramics it assembles well, "
                  f"but on only {jug_fired*100:.2f}% of Juglet's worn rims ({response_ratio:.2f}x). "
                  f"The pretrained representation, trained on synthetic FRESH breaks, does not "
                  f"recognize worn/smoothed archaeological fracture surfaces as fractures at all — "
                  f"so the same backbone hands the denoiser features with no mating cue. This is "
                  f"the DIRECT, feature-level cause of the pairwise perception failure that the "
                  f"black-box experiments (Exp 6) inferred, and a different lever than the denoiser "
                  f"knobs / rim oversampling Exp 5/8 ruled out.\n\n"
                  f"**Remedy:** adapt the frozen FracSeg backbone so it recognizes worn breaks — "
                  f"fine-tune the feature extractor on fracture segmentation with "
                  f"`erode_fracture_band` worn-break augmentation (labels preserved from the "
                  f"synthetic breaks), then re-extract features and re-run the pairwise oracle / "
                  f"9-piece assembly. Success test: Juglet fired% rises toward the control's "
                  f"{ctrl_fired*100:.1f}% and pairwise chamfer separates true mates from non-mates.")
    elif detection_intact:
        md.append(f"**P-B REJECTED — detection intact:** the encoder fires on Juglet's worn rims "
                  f"about as often as on control ({jug_fired*100:.2f}% vs {ctrl_fired*100:.2f}%). "
                  f"The deficit is DOWNSTREAM of fracture detection (feature mating "
                  f"complementarity), redirecting the remedy toward the denoiser.")
    else:
        md.append(f"**PARTIAL:** probe valid; Juglet fired% {jug_fired*100:.2f}% is "
                  f"{response_ratio:.2f}x the control {ctrl_fired*100:.2f}% — a real but not "
                  f"order-of-magnitude drop. Worn-rim perception is degraded; inspect per-object "
                  f"rows before committing the remedy.")
    (args.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md[-3:]))
    print(f"\nwrote {args.out}/summary.md and summary.json")


if __name__ == "__main__":
    main()
