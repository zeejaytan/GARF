#!/usr/bin/env python3
"""Shared mesh ops for Juglet root-cause confirmation (erode) and remedy (densify).

GARF's Juglet failure is a pairwise *perception* failure on worn archaeological
fracture rims. These helpers:
  - erode_fracture_rim: round sharp fracture creases toward Juglet-like relief
  - densify_fracture_rim: subdivide the rim band to upweight area-based sampling
"""

from __future__ import annotations

import numpy as np
import trimesh


def vertex_max_dihedral(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex max adjacent-face dihedral angle in degrees."""
    sharp = np.zeros(len(mesh.vertices))
    if len(mesh.face_adjacency) == 0:
        return sharp
    edges, angles = mesh.face_adjacency_edges, np.degrees(np.abs(mesh.face_adjacency_angles))
    np.maximum.at(sharp, edges[:, 0], angles)
    np.maximum.at(sharp, edges[:, 1], angles)
    return sharp


def piece_relief_stats(verts: np.ndarray, faces: np.ndarray, radius_frac: float = 0.03, n_pts: int = 4000) -> dict:
    """Scale-normalized surface relief (matches fracture_sharpness_analysis.py)."""
    from scipy.spatial import cKDTree

    m = trimesh.Trimesh(vertices=verts.astype(np.float64), faces=faces.astype(np.int64), process=False)
    scale = float(max(m.extents))
    if scale <= 0 or len(m.faces) < 8:
        return {"scale": scale, "relief_p90": 0.0, "relief_mean": 0.0, "sharp_frac": 0.0}
    pts, fid = trimesh.sample.sample_surface(m, n_pts)
    fn = m.face_normals[fid]
    r = radius_frac * scale
    tree = cKDTree(pts)
    relief = np.zeros(len(pts))
    for i, ne in enumerate(tree.query_ball_point(pts, r)):
        if len(ne) < 3:
            continue
        cos = fn[ne] @ fn[i]
        relief[i] = 1.0 - float(np.clip(cos, -1, 1).mean())
    return {
        "scale": scale,
        "relief_p90": float(np.percentile(relief, 90)),
        "relief_mean": float(np.mean(relief)),
        "sharp_frac": float(np.mean(relief > 0.15)),
    }


def erode_fracture_rim(
    verts: np.ndarray,
    faces: np.ndarray,
    strength: float,
    *,
    dihedral_pct: float = 55.0,
    max_iters: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Round sharp fracture creases (archaeological wear simulation).

    ``strength`` in [0, 1]: fraction of relief to remove toward a Taubin-smoothed
    target on vertices whose max dihedral exceeds the ``dihedral_pct`` percentile.
    GT assembled pose is preserved (in-place vertex edit only).
    """
    if strength <= 0:
        return verts, faces
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy(), process=False)
    sharp = vertex_max_dihedral(mesh)
    thr = float(np.percentile(sharp[sharp > 0], dihedral_pct)) if np.any(sharp > 0) else 90.0
    mask = sharp >= thr
    if not mask.any():
        mask = sharp >= np.percentile(sharp, 75)

    smooth = mesh.copy()
    iters = max(1, min(max_iters, int(round(4 + strength * 36))))
    try:
        trimesh.smoothing.filter_taubin(smooth, lamb=0.5, nu=0.53, iterations=iters)
    except Exception:
        return verts, faces

    out = mesh.vertices.copy()
    w = np.clip(strength, 0.0, 1.0)
    out[mask] = (1.0 - w) * mesh.vertices[mask] + w * smooth.vertices[mask]
    return out.astype(np.float64), faces


def erode_fracture_band(
    pieces: list[tuple[np.ndarray, np.ndarray]],
    strength: float,
    *,
    band_tau_frac: float = 0.02,
    feather_mult: float = 3.0,
    kernel_frac_max: float = 0.05,
    n_self_samples: int = 20000,
    n_other_samples: int = 20000,
    knn: int = 48,
    seed: int = 0,
) -> list[np.ndarray]:
    """Archaeological-wear simulation on the TRUE fracture band of an assembled object.

    ``pieces`` are (verts, faces) in the ASSEMBLED (GT) pose. The fracture
    surface of each piece is identified physically: vertices within
    ``band_tau_frac`` * object_scale of any OTHER piece's surface (the mating
    contact band), feathered out to ``feather_mult`` * tau. Those vertices are
    mollified — replaced by a Gaussian-weighted average of surface samples
    within a FIXED PHYSICAL radius ``strength * kernel_frac_max * piece_scale``
    — which removes fine fracture relief and rounds rim creases at a
    resolution-independent physical wear scale, exactly what centuries of
    abrasion do. In-place vertex edit only: faces and GT pose are preserved,
    so benchmark metrics stay valid.

    Returns the list of eroded vertex arrays (faces unchanged).

    Rationale: Taubin/Laplacian smoothing diffuses at a mesh-resolution-
    dependent rate (~edge_len * sqrt(iters)), which is why it barely moves the
    relief_p90 statistic on ~1e6-face Fractura scans. Mollification over a
    physical radius is resolution-independent by construction.
    """
    if strength <= 0:
        return [np.asarray(v, dtype=np.float64) for v, _ in pieces]
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    meshes = [
        trimesh.Trimesh(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.int64), process=False)
        for v, f in pieces
    ]
    allv = np.concatenate([m.vertices for m in meshes], axis=0)
    obj_scale = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    tau = band_tau_frac * obj_scale

    # Surface samples per piece (fixed count -> resolution-independent kernels).
    samples = []
    for m in meshes:
        pts, _ = trimesh.sample.sample_surface(m, n_other_samples, seed=int(rng.integers(2**31)))
        samples.append(np.asarray(pts, dtype=np.float64))

    out = []
    for i, m in enumerate(meshes):
        verts = np.asarray(m.vertices, dtype=np.float64)
        others = np.concatenate([samples[j] for j in range(len(meshes)) if j != i], axis=0)
        d_other, _ = cKDTree(others).query(verts)
        # Fracture-band weight: 1 inside tau, feathered to 0 at feather_mult*tau.
        w = np.clip((feather_mult * tau - d_other) / ((feather_mult - 1.0) * tau), 0.0, 1.0)
        band = w > 0
        if not band.any():
            out.append(verts)
            continue
        piece_scale = float(max(m.extents))
        r = strength * kernel_frac_max * piece_scale
        self_pts, _ = trimesh.sample.sample_surface(m, n_self_samples, seed=int(rng.integers(2**31)))
        self_pts = np.asarray(self_pts, dtype=np.float64)
        dist, idx = cKDTree(self_pts).query(verts[band], k=knn, distance_upper_bound=r)
        valid = np.isfinite(dist)
        sigma = 0.5 * r
        gw = np.where(valid, np.exp(-0.5 * (dist / sigma) ** 2), 0.0)
        gw_sum = gw.sum(axis=1, keepdims=True)
        ok = gw_sum[:, 0] > 1e-12
        idx_safe = np.where(valid, idx, 0)
        target = np.einsum("nk,nkd->nd", gw, self_pts[idx_safe]) / np.maximum(gw_sum, 1e-12)
        new_band = verts[band].copy()
        blend = (strength * w[band])[:, None]
        new_band[ok] = (1.0 - blend[ok]) * verts[band][ok] + blend[ok] * target[ok]
        new_v = verts.copy()
        new_v[band] = new_band
        out.append(new_v)
    return out


def sharpen_fracture_band_solo(
    verts: np.ndarray,
    faces: np.ndarray,
    strength: float,
    *,
    relief_pct: float = 85.0,
    band_frac: float = 0.05,
    kernel_frac: float = 0.03,
    n_probe: int = 20000,
    n_self_samples: int = 20000,
    knn: int = 48,
    seed: int = 0,
    region: str = "band",
) -> np.ndarray:
    """De-weathering (Exp 14): inverse of ``erode_fracture_band``, pose-free.

    Surface unsharp mask on the detected fracture band: every band vertex is
    displaced AWAY from its Gaussian-mollified target,

        v' = v + strength * w * (v - mollify(v)),

    amplifying the residual micro-relief that abrasion attenuated — the exact
    inverse of the mollifier wear model validated causally in Exp 7/7b/10b.
    True-mate complementarity is preserved to first order: both mating faces
    carry (mirrored) copies of the same underlying fracture surface, and the
    transform is a deterministic local functional of that surface, so matching
    bumps/dents amplify coherently.

    The band is detected per piece with the validated relief-band detector
    (same statistic and default params as fracseg_introspection / the Exp 8
    rim sampler) — no assembled pose is needed. PF++ poses give a zero contact
    band on Juglet (Exp 9), so physical band detection is unusable there; the
    pose-free detector also makes this transform deployable at inference time
    on any scan.

    ``region="band"`` sharpens the fracture band (the treatment);
    ``region="offband"`` sharpens everything BUT the band (Exp 14 arm C
    specificity control: amplifying the original vessel surface must NOT
    restore mating). Displacements are clamped to the mollify kernel radius.
    Returns the new vertex array (faces unchanged; pose preserved).
    """
    if strength <= 0:
        return np.asarray(verts, dtype=np.float64)
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    mesh = trimesh.Trimesh(vertices=np.asarray(verts, np.float64),
                           faces=np.asarray(faces, np.int64), process=False)
    scale = float(max(mesh.extents))
    if scale <= 0 or len(mesh.faces) < 8:
        return np.asarray(verts, dtype=np.float64)

    # --- relief-band detection (pose-free, per piece) ---
    pts, fid = trimesh.sample.sample_surface(mesh, n_probe, seed=int(rng.integers(2**31)))
    pts = np.asarray(pts, np.float64)
    fn = mesh.face_normals[fid]
    r_relief = 0.03 * scale
    tree = cKDTree(pts)
    relief = np.zeros(len(pts))
    for i, nb in enumerate(tree.query_ball_point(pts, r_relief)):
        if len(nb) < 3:
            continue
        cos = fn[nb] @ fn[i]
        relief[i] = 1.0 - float(np.clip(cos, -1.0, 1.0).mean())
    if not np.any(relief > 0):
        return np.asarray(verts, dtype=np.float64)
    anchors = pts[relief >= np.percentile(relief, relief_pct)]
    if len(anchors) == 0:
        return np.asarray(verts, dtype=np.float64)

    v = np.asarray(mesh.vertices, np.float64)
    d_anchor, _ = cKDTree(anchors).query(v)
    R = band_frac * scale
    # Feathered band weight: 1 inside 0.6R, linear to 0 at R.
    w = np.clip((R - d_anchor) / (0.4 * R), 0.0, 1.0)
    if region == "offband":
        w = 1.0 - w
    elif region != "band":
        raise ValueError(f"region must be 'band' or 'offband', got {region!r}")
    active = w > 0
    if not active.any():
        return v

    # --- mollified target (same kernel construction as erode_fracture_band) ---
    r = kernel_frac * scale
    self_pts, _ = trimesh.sample.sample_surface(mesh, n_self_samples,
                                                seed=int(rng.integers(2**31)))
    self_pts = np.asarray(self_pts, np.float64)
    dist, idx = cKDTree(self_pts).query(v[active], k=knn, distance_upper_bound=r)
    valid = np.isfinite(dist)
    sigma = 0.5 * r
    gw = np.where(valid, np.exp(-0.5 * (dist / sigma) ** 2), 0.0)
    gw_sum = gw.sum(axis=1, keepdims=True)
    ok = gw_sum[:, 0] > 1e-12
    idx_safe = np.where(valid, idx, 0)
    target = np.einsum("nk,nkd->nd", gw, self_pts[idx_safe]) / np.maximum(gw_sum, 1e-12)

    disp = strength * w[active, None] * (v[active] - target)
    disp[~ok] = 0.0
    # Clamp: a de-weathered bump should not exceed the wear scale itself.
    norm = np.linalg.norm(disp, axis=1, keepdims=True)
    over = norm[:, 0] > r
    disp[over] *= r / norm[over]
    out = v.copy()
    out[active] = v[active] + disp
    return out


def _rim_face_mask(mesh: trimesh.Trimesh, dihedral_pct: float = 50.0) -> np.ndarray:
    """Faces touching at least one high-dihedral edge (fracture rim band)."""
    if len(mesh.face_adjacency) == 0:
        return np.zeros(len(mesh.faces), dtype=bool)
    angles = np.degrees(np.abs(mesh.face_adjacency_angles))
    thr = float(np.percentile(angles, dihedral_pct))
    mask = np.zeros(len(mesh.faces), dtype=bool)
    mask[mesh.face_adjacency[angles >= thr].ravel()] = True
    return mask


def densify_fracture_rim(
    verts: np.ndarray,
    faces: np.ndarray,
    strength: float,
    *,
    dihedral_pct: float = 50.0,
    max_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide faces in the fracture-rim band (visualization / resolution aid).

    WARNING: this does NOT upweight GARF's point sampling. The dataloader
    (assembly/data/breaking_bad/weighted.py) uses trimesh.sample.sample_surface,
    which is area-weighted — midpoint subdivision preserves total area, so the
    sampled point distribution is unchanged. The actual rim-oversampling remedy
    lives in the sampler (data.rim_oversample_frac), not in mesh edits.

    ``strength`` in [0, 1] maps to 0..max_passes midpoint subdivisions of rim faces.
    """
    if strength <= 0:
        return verts, faces
    mesh = trimesh.Trimesh(vertices=verts.copy(), faces=faces.copy(), process=False)
    passes = max(1, min(max_passes, int(round(1 + strength * (max_passes - 1)))))
    for _ in range(passes):
        fmask = _rim_face_mask(mesh, dihedral_pct=dihedral_pct)
        if not fmask.any():
            break
        try:
            mesh = mesh.subdivide(fmask)
        except Exception:
            break
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)
