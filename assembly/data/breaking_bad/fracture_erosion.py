"""Worn-break training augmentation for the FracSeg fracture-aware encoder.

Exp 10 (JUGLET_ROOTCAUSE_FINDINGS.md) showed GARF's frozen encoder is blind to
Juglet's worn archaeological breaks — it fires on only 0.57% of points there vs
3.4% on fresh ceramics it assembles well — because it was pretrained on synthetic
FRESH breaks. Exp 10b confirmed the cause is causal: eroding fresh breaks toward
worn drives the encoder's fracture response down (fired% 9.8% -> 2.7%).

This augmentation teaches the encoder that worn/smoothed breaks are still fracture
surfaces: during training it mollifies each object's TRUE contact band (the mating
fracture faces) at a random wear strength, exactly as centuries of abrasion would,
while leaving faces and the ``shared_faces`` fracture labels untouched. Same
resolution-independent physical-radius mollification as
``scripts/fracture_mesh_ops.erode_fracture_band`` (validated across Exp 7/7b),
reimplemented here as a package-local dependency so it runs inside DataLoader
worker processes without a scripts/ import.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def erode_contact_bands(
    meshes: list[trimesh.Trimesh],
    strength: float,
    *,
    band_tau_frac: float = 0.02,
    feather_mult: float = 3.0,
    kernel_frac_max: float = 0.05,
    n_self_samples: int = 8000,
    n_other_samples: int = 8000,
    knn: int = 48,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Mollify the true fracture contact band of each assembled piece.

    ``meshes`` are in the assembled (relative) pose so the mating band is found
    physically: a piece's vertices within ``band_tau_frac`` * object_scale of any
    OTHER piece's surface, feathered out to ``feather_mult`` * tau. Those vertices
    are replaced by a Gaussian-weighted average of surface samples within a fixed
    physical radius ``strength * kernel_frac_max * piece_scale`` — removing fine
    fracture relief at a resolution-independent wear scale. Faces are unchanged, so
    the caller's ``shared_faces`` fracture labels remain valid.

    Returns a list of eroded vertex arrays (one per piece); ``strength <= 0``
    returns the vertices unchanged.
    """
    if strength <= 0 or len(meshes) < 2:
        return [np.asarray(m.vertices, dtype=np.float64) for m in meshes]
    if rng is None:
        rng = np.random.default_rng()

    allv = np.concatenate([m.vertices for m in meshes], axis=0)
    obj_scale = float(np.linalg.norm(allv.max(0) - allv.min(0)))
    if obj_scale <= 0:
        return [np.asarray(m.vertices, dtype=np.float64) for m in meshes]
    tau = band_tau_frac * obj_scale

    samples = []
    for m in meshes:
        pts, _ = trimesh.sample.sample_surface(m, n_other_samples, seed=int(rng.integers(2**31)))
        samples.append(np.asarray(pts, dtype=np.float64))

    out = []
    for i, m in enumerate(meshes):
        verts = np.asarray(m.vertices, dtype=np.float64)
        others = np.concatenate([samples[j] for j in range(len(meshes)) if j != i], axis=0)
        d_other, _ = cKDTree(others).query(verts)
        w = np.clip((feather_mult * tau - d_other) / ((feather_mult - 1.0) * tau), 0.0, 1.0)
        band = w > 0
        if not band.any():
            out.append(verts)
            continue
        piece_scale = float(max(m.extents)) if max(m.extents) > 0 else obj_scale
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
