"""Associating real radar detections to LiDAR-derived candidates.

This is the supervision step: it decides which candidates a real radar actually
returned, and therefore what the detection head learns. It only runs on datasets
where `has_real_radar()` is True.

The operating point is per-dataset config (`fitted.association`), never a
literal here. VoD's is a 1.0 m / 5 deg cone -- selected by null-controlled true
signal, not by raw match rate. See `rear.validation.null_control` for why
that distinction is the whole ballgame: a wider tolerance always raises the
observed rate and usually lowers the true signal.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def cone_match(detections_xyz, candidates_xyz, origin,
               range_tolerance_m=1.0, cone_angle_deg=5.0):
    """Per-detection nearest candidate within a range/angle cone about `origin`.

    The cone -- rather than a plain Euclidean ball -- is the radar's own error
    geometry: range resolution is fine and roughly constant, angular resolution
    is coarse and its cross-range extent grows with distance. A fixed-radius
    ball is simultaneously too loose near the sensor and too tight far from it.

    Returns (matched_bool, candidate_index) with -1 where unmatched.
    """
    det = np.asarray(detections_xyz, np.float64)
    cand = np.asarray(candidates_xyz, np.float64)
    if not len(det) or not len(cand):
        return np.zeros(len(det), bool), np.full(len(det), -1, int)
    o = np.asarray(origin, np.float64)
    dr, cr = det - o, cand - o
    d_rng = np.linalg.norm(dr, axis=1)
    c_rng = np.linalg.norm(cr, axis=1)
    # cross-range half-width the cone allows at each detection's range
    max_cross = np.tan(np.radians(cone_angle_deg)) * d_rng
    search = np.sqrt(range_tolerance_m ** 2 + max_cross ** 2)

    tree = cKDTree(cand)
    ok = np.zeros(len(det), bool)
    idx = np.full(len(det), -1, int)
    for i, (p, r) in enumerate(zip(det, search)):
        near = tree.query_ball_point(p, r)
        if not near:
            continue
        near = np.asarray(near)
        d_along = np.abs(c_rng[near] - d_rng[i])
        cosang = np.clip((cr[near] @ dr[i]) / (c_rng[near] * d_rng[i] + 1e-12), -1, 1)
        ang = np.degrees(np.arccos(cosang))
        good = (d_along <= range_tolerance_m) & (ang <= cone_angle_deg)
        if good.any():
            sel = near[good]
            ok[i] = True
            idx[i] = int(sel[np.argmin(np.linalg.norm(cand[sel] - p, axis=1))])
    return ok, idx


def matcher_for(adapter, candidates_xyz, origin):
    """A `match_fn(points) -> bool array` bound to this dataset's fitted
    operating point, ready to hand to `null_control.true_signal`."""
    tol = adapter.fitted("association", "range_tolerance_m", default=1.0)
    ang = adapter.fitted("association", "cone_angle_deg", default=5.0)

    def match_fn(points):
        return cone_match(points, candidates_xyz, origin, tol, ang)[0]
    return match_fn
