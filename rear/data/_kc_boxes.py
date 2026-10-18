"""Derive 3D object boxes for KITTI-CARLA from per-point instance labels.

KITTI-CARLA ships no box annotations, but every LiDAR point carries a CARLA
`semantic` tag and an `instance` id. Grouping points by (semantic, instance)
and fitting an oriented box to each group recovers the labels the sim2real
experiment needs on the simulated side.

CLASS COVERAGE -- a structural limit of this dataset, not of the method.
CARLA's palette has tag 10 = Vehicles and tag 4 = Pedestrian. It has **no
bicycle/cyclist tag**: bicycles and their riders fall under Vehicles. So the
KITTI-CARLA label space is {Car, Pedestrian} while VoD's is {Car, Pedestrian,
Cyclist}. Any experiment trained only on KITTI-CARLA therefore cannot detect
Cyclist at all -- a LABEL-SPACE gap, which must not be confused with a domain
gap or with the generator's known VRU under-sampling.

Measured across 50 frames spanning Town01 and Town02:
  Vehicles    7.46 instances/frame (5.70 with >= 20 points), in 100% of frames
  Pedestrian  4.78 instances/frame (1.86 with >= 20 points), in  76% of frames

Box fitting is a minimum-area rectangle in BEV (rotating calipers on the
convex hull) plus the observed z extent. LiDAR sees only the visible faces of
an object, so a box fitted to returns is a lower bound on true extent; the
dimensions are therefore inflated to a per-class floor rather than trusted
outright for sparsely-seen objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

# CARLA semantic tags (CityScapes-derived palette used by CARLA 0.9.x).
CARLA_VEHICLE = 10
CARLA_PEDESTRIAN = 4

# Mapped onto VoD's class names so both domains share a label space.
KC_CLASS_MAP = {CARLA_VEHICLE: "Car", CARLA_PEDESTRIAN: "Pedestrian"}

MIN_POINTS_PER_BOX = 20      # below this an oriented fit is not meaningful
# Per-class minimum extents (l, w, h) in metres. A LiDAR sweep sees only the
# near faces, so a raw fit systematically under-measures; clamping up to a
# plausible floor is closer to the truth than the raw hull for sparse objects.
MIN_EXTENT = {"Car": (3.2, 1.6, 1.4), "Pedestrian": (0.5, 0.5, 1.6)}
MAX_EXTENT = {"Car": (12.0, 3.5, 4.5), "Pedestrian": (1.5, 1.5, 2.2)}


@dataclass
class KCBox:
    """Same field names as VoD's Box3D, so downstream code is domain-agnostic."""
    class_name: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    rotation_z: float
    n_points: int


def min_area_rect(xy: np.ndarray):
    """Minimum-area enclosing rectangle of a 2-D point set.

    Rotating calipers: the minimum-area rectangle must have one side flush
    with a convex-hull edge, so testing every hull edge's orientation is
    exact rather than a search. Returns (cx, cy, length, width, yaw) with
    length >= width and yaw the orientation of the LONG axis.
    """
    if len(xy) < 3:
        c = xy.mean(axis=0)
        return float(c[0]), float(c[1]), 0.1, 0.1, 0.0
    try:
        hull = xy[ConvexHull(xy).vertices]
    except Exception:          # degenerate (collinear) point set
        c = xy.mean(axis=0)
        return float(c[0]), float(c[1]), 0.1, 0.1, 0.0

    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        n = np.hypot(*edge)
        if n < 1e-9:
            continue
        ux, uy = edge / n
        R = np.array([[ux, uy], [-uy, ux]])       # world -> edge-aligned
        proj = hull @ R.T
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        ext = hi - lo
        area = ext[0] * ext[1]
        if best is None or area < best[0]:
            centre = R.T @ ((lo + hi) / 2.0)
            best = (area, centre, ext, np.arctan2(uy, ux))

    _, centre, ext, yaw = best
    if ext[0] >= ext[1]:
        length, width = ext
    else:
        length, width, yaw = ext[1], ext[0], yaw + np.pi / 2
    return (float(centre[0]), float(centre[1]), float(length), float(width),
            float(np.arctan2(np.sin(yaw), np.cos(yaw))))


def boxes_from_instances(points_xyz: np.ndarray, semantic: np.ndarray,
                         instance: np.ndarray,
                         min_points: int = MIN_POINTS_PER_BOX) -> list:
    """Fit one oriented 3-D box per (semantic, instance) group.

    Only the classes in KC_CLASS_MAP are emitted; everything else in the
    CARLA palette is background (road, building, vegetation, ...).
    """
    out: list = []
    for tag, name in KC_CLASS_MAP.items():
        m = semantic == tag
        if not m.any():
            continue
        ids = instance[m]
        pts = points_xyz[m]
        for iid in np.unique(ids):
            p = pts[ids == iid]
            if len(p) < min_points:
                continue
            cx, cy, length, width, yaw = min_area_rect(p[:, :2])
            zlo, zhi = float(p[:, 2].min()), float(p[:, 2].max())
            height = zhi - zlo
            lo, hi = MIN_EXTENT[name], MAX_EXTENT[name]
            length = float(np.clip(length, lo[0], hi[0]))
            width = float(np.clip(width, lo[1], hi[1]))
            height = float(np.clip(height, lo[2], hi[2]))
            out.append(KCBox(name, cx, cy, (zlo + zhi) / 2.0,
                             length, width, height, yaw, int(len(p))))
    return out
