"""The 8 physics features, and the local-ground reference they need.

Split out of the enumeration module so the feature definition is readable on its
own — it is the contract every head, every checkpoint's scaling constants, and
every swap test depends on.

Columns, in order (variant A, `expected_vr` dropped after it was measured to be
noise following the ego-velocity frame fix):

    0 range                distance from the radar to the candidate (m)
    1 azimuth              horizontal angle to the candidate (rad)
    2 elevation            vertical angle (rad)
    3 cos_angle            LiDAR incidence cosine -- NATIVE on KITTI-CARLA,
                           ESTIMATED from local normals on VoD (VoD ships
                           intensity in that column, and using it directly was a
                           measured sim2real hazard)
    4 height_above_ground  above the LOCAL ground surface, not the scene minimum
    5 point_density        points in the voxel per cubic metre
    6 path_loss            1/range^4
    7 occlusion_depth      occupied voxels between radar and candidate

Nothing here is dataset-specific: which SOURCE feeds column 3, and the voxel
size that sets column 5's denominator, both arrive via VoxelGridConfig.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("rear.features")

# The only ground reference. A semantic-road variant existed and is REMOVED:
# it made the same feature slot mean different things on different datasets --
# measured VoD mean 5.55 m against KITTI-CARLA 0.89 m -- because it depends on
# each dataset's own road labelling. The geometric estimator removes that
# dependency entirely, which is what a cross-dataset feature has to do.
GROUND_FROM_GEOMETRY = "geometric_percentile"

# same feature slot. Measured consequence: VoD mean 5.55m vs KITTI-CARLA
# 0.89m. The geometric version removes that dependency entirely.
GROUND_PERCENTILE = 5.0
GROUND_RADIUS_M = 5.0          # matches the semantic version's search radius
GROUND_WIDER_RADIUS_M = 15.0   # first fallback for sparse neighbourhoods
GROUND_MIN_POINTS = 8          # below this a percentile is not meaningful
# A 5m horizontal disc over a 178k-point VoD sweep holds ~14,200 points --
# wildly oversampled for a 5th percentile, and it makes the estimator ~1.7s
# per frame. Uniform random subsampling is UNBIASED for percentiles (unlike
# keeping the lowest points per cell, which biases the estimate downward by
# construction), so the cloud is capped. Measured against the full-cloud
# reference at this cap: mean |error| 0.029m, p99 0.52m, on a feature that
# spans several metres. Seeded, so runs are reproducible.
GROUND_MAX_POINTS = 60000
GROUND_SUBSAMPLE_SEED = 0

def compute_local_ground_z_geometric(
    lidar_xyz: np.ndarray,
    centres: np.ndarray,
    radius: float = GROUND_RADIUS_M,
    percentile: float = GROUND_PERCENTILE,
    wider_radius: float = GROUND_WIDER_RADIUS_M,
    min_points: int = GROUND_MIN_POINTS,
    max_points: int | None = GROUND_MAX_POINTS,
    z_bounds: tuple | None = None,
) -> tuple:
    """Local ground height per candidate, from geometry alone.

    Takes the `percentile`-th percentile of z among all LiDAR points within
    `radius` HORIZONTALLY of the candidate. Horizontal-only is the point: a
    3D ball around a candidate 4m up a facade would never reach the road,
    whereas a vertical column always does.

    Why the 5th percentile rather than the minimum: within a 5m column the
    lowest returns are the ground, but the outright minimum is set by
    whatever single noisiest point exists -- multipath under a vehicle, a
    stray low return, sensor noise. p5 sits on the ground surface while
    discarding that tail. Higher percentiles (p25+) start riding up onto
    curbs, parked cars and vegetation in cluttered urban scenes, which is
    exactly the bias this feature must not have.

    Deliberately NOT filtered against "the candidate looks like ground":
    when a candidate IS road surface it will be among the lowest points in
    its own column, and the correct answer is then a height of ~0. Treating
    that as suspicious would push genuine ground candidates to spurious
    negative heights, so no such guard is applied.

    Sparse neighbourhoods fall back the same way the semantic version does:
    widen the radius, then use the frame-level percentile, warning once.

    Returns (ground_z (M,) float64, n_widened int, n_frame_fallback int).
    """
    if len(centres) == 0:
        return np.zeros(0, dtype=np.float64), 0, 0

    xyz = np.asarray(lidar_xyz, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64)
    if z_bounds is not None:
        # Restrict to the SAME z band the voxel grid uses. Points below the
        # grid floor are never voxelised and can never be candidates, but
        # left in they still drag the percentile down and invent height. On
        # KITTI-CARLA 2.43% of returns sit below the floor (down to -6.1m
        # against a -3.0m floor) and removing them moves agreement with the
        # semantic ground truth from Spearman +0.62 to +0.86.
        inside = (xyz[:, 2] >= z_bounds[0]) & (xyz[:, 2] < z_bounds[1])
        if inside.sum() >= min_points:
            xyz = xyz[inside]
    if max_points is not None and len(xyz) > max_points:
        rng = np.random.default_rng(GROUND_SUBSAMPLE_SEED)
        xyz = xyz[rng.choice(len(xyz), size=max_points, replace=False)]
    frame_level = float(np.percentile(xyz[:, 2], percentile))

    # 2D tree: horizontal proximity only, so a vertical column is searched.
    tree = cKDTree(xyz[:, :2])
    ground = np.empty(len(centres), dtype=np.float64)
    n_widened = n_frame = 0

    neighbourhoods = tree.query_ball_point(centres[:, :2], r=radius)
    sparse = []
    for i, idx in enumerate(neighbourhoods):
        if len(idx) >= min_points:
            ground[i] = np.percentile(xyz[idx, 2], percentile)
        else:
            sparse.append(i)

    if sparse:
        wider = tree.query_ball_point(centres[sparse, :2], r=wider_radius)
        for i, idx in zip(sparse, wider):
            if len(idx) >= min_points:
                ground[i] = np.percentile(xyz[idx, 2], percentile)
                n_widened += 1
            else:
                ground[i] = frame_level
                n_frame += 1
        log.warning(
            "Geometric ground estimate: %d candidate(s) needed a wider %.0fm radius, "
            "%d fell back to the frame-level p%.0f.",
            n_widened, wider_radius, n_frame, percentile,
        )

    return ground, n_widened, n_frame


def compute_features(
    centres: np.ndarray,
    mean_cos_angle: np.ndarray,
    point_count: np.ndarray,
    config: VoxelGridConfig,
    local_ground_z: np.ndarray,
    ego_velocity: np.ndarray,
    occ_depth: np.ndarray | None = None,
    voxel_volume: np.ndarray | None = None,
) -> np.ndarray:
    """Step 3: compute the 8-value physics feature vector for each
    candidate, purely from geometry and ego motion -- nothing learned.

    Columns, in order:
      0 range                distance from the radar to the candidate (m)
      1 azimuth               horizontal angle to the candidate (rad)
      2 elevation              vertical angle to the candidate (rad)
      3 cos_angle             mean LiDAR incidence-angle cosine in that voxel
      4 height_above_ground   candidate height above the local road surface (m)
                               -- see compute_local_ground_z_geometric() for how the
                               local ground reference is found
      5 point_density         points in the voxel, per cubic metre
      6 path_loss             1/range^4, the free-space radar path-loss falloff
      7 expected_vr           radial velocity implied by ego motion alone (m/s)
      8 occlusion_depth       occupied voxels between the radar and this
                               candidate. 0 = clear line of sight. LiDAR
                               occlusion is exactly computable and is computed
                               here; whether radar still returns from behind
                               that much material is NOT computable from what
                               this pipeline observes, so it is left to the
                               head rather than hard-coded as a threshold.
    """
    radar_position = np.asarray(config.radar_position, dtype=np.float64)
    to_candidate = centres.astype(np.float64) - radar_position
    dx, dy, dz = to_candidate[:, 0], to_candidate[:, 1], to_candidate[:, 2]

    ranges = np.linalg.norm(to_candidate, axis=1)
    ranges_safe = np.maximum(ranges, config.voxel_size * 0.5)  # avoid a divide-by-zero blowup at r->0

    azimuth = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, np.hypot(dx, dy))
    height_above_ground = centres[:, 2].astype(np.float64) - local_ground_z
    point_density = point_count.astype(np.float64) / (voxel_volume
                                                      if voxel_volume is not None
                                                      else config.voxel_size ** 3)
    path_loss = 1.0 / (ranges_safe ** 4)

    unit_to_candidate = to_candidate / ranges_safe[:, None]
    expected_vr = -np.einsum("j,ij->i", np.asarray(ego_velocity, dtype=np.float64), unit_to_candidate)

    if occ_depth is None:
        occ_depth = np.zeros(len(centres), dtype=np.float64)
    features = np.stack(
        [ranges, azimuth, elevation, mean_cos_angle.astype(np.float64),
         height_above_ground, point_density, path_loss, expected_vr,
         np.asarray(occ_depth, dtype=np.float64)],
        axis=-1,
    )
    return features.astype(np.float32)


