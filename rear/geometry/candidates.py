"""Candidate enumeration: voxelise -> cull occlusion -> FOV mask.

The structural mechanism behind two of the project's non-negotiables (CLAUDE.md):
occluded voxels are never enumerated, and v_r is computed rather than predicted.
Nothing here is learned.

DATASET-AGNOSTIC BY CONSTRUCTION. Every knob is a `VoxelGridConfig` field and
every field is populated from `configs/datasets/<name>.yaml` via
`DatasetAdapter.grid_config()`. The dataclass defaults are deliberately generic
placeholders, NOT any dataset's fitted values -- VoD's 0.4 m voxel, 60 deg FOV,
(-20,100)x(-90,90)x(-7,5) volume and normal-estimated cosine all live in
`configs/datasets/vod.yaml`. Before this refactor they were scattered across a
module constant, a dataclass default and a `VOD_SIZED_CONFIG` literal inside an
analysis script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

from rear.geometry.features import (  # noqa: F401  (re-exported)
    GROUND_FROM_GEOMETRY,
    compute_features,
    compute_local_ground_z_geometric)

log = logging.getLogger("candidate_enumeration")

MIN_EXPECTED_CANDIDATES = 500
MAX_EXPECTED_CANDIDATES = 8000

# CARLA's semantic segmentation tag for "Road" (confirmed present in this
# dataset's per-point `semantic` field -- see CHECKPOINT.md).

# --- How feature 3 ("cos_angle") is obtained -----------------------------
# Datasets do not agree on what column 3 of the LiDAR array holds:
#   KITTI-CARLA: cos_angle_lidar_surface -- a true incidence-angle cosine,
#                measured range [0.017, 1.0].
#   VoD:         reflectance/intensity   -- measured range [1.8, 255.0].
#                VoD has no incidence-angle field at all.
# Putting both into the same feature slot conflates two different physical
# quantities and is a live sim2real hazard (it silently made "cos_angle" the
# single strongest predictor of a model trained on VoD). COS_ANGLE_FROM_COLUMN
# takes column 3 as-is (correct for KITTI-CARLA); COS_ANGLE_FROM_NORMALS
# estimates a true incidence cosine from local surface geometry (needed for
# VoD, and for any dataset lacking the field).
# --- Radar field of view --------------------------------------------------
# MEASURED, not assumed. Over the RCS head's VoD training candidates the
# azimuth p1/p99 are -53.3 deg / +62.6 deg, and 98.3% lie within |az| < 60 --
# VoD's ZF FRGen21 is forward-looking. KITTI-CARLA's LiDAR is full-surround,
# so only 38.5% of its candidates fall in that sector; the other 61.5% sit at
# angles no VoD-trained model has any support for, and both heads extrapolate
# there (measured: clamping azimuth alone pulled the RCS head's KITTI-CARLA
# mean from +19.53 to +0.61 dBsm, recovering essentially the whole shift).
# Generic placeholder only. The real value is per dataset:
# configs/datasets/<name>.yaml -> sensors.radar_fov_degrees
DEFAULT_RADAR_FOV_DEG = 60.0


COS_ANGLE_FROM_COLUMN = "column3"
COS_ANGLE_FROM_NORMALS = "estimate_normals"

# Neighbourhood used for the local plane fit.
NORMAL_ESTIMATION_RADIUS_M = 0.6
NORMAL_ESTIMATION_K = 10
MIN_POINTS_FOR_PLANE_FIT = 3   # a plane needs 3 non-collinear points


@dataclass
class VoxelGridConfig:
    """Every tunable knob for candidate enumeration, in one place so
    nothing is hardcoded inside the functions below."""

    voxel_size: float = 0.4
    x_range: Tuple[float, float] = (-50.0, 50.0)
    y_range: Tuple[float, float] = (-50.0, 50.0)
    z_range: Tuple[float, float] = (-3.0, 5.0)
    radar_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # One voxel size everywhere. Kept as a named field rather than assumed so
    # a config that requests anything else fails loudly instead of silently
    # getting the uniform lattice: a range-adaptive variant was trialled and
    # REJECTED, and its code is removed.
    voxel_size_schedule: str = "fixed"

    # --- occlusion (Part B) ------------------------------------------------
    # Resolution of the occupancy grid the occlusion ray march reads. Kept
    # SEPARATE from voxel_size so that changing candidate resolution does not
    # silently change what counts as an occluder -- otherwise Part A and Part B
    # could not be attributed apart. None means "same as voxel_size", which
    # reproduces the pre-existing behaviour exactly.
    occlusion_voxel_size: float | None = None
    # Whether a voxel sitting at local ground level may act as an occluder.
    # False applies the Part B fix: the radar is 1.15 m BELOW the roof LiDAR,
    # so rays to near, low targets run nearly parallel to the road and clip the
    # continuous sheet of road-surface voxels they never physically pass
    # through. A cell at ground level cannot occlude a target above it from a
    # sensor above it. This is a geometric statement, not a loosened tolerance.
    ground_voxels_occlude: bool = True
    # A cell counts as ground when its centre sits within this margin of the
    # LOCAL ground height (the same estimate height_above_ground uses).
    # HALF a voxel is the geometric definition -- that is exactly the set of
    # cells whose z-extent CONTAINS the ground surface. A full voxel (0.4 m)
    # additionally excludes the layer sitting entirely above the road, which is
    # a tolerance loosening rather than a geometric statement; both were
    # measured (see CHECKPOINT.md) and both fail the true-signal gate.
    ground_occluder_margin_m: float = 0.2
    # How many CONSECUTIVE occupied voxels a ray must pass through before
    # the candidate behind them counts as occluded.
    #
    # REVERTED TO 1 (original single-hit behaviour) on 29/07/26. A value of
    # 2 was trialled to fix apparent road-grazing over-culling and did raise
    # box-conditioned association 77.0% -> 85.4%, but a null control showed
    # the gain was coincidental: it tripled the candidate pool (793 -> 2548
    # per frame), and true signal over an azimuth-jittered null FELL from
    # 43.7pp to 37.7pp. The mechanism is kept available and configurable,
    # but the default stays at the measured-best setting. See CHECKPOINT.md,
    # "PHASES 1-3 ATTEMPTED AND REVERTED".
    min_consecutive_blocking_voxels: int = 1
    # Retain candidates whose ray passes through at most this many occupied
    # voxels, instead of discarding everything with a single blocker. The
    # cutoff BOUNDS COMPUTE; it is not a physical claim. Chosen at the
    # saturation point of object coverage: measured on 456 VoD test objects,
    # the fraction with at least one retained voxel goes 83.3% (depth 0) ->
    # 90.8% (<=1) -> 97.8% (<=2) -> 99.6% (<=3), and <=6 is still 99.6%. So 3
    # captures essentially every recoverable object; beyond it candidates keep
    # growing (2.6x -> 5.1x per frame) for no coverage gain.
    # Set to 0 to reproduce the old binary-cull behaviour exactly.
    max_occlusion_depth: int = 3
    # Where feature 3 comes from. Default suits KITTI-CARLA, which supplies a
    # real cosine; VoD callers must pass COS_ANGLE_FROM_NORMALS. See the
    # constants above for why this is not a cosmetic scaling choice.
    cos_angle_source: str = COS_ANGLE_FROM_COLUMN
    normal_radius: float = NORMAL_ESTIMATION_RADIUS_M
    normal_k_neighbors: int = NORMAL_ESTIMATION_K
    # Where height_above_ground's reference comes from. The semantic version
    # needs a road class and silently degrades to scene-minimum-z without
    # one; the geometric version needs only xyz and so computes the SAME
    # quantity on every dataset. See the ground-estimation section above.
    ground_z_source: str = GROUND_FROM_GEOMETRY
    # Restrict candidates to the modelled radar's azimuth coverage. ON by
    # default, and treated exactly like occlusion: an out-of-FOV candidate is
    # REMOVED from enumeration, not enumerated and then predicted unlikely.
    #
    # Turning this off does not merely make results "less accurate" -- it
    # scores candidates outside any modelled sensor's physical coverage,
    # where the heads have no training support and their outputs are
    # meaningless by construction. Off is for diagnostics only.
    apply_fov_mask: bool = True
    radar_fov_deg: float = DEFAULT_RADAR_FOV_DEG


@dataclass
class CandidateSet:
    """The final, ready-to-use output of enumerate_candidates(). Only
    non-occluded candidates are included, so `valid_mask` is True for
    every row by construction -- it's kept on the dataclass so callers
    that concatenate/filter CandidateSets downstream have a consistent
    field to check rather than assuming truthiness implicitly."""

    centres: np.ndarray    # (M, 3) float32 -- x, y, z of each candidate, metres
    features: np.ndarray   # (M, 8) float32 -- see compute_features() for column meanings
    semantic: np.ndarray   # (M,) int64 -- dominant semantic class per candidate
    valid_mask: np.ndarray  # (M,) bool -- always True; kept for API consistency
    # Mean of LiDAR column 3 per voxel, kept only when that column is NOT the
    # cos_angle -- i.e. VoD's reflectance/intensity. None for KITTI-CARLA,
    # whose column 3 IS the cosine and which has no intensity field at all.
    #
    # Deliberately NOT appended to `features`: KITTI-CARLA supplies no
    # intensity whatsoever, so a shared 9th slot would have to be filled with
    # a placeholder there -- recreating in reverse the exact conflation this
    # field exists to fix. Kept accessible so a VoD-only experiment can
    # concatenate it explicitly, which is a decision the caller should make
    # knowingly.
    mean_intensity: np.ndarray | None = None
    n_normal_fit_failures: int = 0   # voxels too sparse to plane-fit


def _voxel_grid_shape(config: VoxelGridConfig) -> Tuple[int, int, int]:
    """Work out how many voxels fit along each axis given the configured
    ranges and voxel size, e.g. a 100m range at 0.4m voxels gives 250."""
    nx = int(round((config.x_range[1] - config.x_range[0]) / config.voxel_size))
    ny = int(round((config.y_range[1] - config.y_range[0]) / config.voxel_size))
    nz = int(round((config.z_range[1] - config.z_range[0]) / config.voxel_size))
    return nx, ny, nz


SCHEDULE_FIXED = "fixed"


def _grid_shape_for(config: VoxelGridConfig, size: float) -> Tuple[int, int, int]:
    return (max(int(round((config.x_range[1] - config.x_range[0]) / size)), 1),
            max(int(round((config.y_range[1] - config.y_range[0]) / size)), 1),
            max(int(round((config.z_range[1] - config.z_range[0]) / size)), 1))


def occupancy_cells(lidar: np.ndarray, config: VoxelGridConfig):
    """Occupied cells of the OCCLUSION grid, straight from the LiDAR points.

    The occlusion grid is deliberately built from the raw points at
    `occlusion_voxel_size` rather than from the candidate voxels. Under a
    range-adaptive schedule the candidate lattice is no longer uniform, so it
    cannot index a dense occupancy array at all; more importantly, tying the
    two together would mean refining candidates also refined the occluders,
    and Part A and Part B could no longer be told apart.

    When the schedule is fixed and `occlusion_voxel_size` is None this returns
    exactly the same cell set as the candidate voxels -- both are the unique
    cells of the same in-bounds points on the same lattice. That equivalence
    is asserted by tests/verify_voxel_schedule.py rather than assumed.

    Returns (ijk (K,3) int64, shape (3,), size float).
    """
    size = config.occlusion_voxel_size or config.voxel_size
    nx, ny, nz = _grid_shape_for(config, size)
    x, y, z = lidar[:, 0], lidar[:, 1], lidar[:, 2]
    in_bounds = (
        (x >= config.x_range[0]) & (x < config.x_range[1])
        & (y >= config.y_range[0]) & (y < config.y_range[1])
        & (z >= config.z_range[0]) & (z < config.z_range[1])
    )
    ix = np.clip(np.floor((x[in_bounds] - config.x_range[0]) / size).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor((y[in_bounds] - config.y_range[0]) / size).astype(np.int64), 0, ny - 1)
    iz = np.clip(np.floor((z[in_bounds] - config.z_range[0]) / size).astype(np.int64), 0, nz - 1)
    flat = np.unique((ix * ny + iy) * nz + iz)
    return np.stack(np.unravel_index(flat, (nx, ny, nz)), axis=-1), (nx, ny, nz), size


def ground_occluder_mask(cell_ijk: np.ndarray, shape, size: float,
                         lidar: np.ndarray, config: VoxelGridConfig) -> np.ndarray:
    """True for occupancy cells that sit at local ground level (Part B).

    Uses the SAME local ground estimate that `height_above_ground` already
    computes -- the 5th percentile of z in a horizontal column around the cell
    -- so this introduces no new notion of "ground" into the pipeline.

    Limitation, stated rather than buried: the estimate follows terrain, so on
    a rising road the crest is also "local ground" and would stop occluding
    even though it genuinely can. VoD is flat, and the p5-in-a-column estimate
    was validated there (scripts/validate_ground_estimator.py); on markedly
    non-flat terrain this exclusion would need a slope test as well.
    """
    centres = np.stack([
        config.x_range[0] + (cell_ijk[:, 0] + 0.5) * size,
        config.y_range[0] + (cell_ijk[:, 1] + 0.5) * size,
        config.z_range[0] + (cell_ijk[:, 2] + 0.5) * size,
    ], axis=-1)
    ground_z, _, _ = compute_local_ground_z_geometric(
        lidar[:, :3], centres, z_bounds=config.z_range)
    return centres[:, 2] <= ground_z + config.ground_occluder_margin_m


def voxelise(
    lidar: np.ndarray,
    semantic: np.ndarray,
    config: VoxelGridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 1: group LiDAR points into a 3D grid of voxels and summarise
    each occupied voxel.

    Every LiDAR point falls into exactly one voxel cell based on its
    (x, y, z) position. Any voxel that contains at least one point
    becomes a candidate. For each such voxel we compute: its centre in
    metres, the mean cos_angle of the points inside it, the most common
    semantic label among those points (its "dominant class"), and how
    many points landed in it.

    Points outside x_range/y_range/z_range are dropped first, since
    they're outside the volume the radar is being modelled over.

    Under a range-adaptive schedule (config.voxel_size_schedule) the grid is
    no longer one lattice: points are split into range shells and each shell
    is voxelised at its own size, then the results are concatenated. Points
    are assigned to a shell by their OWN range from the radar, so a surface
    straddling a shell edge can contribute one candidate on each side of it;
    that is a boundary artefact of any multi-resolution grid and is bounded by
    the number of shells.

    Returns (centres (M,3) float32, mean_cos_angle (M,) float32,
    dominant_semantic (M,) int64, point_count (M,) int64, voxel_ijk
    (M,3) int64 -- the integer grid index of each occupied voxel, needed
    later for occlusion culling. Under a schedule these indices are given on
    """
    if config.voxel_size_schedule != SCHEDULE_FIXED:
        raise ValueError(
            f"voxel_size_schedule must be {SCHEDULE_FIXED!r}, got "
            f"{config.voxel_size_schedule!r}. Range-adaptive voxel sizing was "
            f"trialled and rejected, and the material pipeline additionally "
            f"needs a fixed lattice to reconstruct the point->candidate map "
            f"(rear/materials/pipeline.py).")
    return _voxelise_uniform(lidar, semantic, config, config.voxel_size)


def _voxelise_uniform(
    lidar: np.ndarray,
    semantic: np.ndarray,
    config: VoxelGridConfig,
    size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One uniform lattice at `size`. This is the original voxelise body."""
    x, y, z, cos_angle = lidar[:, 0], lidar[:, 1], lidar[:, 2], lidar[:, 3]

    in_bounds = (
        (x >= config.x_range[0]) & (x < config.x_range[1])
        & (y >= config.y_range[0]) & (y < config.y_range[1])
        & (z >= config.z_range[0]) & (z < config.z_range[1])
    )
    x, y, z, cos_angle = x[in_bounds], y[in_bounds], z[in_bounds], cos_angle[in_bounds]
    semantic = semantic[in_bounds]
    if len(x) == 0:
        e = np.zeros(0)
        return (np.zeros((0, 3), np.float32), e.astype(np.float32),
                e.astype(np.int64), e.astype(np.int64), np.zeros((0, 3), np.int64), e)

    nx, ny, nz = _grid_shape_for(config, size)
    ix = np.floor((x - config.x_range[0]) / size).astype(np.int64)
    iy = np.floor((y - config.y_range[0]) / size).astype(np.int64)
    iz = np.floor((z - config.z_range[0]) / size).astype(np.int64)
    # Points exactly on the upper boundary of a range can floor to n
    # (one past the last valid index); clip them into the last voxel.
    ix, iy, iz = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1), np.clip(iz, 0, nz - 1)

    flat_idx = (ix * ny + iy) * nz + iz
    unique_flat, inverse, counts = np.unique(flat_idx, return_inverse=True, return_counts=True)
    m = len(unique_flat)

    mean_cos_angle = (np.bincount(inverse, weights=cos_angle, minlength=m) / counts).astype(np.float32)

    voxel_ijk = np.stack(np.unravel_index(unique_flat, (nx, ny, nz)), axis=-1)
    centres = np.stack(
        [
            config.x_range[0] + (voxel_ijk[:, 0] + 0.5) * size,
            config.y_range[0] + (voxel_ijk[:, 1] + 0.5) * size,
            config.z_range[0] + (voxel_ijk[:, 2] + 0.5) * size,
        ],
        axis=-1,
    ).astype(np.float32)

    dominant_semantic = _dominant_per_group(flat_idx, semantic, unique_flat)

    return (centres, mean_cos_angle, dominant_semantic, counts.astype(np.int64),
            voxel_ijk, np.full(len(centres), size, np.float64))


def _dominant_per_group(flat_idx: np.ndarray, semantic: np.ndarray, unique_flat: np.ndarray) -> np.ndarray:
    """Find the most frequent semantic label inside each voxel, fully
    vectorised (no Python loop over voxels, which matters since there
    can be tens of thousands of them).

    Trick: encode each (voxel, semantic_label) pair as one integer, count
    how often each pair occurs, then for every voxel keep whichever
    semantic label had the highest count. Sorting by (voxel, -count) and
    taking the first row per voxel gives exactly that maximum in one pass.
    """
    n_classes = int(semantic.max()) + 1 if len(semantic) else 1
    combined = flat_idx.astype(np.int64) * n_classes + semantic.astype(np.int64)
    unique_combined, pair_counts = np.unique(combined, return_counts=True)
    combined_voxel = unique_combined // n_classes
    combined_semantic = unique_combined % n_classes

    order = np.lexsort((-pair_counts, combined_voxel))
    combined_voxel_sorted = combined_voxel[order]
    combined_semantic_sorted = combined_semantic[order]
    first_of_voxel = np.concatenate(([True], combined_voxel_sorted[1:] != combined_voxel_sorted[:-1]))

    dominant_by_voxel = combined_semantic_sorted[first_of_voxel]
    voxel_ids_sorted = combined_voxel_sorted[first_of_voxel]
    # voxel_ids_sorted is already the sorted set of unique voxels, which
    # matches unique_flat's own (sorted) order, so a direct assignment
    # lines them up correctly.
    assert np.array_equal(voxel_ids_sorted, unique_flat)
    return dominant_by_voxel.astype(np.int64)


def _max_consecutive_run(flags: np.ndarray) -> np.ndarray:
    """Longest run of consecutive True values along axis 1 of a (M, S)
    boolean array, returned per row. Used to distinguish a ray that
    genuinely passes through a solid object from one that merely clips
    a single voxel in passing."""
    max_run = np.zeros(flags.shape[0], dtype=np.int32)
    current_run = np.zeros(flags.shape[0], dtype=np.int32)
    for step in range(flags.shape[1]):
        current_run = (current_run + 1) * flags[:, step]
        np.maximum(max_run, current_run, out=max_run)
    return max_run


def within_radar_fov(centres: np.ndarray, config: VoxelGridConfig) -> np.ndarray:
    """Which candidates lie inside the modelled radar's azimuth coverage.

    Deliberately the same shape of operation as cull_occluded(): a boolean
    filter applied BEFORE features are computed, so an out-of-FOV candidate
    never becomes a training example or a scored output at all. It is not a
    post-hoc correction and not a low predicted probability -- a candidate
    behind a forward-looking radar is not "unlikely to be detected", it is
    outside what this sensor model describes.

    Azimuth is measured from the radar mount, matching compute_features().

    Returns a boolean array (M,), True where a candidate is inside the FOV.
    """
    radar_position = np.asarray(config.radar_position, dtype=np.float64)
    rel = centres.astype(np.float64) - radar_position
    azimuth_deg = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    return np.abs(azimuth_deg) <= config.radar_fov_deg


def cull_occluded(
    centres: np.ndarray,
    voxel_ijk: np.ndarray,
    config: VoxelGridConfig,
    min_consecutive_blocking_voxels: int | None = None,
    occupancy: tuple | None = None,
) -> np.ndarray:
    """Step 2: mark candidates that are hidden behind another surface.

    For each candidate, walk a ray from the radar to that candidate's
    centre in steps of one voxel_size, and check which of those steps land
    in occupied voxels. This is a hard geometric constraint, not a
    probability -- per CLAUDE.md, occluded voxels must never be treated as
    detectable candidates.

    The subtlety is what counts as "blocked". Testing for a *single*
    occupied voxel along the ray (the original behaviour, still available
    via min_consecutive_blocking_voxels=1) badly over-culls on a
    bumper-mounted sensor: the road is a continuous sheet of occupied
    voxels, and a ray to a low target descends so gradually that it clips
    road voxels it never actually passes through -- the ray is inside the
    voxel but above the surface within it. Measured on VoD, that removed
    80-97% of voxels lying ON annotated objects, and hit low-body returns
    (trucks) far harder than torso-height ones (pedestrians).

    Requiring N *consecutive* occupied voxels fixes this: a genuine
    occluder (wall, vehicle body) spans several voxels along the ray,
    whereas a grazing road clip is one isolated voxel. Samples within
    1.5 voxels of the candidate are excluded so that neither the
    candidate's own voxel nor its immediate neighbour along the ray can
    ever count as its occluder.

    Returns a boolean array (M,), True where a candidate is visible
    (not occluded).
    """
    if min_consecutive_blocking_voxels is None:
        min_consecutive_blocking_voxels = config.min_consecutive_blocking_voxels
    occupied_along_ray = ray_occupancy(centres, voxel_ijk, config, occupancy)
    if min_consecutive_blocking_voxels <= 1:
        occluded = np.any(occupied_along_ray, axis=1)
    else:
        occluded = _max_consecutive_run(occupied_along_ray) >= min_consecutive_blocking_voxels
    return ~occluded


def occlusion_depth(centres: np.ndarray, voxel_ijk: np.ndarray,
                    config: VoxelGridConfig,
                    occupancy: tuple | None = None) -> np.ndarray:
    """How many occupied voxels lie between the radar and each candidate.

    The binary cull asks "is anything in the way"; this measures HOW MUCH is
    in the way, which is the quantity the sensor being modelled actually
    responds to. LiDAR occlusion is binary and exactly computable; radar
    visibility through and around an obstacle is not -- it depends on
    material, frequency and geometry this pipeline never observes. So per
    CLAUDE.md the exactly-computable geometric quantity is computed here and
    its relationship to detectability is left to the head to learn, rather
    than being hard-coded as a threshold.
    """
    return ray_occupancy(centres, voxel_ijk, config, occupancy).sum(axis=1).astype(np.float32)


def ray_occupancy(centres: np.ndarray, voxel_ijk: np.ndarray,
                  config: VoxelGridConfig,
                  occupancy: tuple | None = None) -> np.ndarray:
    """(M, S) boolean: which samples along each candidate's ray are occupied.

    Factored out of cull_occluded so the binary cull and the depth measure
    share one ray march instead of paying for it twice.

    `occupancy` is an optional (cell_ijk, shape, size, exclude_mask) tuple from
    `occupancy_cells`, letting the occluder lattice differ from the candidate
    lattice and letting cells be excluded from acting as occluders (Part B).
    Omitted, the occluders are the candidate voxels themselves at
    `config.voxel_size` -- the original behaviour, unchanged.
    """
    if occupancy is None:
        nx, ny, nz = _voxel_grid_shape(config)
        cells, size, exclude = voxel_ijk, config.voxel_size, None
    else:
        cells, (nx, ny, nz), size, exclude = occupancy
    occ_grid = np.zeros((nx, ny, nz), dtype=bool)
    keep = cells if exclude is None else cells[~exclude]
    occ_grid[keep[:, 0], keep[:, 1], keep[:, 2]] = True

    radar_position = np.asarray(config.radar_position, dtype=np.float64)
    to_candidate = centres.astype(np.float64) - radar_position
    ranges = np.linalg.norm(to_candidate, axis=1)
    ranges_safe = np.maximum(ranges, 1e-6)
    directions = to_candidate / ranges_safe[:, None]

    voxel_size = size
    n_steps = max(int(np.ceil(ranges.max() / voxel_size)), 1) if len(ranges) else 1
    step_offsets = (np.arange(n_steps) + 0.5) * voxel_size  # (S,) sample at voxel centres along the ray

    # (M, S, 3): every sample point along every candidate's ray, in one shot
    sample_pts = radar_position[None, None, :] + step_offsets[None, :, None] * directions[:, None, :]

    # Stop half a voxel short of the candidate itself, so its own voxel
    # is never mistaken for an occluder in front of it.
    # (Widened to 1.5 voxels during the reverted Phase 1 trial, which also
    # excluded the immediately-preceding voxel; reverted 29/07/26 because it
    # inflated the candidate pool 793 -> 1348 per frame with no true-signal
    # gain. See CHECKPOINT.md.)
    valid_step = step_offsets[None, :] < (ranges[:, None] - 0.5 * voxel_size)

    origin = np.array([config.x_range[0], config.y_range[0], config.z_range[0]])
    idx = np.floor((sample_pts - origin) / voxel_size).astype(np.int64)
    in_bounds = (
        (idx[..., 0] >= 0) & (idx[..., 0] < nx)
        & (idx[..., 1] >= 0) & (idx[..., 1] < ny)
        & (idx[..., 2] >= 0) & (idx[..., 2] < nz)
    )
    lookup_mask = valid_step & in_bounds
    ix = np.clip(idx[..., 0], 0, nx - 1)
    iy = np.clip(idx[..., 1], 0, ny - 1)
    iz = np.clip(idx[..., 2], 0, nz - 1)
    return occ_grid[ix, iy, iz] & lookup_mask


def compute_incidence_cosine_from_normals(
    points_in_voxel: np.ndarray,
    voxel_center: np.ndarray,
    sensor_position: np.ndarray,
    k_neighbors: int = NORMAL_ESTIMATION_K,
) -> float:
    """Cosine of the LiDAR incidence angle at one voxel, from local geometry.

    Fits a plane to the `k_neighbors` points nearest the voxel centre by PCA
    -- the surface normal is the eigenvector of the smallest eigenvalue of
    their covariance, i.e. the direction of least spread. The incidence
    cosine is then |dot(normal, unit ray from sensor to voxel)|.

    The absolute value matches KITTI-CARLA's convention, whose native
    cos_angle_lidar_surface is strictly positive: a surface and its flipped
    normal describe the same geometry, and PCA gives no consistent sign.

    Returns NaN when the neighbourhood has too few points to define a plane;
    callers decide how to fill those (see estimate_incidence_cosines).
    """
    if len(points_in_voxel) < MIN_POINTS_FOR_PLANE_FIT:
        return float("nan")

    points = np.asarray(points_in_voxel, dtype=np.float64)
    if len(points) > k_neighbors:
        nearest = np.argsort(np.linalg.norm(points - voxel_center, axis=1))[:k_neighbors]
        points = points[nearest]
        if len(points) < MIN_POINTS_FOR_PLANE_FIT:
            return float("nan")

    centred = points - points.mean(axis=0)
    # eigh returns ascending eigenvalues, so column 0 is the least-spread
    # direction = the plane normal.
    _, eigenvectors = np.linalg.eigh(centred.T @ centred)
    normal = eigenvectors[:, 0]

    ray = np.asarray(voxel_center, dtype=np.float64) - np.asarray(sensor_position, dtype=np.float64)
    ray_norm = np.linalg.norm(ray)
    if ray_norm < 1e-9:
        return float("nan")

    return float(np.clip(abs(np.dot(normal, ray / ray_norm)), 0.0, 1.0))


def estimate_incidence_cosines(
    lidar_xyz: np.ndarray,
    centres: np.ndarray,
    sensor_position: np.ndarray,
    radius: float = NORMAL_ESTIMATION_RADIUS_M,
    k_neighbors: int = NORMAL_ESTIMATION_K,
) -> tuple:
    """Batch driver for compute_incidence_cosine_from_normals.

    Gathers each candidate's neighbourhood with one cKDTree radius query
    rather than rescanning the cloud per voxel, then plane-fits each. Voxels
    whose neighbourhood is too sparse to fit a plane are filled with the
    median of the successful estimates in the same frame -- a neutral choice
    that avoids inventing either a grazing or a face-on incidence.

    Returns (cosines (M,) float64, n_failed int).
    """
    if len(centres) == 0:
        return np.zeros(0, dtype=np.float64), 0

    tree = cKDTree(np.asarray(lidar_xyz, dtype=np.float64))
    neighbourhoods = tree.query_ball_point(np.asarray(centres, dtype=np.float64), r=radius)

    cosines = np.empty(len(centres), dtype=np.float64)
    for i, neighbour_idx in enumerate(neighbourhoods):
        cosines[i] = compute_incidence_cosine_from_normals(
            lidar_xyz[neighbour_idx], centres[i], sensor_position, k_neighbors
        )

    failed = ~np.isfinite(cosines)
    n_failed = int(failed.sum())
    if n_failed:
        fill = np.median(cosines[~failed]) if n_failed < len(cosines) else 0.5
        cosines[failed] = fill
    return cosines, n_failed


# --- Ground-height estimation --------------------------------------------
# Two estimators exist for the same quantity, for the same reason cos_angle
# has two: datasets disagree on what they provide.
#   compute_local_ground_z_geometric()  needs no semantics at all.
#   compute_local_ground_z_geometric()  needs only xyz.
# The semantic version silently degrades to scene-minimum-z on any dataset
# without a road class -- which is VoD -- putting "height above the local
# road" and "height above the lowest point anywhere in the scene" in the
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


def enumerate_candidates(
    lidar: np.ndarray,
    lidar_semantic: np.ndarray,
    ego_velocity: np.ndarray,
    config: VoxelGridConfig | None = None,
) -> CandidateSet:
    """Run the full pipeline on one frame's LiDAR points: voxelise, cull
    occluded voxels, compute physics features, and return only the
    surviving (visible) candidates.

    `lidar` is (N, 4) and `lidar_semantic` is (N,). Column 3 of `lidar` is
    interpreted per `config.cos_angle_source`: taken directly as the
    incidence cosine (KITTI-CARLA's native field) under
    COS_ANGLE_FROM_COLUMN, or treated as intensity and replaced by a
    geometry-derived cosine under COS_ANGLE_FROM_NORMALS (VoD, which has no
    incidence-angle field). `ego_velocity` is (3,) m/s.
    """
    config = config or VoxelGridConfig()
    if config.cos_angle_source not in (COS_ANGLE_FROM_COLUMN, COS_ANGLE_FROM_NORMALS):
        raise ValueError(
            f"cos_angle_source must be {COS_ANGLE_FROM_COLUMN!r} or "
            f"{COS_ANGLE_FROM_NORMALS!r}, got {config.cos_angle_source!r}"
        )

    centres, mean_column3, dominant_semantic, point_count, voxel_ijk, voxel_sizes = voxelise(
        lidar, lidar_semantic, config
    )
    n_before_culling = len(centres)

    # The occluder lattice is built from the raw points, independently of the
    # candidate lattice, so Part A (candidate resolution) and Part B (which
    # cells may occlude) stay separately attributable. With a fixed schedule
    # and no ground exclusion this is provably the same occupancy set the
    # candidate voxels gave, so the default path is unchanged.
    occupancy = None
    if (config.voxel_size_schedule != SCHEDULE_FIXED
            or config.occlusion_voxel_size is not None
            or not config.ground_voxels_occlude):
        cells, shape, occ_size = occupancy_cells(lidar, config)
        exclude = (None if config.ground_voxels_occlude
                   else ground_occluder_mask(cells, shape, occ_size, lidar, config))
        occupancy = (cells, shape, occ_size, exclude)

    depth = occlusion_depth(centres, voxel_ijk, config, occupancy)
    if config.max_occlusion_depth > 0:
        visible = depth <= config.max_occlusion_depth
    else:
        visible = cull_occluded(centres, voxel_ijk, config, occupancy=occupancy)
    if config.apply_fov_mask:
        visible &= within_radar_fov(centres, config)
    n_after_culling = int(visible.sum())

    mean_intensity = None
    n_normal_failures = 0
    if config.cos_angle_source == COS_ANGLE_FROM_COLUMN:
        cos_angle = mean_column3[visible]
    else:
        # Column 3 is intensity, not a cosine. Keep it separately and derive
        # the real incidence cosine from local surface geometry.
        mean_intensity = mean_column3[visible]
        cos_angle, n_normal_failures = estimate_incidence_cosines(
            lidar[:, :3].astype(np.float64), centres[visible].astype(np.float64),
            np.asarray(config.radar_position, dtype=np.float64),
            config.normal_radius, config.normal_k_neighbors,
        )
        cos_angle = cos_angle.astype(np.float32)

    if config.ground_z_source != GROUND_FROM_GEOMETRY:
        raise ValueError(
            f"ground_z_source must be {GROUND_FROM_GEOMETRY!r}, got "
            f"{config.ground_z_source!r}. The semantic-road variant was removed: "
            f"it made the same feature slot mean different things on different "
            f"datasets (VoD mean 5.55 m vs KITTI-CARLA 0.89 m).")
    local_ground_z, _, _ = compute_local_ground_z_geometric(
        lidar[:, :3], centres[visible], z_bounds=config.z_range)
    features = compute_features(
        centres[visible], cos_angle, point_count[visible],
        config, local_ground_z, ego_velocity, depth[visible],
        voxel_volume=voxel_sizes[visible] ** 3,
    )

    if not (MIN_EXPECTED_CANDIDATES <= n_after_culling <= MAX_EXPECTED_CANDIDATES):
        log.warning(
            "Candidate count after occlusion culling is %d, outside the expected "
            "[%d, %d] range -- consider tuning voxel_size (currently %.2fm).",
            n_after_culling, MIN_EXPECTED_CANDIDATES, MAX_EXPECTED_CANDIDATES, config.voxel_size,
        )

    log.info("Candidates before culling: %d, after culling: %d", n_before_culling, n_after_culling)

    return CandidateSet(
        centres=centres[visible],
        features=features,
        semantic=dominant_semantic[visible],
        valid_mask=np.ones(n_after_culling, dtype=bool),
        mean_intensity=mean_intensity,
        n_normal_fit_failures=n_normal_failures,
    )
