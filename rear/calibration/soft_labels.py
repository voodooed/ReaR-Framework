"""Soft training labels for the surface process.

Hard binary labels (matched = 1, unmatched = 0) treat the association
matcher as a clean oracle. The null-control measurements in CHECKPOINT.md
say it is not: at the operating point used here (1.0m range, 5 deg cone,
reverted pipeline) box-conditioned association is 66.4% observed against a
21.0% azimuth-jitter null, i.e. a true signal of 45.4 pp. Roughly a third
of matches at this operating point are indistinguishable from coincidence
(implied match precision ~= 45.4 / 66.4 = 0.68). Asserting label 1.0 on
all of them would inject exactly that much noise straight into the
detection head, and asserting 0.0 on every unmatched candidate would claim
a certainty the measurement does not support either.

So every label here is a confidence in [EPSILON_UNMATCHED, MAX_CONFIDENCE],
never a hard 0 or 1.

THREE FORMULAS are implemented. **V3 is the selected default.** V1 and V2
are kept for reference and reproducibility, not as live options.

    V1  build_soft_labels_v1_competing_split
        confidence = (1 - angular_dist / cone_angle) / n_competing

    V2  build_soft_labels_v2_density_weighted
        confidence = (1 - angular_dist / cone_angle) * point_count
                     / max(point_count among competitors)

    V3  build_soft_labels_v3_rank_weighted           <-- DEFAULT
        confidence = (1 - angular_dist / cone_angle)
                     * rankdata(point_count, "average") / n_competing

All three then apply, per candidate:

    unmatched                -> EPSILON_UNMATCHED
    not on an annotated box  -> multiplied by OUT_OF_BOX_DISCOUNT
    all values               -> clipped to [EPSILON_UNMATCHED, MAX_CONFIDENCE]

and a candidate claimed by several detections keeps its best (max)
confidence, so labels never accumulate past 1.0.

WHY V3 (measured, not assumed -- see scripts/compare_soft_label_formulas.py
and scripts/point_count_transfer_check.py):

  V1 collapses with competition count. `n_competing` is how many 0.4m
  voxels fall inside one detection's tolerance, but a radar resolution
  cell is ~1.75m x 1.0m at 20m -- far coarser than the grid -- so those n
  voxels are usually ONE physical return subdivided by our enumeration
  choice, not n independent claims. Dividing by n therefore penalises a
  candidate for sitting on a densely sampled surface. Measured: mean
  confidence falls 15.4x from n=1 to n=21+ (0.319 -> 0.021), a grid
  artifact rather than evidence.

  V2 fixes that on VoD and is better calibrated there, but its magnitude
  information does NOT transfer to simulated LiDAR. KITTI-CARLA's median
  voxel holds 3 points against VoD's 14; at that density the difference
  between 2 and 3 returns is Poisson sampling noise, not surface
  evidence. The within-group ratio consequently degenerates into a comb
  of discrete spikes (1/2, 1/3, 2/3, ...): over half of KITTI-CARLA's V2
  weights land on just 5 distinct values, against under a quarter for
  VoD.

  V3 keeps only the ORDERING of point counts within a group, which is
  stable across both densities. VoD-vs-KITTI-CARLA mean gap 0.031 (V2:
  0.100); top-5 concentration gap 0.160 (V2: 0.294); decile shapes track
  closely where V2 diverges badly at the low end. V3 also happens to beat
  V2 on the VoD-only axes (high/low competition ratio 0.723 vs 0.544,
  calibration 0.533 vs 0.444 of the measured precision).

  The cost of V3 is real: ranking discards fine magnitude differences, so
  counts (100, 99, 98) score identically to (100, 5, 1). That loses
  genuine information about how much better observed the top voxel is.
  It is accepted deliberately -- this label scheme exists to support
  sim2real transfer, so transfer stability is the deciding criterion, and
  a weight that means different things on real and simulated LiDAR would
  defeat the purpose of the whole pipeline.

Everything tunable is a named module constant below. Nothing is buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

# --- Association operating point -----------------------------------------
# Fixed to the measured-optimal cell. The true-signal sweep
# (outputs/true_signal_sweep.png) showed the best alternative, (0.5m, 7 deg),
# beats this by only +1.4 pp with a bootstrap 95% CI of [-1.9, +4.8] pp --
# i.e. not statistically distinguishable. That question is closed; these are
# not knobs to retune without a fresh paired null control.
RANGE_TOLERANCE_M = 1.0
CONE_ANGLE_DEG = 5.0

# --- Label constants ------------------------------------------------------
# Floor for unmatched candidates. Deliberately > 0: "no radar detection
# landed within tolerance" is weak evidence of non-detectability, not proof.
# The candidate may be a real reflector the radar missed on this sweep, or
# one the matcher failed on. 0.02 keeps such candidates as strong negatives
# without asserting impossibility.
EPSILON_UNMATCHED = 0.02

# Ceiling for a perfect, unambiguous, on-box match.
MAX_CONFIDENCE = 1.0

# ASSUMPTION, NOT A MEASURED FACT -- flagged for future validation.
# The 45.4 pp true-signal ceiling was measured ONLY on the box-conditioned
# subset (detections on annotated objects, 2,369 of 14,105 = 16.8% of all
# detections). Whether that signal quality carries over to the general
# scene -- facades, vegetation, parked clutter -- is untested. Matches off
# annotated objects are therefore discounted. 0.7 is a judgement call chosen
# to be conservative, not a fitted value. To validate it, extend the
# azimuth-jitter null control to non-box-conditioned matches and set this to
# the measured ratio of their true signals.
OUT_OF_BOX_DISCOUNT = 0.7

# Distance from an annotated 3D box within which a candidate counts as
# "on an object". Matches BOX_MARGIN_M in the association study so the two
# analyses stay comparable.
BOX_MARGIN_M = 0.5

# NOTE on a definitional subtlety: the association sweep conditioned on the
# *detection* being near a box, whereas labels are per-candidate, so this
# module conditions on the *candidate* being near a box. For a matched pair
# the two differ by at most the range tolerance, so a candidate matched to
# an on-box detection can still fall outside BOX_MARGIN_M and be discounted.
# That is the conservative direction and is intentional, but it means the
# two subsets are near-equivalent rather than identical.


FORMULA_V1 = "v1_competing_split"
FORMULA_V2 = "v2_density_weighted"
FORMULA_V3 = "v3_rank_weighted"

# The selected formula. V1/V2 remain callable for reference and for
# reproducing the comparison, but this is what the pipeline uses.
DEFAULT_FORMULA = FORMULA_V3

# Formulas that weight by LiDAR evidence and therefore need point counts.
FORMULAS_NEEDING_POINT_COUNTS = (FORMULA_V2, FORMULA_V3)


@dataclass
class SoftLabelResult:
    """Labels plus the intermediate masks, so callers verifying the label
    construction do not have to recompute them."""

    labels: np.ndarray          # (M,) float64 in [EPSILON_UNMATCHED, MAX_CONFIDENCE]
    matched: np.ndarray         # (M,) bool -- candidate inside some detection's tolerance
    on_box: np.ndarray          # (M,) bool -- candidate within BOX_MARGIN_M of an annotated box
    raw_confidence: np.ndarray  # (M,) float64 -- before discount and clipping
    n_competing: np.ndarray     # (M,) int -- competitors of the detection that claimed it
    formula: str = DEFAULT_FORMULA  # which confidence formula produced these labels
    # Index of the detection that won each candidate, i.e. the one giving it
    # its highest confidence; -1 where unmatched. Purely a record of the
    # existing argmax the loop already performs -- it changes no label value.
    # Needed downstream to attribute a per-detection quantity (e.g. RCS) to a
    # candidate without averaging across physically distinct objects.
    winning_detection: np.ndarray | None = None


def point_counts_from_features(features: np.ndarray, voxel_size: float) -> np.ndarray:
    """Recover per-voxel LiDAR point counts from a CandidateSet's feature
    matrix. Column 5 is point_density = point_count / voxel_size**3, so this
    inverts exactly (counts are integers)."""
    return np.rint(features[:, 5].astype(np.float64) * voxel_size ** 3).astype(np.int64)


def distance_to_box(points_xyz: np.ndarray, box) -> np.ndarray:
    """Euclidean distance from each point to an oriented 3D box, 0 inside it.

    Reimplemented here rather than imported from scripts/ so that src/ never
    depends on the analysis scripts.
    """
    rel = points_xyz - np.array([box.x, box.y, box.z])
    c, s = np.cos(-box.rotation_z), np.sin(-box.rotation_z)
    local = np.stack([
        c * rel[:, 0] - s * rel[:, 1],
        s * rel[:, 0] + c * rel[:, 1],
        rel[:, 2],
    ], axis=-1)
    half = np.array([box.length / 2.0, box.width / 2.0, box.height / 2.0])
    return np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)


def _range_and_direction(points: np.ndarray, origin: np.ndarray) -> tuple:
    rel = points - origin
    ranges = np.linalg.norm(rel, axis=1)
    return ranges, rel / np.maximum(ranges, 1e-6)[:, None]


def build_soft_labels_detailed(
    candidate_centres: np.ndarray,
    radar_xyz_lidar_frame: np.ndarray,
    boxes: list,
    radar_position: np.ndarray,
    candidate_point_counts: np.ndarray | None = None,
    formula: str = DEFAULT_FORMULA,
    range_tol: float = RANGE_TOLERANCE_M,
    cone_angle_deg: float = CONE_ANGLE_DEG,
    epsilon: float = EPSILON_UNMATCHED,
    out_of_box_discount: float = OUT_OF_BOX_DISCOUNT,
    box_margin: float = BOX_MARGIN_M,
) -> SoftLabelResult:
    """Soft labels for one frame's candidates, with diagnostics.

    All inputs must already be in the LiDAR frame -- `radar_position` is the
    translation of radar_to_lidar, matching the association study.
    `candidate_point_counts` is required for FORMULA_V2 and FORMULA_V3
    (the default) and ignored by V1; recover it with
    point_counts_from_features().
    """
    valid = (FORMULA_V1, FORMULA_V2, FORMULA_V3)
    if formula not in valid:
        raise ValueError(f"formula must be one of {valid}, got {formula!r}")
    if formula in FORMULAS_NEEDING_POINT_COUNTS and candidate_point_counts is None:
        raise ValueError(f"{formula!r} needs candidate_point_counts (see point_counts_from_features)")

    n_candidates = len(candidate_centres)
    raw_confidence = np.zeros(n_candidates, dtype=np.float64)
    n_competing = np.zeros(n_candidates, dtype=np.int64)
    winning_detection = np.full(n_candidates, -1, dtype=np.int64)

    if n_candidates == 0:
        empty_b = np.zeros(0, dtype=bool)
        return SoftLabelResult(np.zeros(0), empty_b, empty_b, np.zeros(0),
                                np.zeros(0, dtype=np.int64), formula,
                                np.zeros(0, dtype=np.int64))

    cone_rad = np.radians(cone_angle_deg)
    cand_range, cand_dir = _range_and_direction(candidate_centres.astype(np.float64), radar_position)
    counts = None if candidate_point_counts is None else np.asarray(candidate_point_counts, dtype=np.float64)

    if len(radar_xyz_lidar_frame) > 0:
        det_range, det_dir = _range_and_direction(radar_xyz_lidar_frame.astype(np.float64), radar_position)
        for i in range(len(radar_xyz_lidar_frame)):
            angle = np.arccos(np.clip(cand_dir @ det_dir[i], -1.0, 1.0))
            within = (np.abs(cand_range - det_range[i]) <= range_tol) & (angle <= cone_rad)
            competitors = int(within.sum())
            if competitors == 0:
                continue

            # Both formulas share this term: confidence falls off linearly
            # to 0 at the edge of the cone.
            angular_term = 1.0 - angle[within] / cone_rad

            if formula == FORMULA_V1:
                # Split the detection's confidence evenly across everything
                # it could equally have come from.
                confidence = angular_term / competitors
            elif formula == FORMULA_V2:
                # Scale by how much LiDAR evidence each voxel holds relative
                # to the best-observed voxel in the group. No division by n,
                # because n voxels inside one radar resolution cell are one
                # return subdivided by our grid, not n independent claims.
                counts_within = counts[within]
                confidence = angular_term * (counts_within / counts_within.max())
            else:
                # V3 (default): keep only the ORDERING of point counts, not
                # their magnitudes. Ranking is stable across the very
                # different point densities of real and simulated LiDAR,
                # where V2's raw ratio degenerates into a few discrete
                # values on sparse simulated data. Ties take the average
                # rank, so equally-observed voxels are treated equally.
                counts_within = counts[within]
                confidence = angular_term * (
                    rankdata(counts_within, method="average") / competitors
                )

            # A candidate claimed by several detections keeps its best
            # evidence rather than accumulating past 1.0.
            improved = confidence > raw_confidence[within]
            idx = np.nonzero(within)[0]
            raw_confidence[idx[improved]] = confidence[improved]
            n_competing[idx[improved]] = competitors
            winning_detection[idx[improved]] = i

    matched = raw_confidence > 0.0

    on_box = np.zeros(n_candidates, dtype=bool)
    for box in boxes:
        on_box |= distance_to_box(candidate_centres.astype(np.float64), box) <= box_margin

    labels = np.where(on_box, raw_confidence, raw_confidence * out_of_box_discount)
    labels = np.clip(labels, epsilon, MAX_CONFIDENCE)

    return SoftLabelResult(
        labels=labels, matched=matched, on_box=on_box,
        raw_confidence=raw_confidence, n_competing=n_competing, formula=formula,
        winning_detection=winning_detection,
    )


def build_soft_labels_v1_competing_split(
    candidate_centres: np.ndarray,
    radar_xyz_lidar_frame: np.ndarray,
    boxes: list,
    radar_position: np.ndarray,
    **kwargs,
) -> SoftLabelResult:
    """FORMULA A: confidence = (1 - angular_dist / cone_angle) / n_competing."""
    kwargs.pop("formula", None)
    return build_soft_labels_detailed(
        candidate_centres, radar_xyz_lidar_frame, boxes, radar_position,
        formula=FORMULA_V1, **kwargs,
    )


def build_soft_labels_v2_density_weighted(
    candidate_centres: np.ndarray,
    radar_xyz_lidar_frame: np.ndarray,
    boxes: list,
    radar_position: np.ndarray,
    candidate_point_counts: np.ndarray,
    **kwargs,
) -> SoftLabelResult:
    """FORMULA B: confidence = (1 - angular_dist / cone_angle) * density_weight,
    density_weight = point_count / max(point_count among competitors)."""
    kwargs.pop("formula", None)
    return build_soft_labels_detailed(
        candidate_centres, radar_xyz_lidar_frame, boxes, radar_position,
        candidate_point_counts=candidate_point_counts, formula=FORMULA_V2, **kwargs,
    )


def build_soft_labels_v3_rank_weighted(
    candidate_centres: np.ndarray,
    radar_xyz_lidar_frame: np.ndarray,
    boxes: list,
    radar_position: np.ndarray,
    candidate_point_counts: np.ndarray,
    **kwargs,
) -> SoftLabelResult:
    """SELECTED DEFAULT. confidence = (1 - angular_dist / cone_angle)
    * rankdata(point_count, "average") / n_competing.

    Uses only the ordering of point counts within a competing group, which
    transfers across the different densities of real and simulated LiDAR.
    See the module docstring for the measurements behind this choice."""
    kwargs.pop("formula", None)
    return build_soft_labels_detailed(
        candidate_centres, radar_xyz_lidar_frame, boxes, radar_position,
        candidate_point_counts=candidate_point_counts, formula=FORMULA_V3, **kwargs,
    )


def build_soft_labels(
    candidate_centres: np.ndarray,
    radar_xyz_lidar_frame: np.ndarray,
    boxes: list,
    radar_position: np.ndarray,
    candidate_point_counts: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Soft label array (M,) aligned to `candidate_centres`.

    Uses V3 (rank-weighted) by default -- the selected formula, chosen on
    transfer stability between real and simulated LiDAR. Because V3 weights
    by LiDAR evidence, `candidate_point_counts` is required; recover it with
    point_counts_from_features(). Pass `formula=FORMULA_V1` to fall back to
    the count-free formula.
    """
    return build_soft_labels_detailed(
        candidate_centres, radar_xyz_lidar_frame, boxes, radar_position,
        candidate_point_counts=candidate_point_counts, **kwargs,
    ).labels
