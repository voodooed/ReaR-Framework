"""Shared per-frame helpers for scripts/fit_dataset.py and scripts/train_heads.py.

Both walk the same real-radar-paired frames through the same candidate
enumeration -> association -> soft-label pipeline; this module is the one
place that logic lives so the two scripts cannot drift apart from each other
or from rear/generate/generate_radar_frame.py's inference-time assumptions.

Everything here takes a `DatasetAdapter` and works for any dataset that
implements it (rear/data/base.py) -- nothing below reads a VoD- or
KITTI-CARLA-specific field directly.
"""

from __future__ import annotations

import numpy as np

from rear.calibration.soft_labels import (
    BOX_MARGIN_M, build_soft_labels_v3_rank_weighted, distance_to_box,
    point_counts_from_features)
from rear.geometry.candidates import enumerate_candidates

RAW_FEATURE_NAMES = ["range", "azimuth", "elevation", "cos_angle",
                     "height_above_ground", "point_density", "path_loss",
                     "expected_vr", "occlusion_depth"]


def radar_position_of(frame, adapter):
    """This frame's own radar mount translation, falling back to the
    dataset-wide constant. Per-frame because nothing in the DatasetAdapter
    contract guarantees the mount is fixed across frames, even though it is
    for both datasets shipped here."""
    T = frame.calibration.get("radar_to_lidar")
    return np.asarray(T)[:3, 3] if T is not None else adapter.radar_mount_offset


def candidates_for_frame(adapter, frame):
    """Enumerate candidates for one frame at this dataset's fitted physics
    profile, using the frame's own radar position.

    Returns `(candidates, grid_config, point_semantic)`. The per-point semantic
    array is returned rather than discarded because the material pipeline
    VERIFIES its reconstructed point->candidate map against the candidate set's
    own dominant-semantic field, and that check needs the exact array
    enumeration was given (rear/materials/pipeline.py).
    """
    pos = radar_position_of(frame, adapter)
    cfg = adapter.grid_config(radar_position=tuple(pos))
    sem = (frame.lidar_semantic if frame.lidar_semantic is not None
           else np.zeros(len(frame.lidar), dtype=np.int64))
    cand = enumerate_candidates(frame.lidar, sem, frame.ego_velocity, cfg)
    return cand, cfg, sem


def radar_columns(adapter):
    """(rcs_index, v_r_comp_index) into this dataset's real radar array, read
    by NAME from `sensors.radar_columns` -- never a hardcoded index, so a new
    dataset only has to name its own columns correctly in its YAML.

    `v_r_comp` (ego-motion-COMPENSATED radial velocity) is required, not
    optional: rear.generate.generate_radar_frame always converts the clutter
    stream's v_r from a compensated pool back to raw before concatenating it
    with the surface stream (see that module's docstring), and asserts the
    pool is tagged 'compensated' before generating a single frame. Fitting
    the clutter process from raw v_r would silently double-count ego motion
    at generation time, so this fails loudly instead of guessing.
    """
    cols = adapter.config["sensors"].get("radar_columns")
    if not cols:
        raise ValueError(f"{adapter.name}: sensors.radar_columns is not set in "
                          f"its config -- needed to locate the rcs/v_r_comp "
                          f"columns of the real radar array by name.")
    if "rcs" in cols:
        rcs_i = cols.index("rcs")
    elif "radar_rcs_column" in adapter.config["sensors"]:
        rcs_i = int(adapter.config["sensors"]["radar_rcs_column"])
    else:
        raise ValueError(f"{adapter.name}: sensors.radar_columns has no 'rcs' "
                          f"entry and no sensors.radar_rcs_column fallback.")
    if "v_r_comp" not in cols:
        raise ValueError(
            f"{adapter.name}: sensors.radar_columns={cols!r} has no 'v_r_comp' "
            f"(ego-motion-compensated radial velocity) column. The clutter "
            f"process must be fit from a compensated column -- see this "
            f"function's docstring. If your radar only reports raw v_r, "
            f"compute a compensated column yourself before fitting: "
            f"v_r_comp = v_r_raw - (-(v_ego . unit_vector_to_detection)).")
    return rcs_i, cols.index("v_r_comp")


def radar_rcs_and_vr_comp(adapter, frame):
    rcs_i, vr_i = radar_columns(adapter)
    r = frame.radar
    return r[:, rcs_i].astype(np.float64), r[:, vr_i].astype(np.float64)


def in_fov_mask(points_xyz, origin, fov_deg):
    rel = np.asarray(points_xyz, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    return np.abs(az) <= (fov_deg / 2.0)


def operating_point(adapter):
    """This dataset's fitted (range_tolerance_m, cone_angle_deg), required.

    `rear.calibration.soft_labels.build_soft_labels_v3_rank_weighted`
    defaults to VoD's OWN fitted cell (1.0 m, 5 deg) as hardcoded module
    constants -- correct only by coincidence for a different dataset's radar.
    Every caller building soft labels for a NEW dataset must pass its own
    fitted values explicitly instead of relying on that default, which is
    why every soft-label call in this module routes through here rather than
    calling build_soft_labels_v3_rank_weighted with no override.
    """
    tol = adapter.fitted("association", "range_tolerance_m", default=None)
    ang = adapter.fitted("association", "cone_angle_deg", default=None)
    if tol is None or ang is None:
        raise SystemExit(
            f"{adapter.name}.yaml has no fitted.association operating point yet. "
            f"Run `scripts/fit_dataset.py --dataset {adapter.name} --stage "
            f"association` first and paste its recommended "
            f"(range_tolerance_m, cone_angle_deg) into the config's `fitted:` "
            f"block before building soft labels for this dataset.")
    return float(tol), float(ang)


def build_labeled_frame(adapter, frame):
    """One frame's candidates + V3 soft labels + association bookkeeping, at
    THIS DATASET'S OWN fitted operating point (see `operating_point` above).

    Returns per-candidate aligned arrays:
      X                 (n, 9) raw physics features
      y                 (n,)   V3 soft label in [epsilon, 1]
      pos               (n,)   proxy positive: matched AND on an annotated box
      neg               (n,)   proxy negative: far from every box AND unmatched
      winning_detection (n,)   index into frame.radar of the detection this
                                candidate won, -1 if unmatched
    Proxy pos/neg exist only for AUC reporting during training -- the actual
    training target is the continuous soft label `y`, never pos/neg.
    """
    cand, cfg, sem = candidates_for_frame(adapter, frame)
    n_feat = cand.features.shape[1] if cand.features.size else len(RAW_FEATURE_NAMES)
    if len(cand.centres) == 0:
        z = np.zeros(0)
        return dict(X=np.zeros((0, n_feat), np.float32), y=z.astype(np.float32),
                    pos=np.zeros(0, bool), neg=np.zeros(0, bool),
                    winning_detection=np.zeros(0, dtype=np.int64),
                    candidates=cand, grid_config=cfg, point_semantic=sem)

    tol, ang = operating_point(adapter)
    radar_lidar = adapter.radar_xyz_in_lidar(frame)
    origin = np.asarray(cfg.radar_position, dtype=np.float64)
    counts = point_counts_from_features(cand.features, cfg.voxel_size)
    res = build_soft_labels_v3_rank_weighted(
        cand.centres, radar_lidar, frame.boxes, origin,
        candidate_point_counts=counts, range_tol=tol, cone_angle_deg=ang)

    far = np.ones(len(cand.centres), dtype=bool)
    for b in frame.boxes:
        far &= distance_to_box(cand.centres.astype(np.float64), b) > (BOX_MARGIN_M * 4)

    return dict(X=cand.features.astype(np.float32), y=res.labels.astype(np.float32),
                pos=(res.matched & res.on_box), neg=(far & ~res.matched),
                winning_detection=res.winning_detection,
                candidates=cand, grid_config=cfg, point_semantic=sem)
