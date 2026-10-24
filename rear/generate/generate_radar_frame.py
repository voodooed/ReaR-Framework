"""Compose the validated components into one synthetic 3+1D radar frame.

One LiDAR sweep in, one radar point cloud out: `[x, y, z, RCS, v_r]`.

    points, is_clutter = generate_radar_frame(lidar, ego_velocity, seed=0)

Pipeline
--------
1-4. `enumerate_candidates` -- voxelise, cull occluded voxels, FOV-mask, and
     compute the 8 physics features. Occluded voxels are never enumerated.
5.   MATERIAL. Per-point semantic labels -> material mixtures -> per-candidate
     composition -> the 10-column electromagnetic block, appended AFTER the
     physics scaler (rear/materials/). A permanent stage, not an option.
6.   The head -- one network, both outputs -- scored on `[physics | material]`
     with the scaling constants persisted on its own checkpoint.
7.   Platt calibration of the detection probability, then a Bernoulli draw per
     candidate. A detection head outputs a probability; realising a frame means
     sampling it.
8.   RCS for the candidates that fired, de-standardised with the checkpoint's
     own y_mu/y_sigma, then quantile-calibrated onto the real RCS marginal.
9.   `v_r` computed analytically from ego motion, never predicted.
10.  Clutter drawn from the fitted inhomogeneous Poisson process, same FOV,
     rate conditioned on ego speed, and concatenated.

Two things the composition forces, both flagged because neither is covered by
the component validations
--------------------------------------------------------------------------
* CLUTTER ELEVATION IS NOT A VALIDATED COMPONENT. The fitted clutter process is
  spatially 2-D -- a KDE over (range, azimuth) only. Emitting a z requires an
  elevation, so the fit adds an empirical elevation pool stratified by the same
  range bins as the RCS/v_r pools. It never went through the null control.
  Elevation structure beyond the per-range marginal is NOT reproduced.

* THE TWO v_r STREAMS ARRIVE IN DIFFERENT CONVENTIONS AND MUST BE RECONCILED.
  Surface v_r is the ego-motion radial component, -(v_ego . u), which is raw v_r
  for a static scatterer. The clutter pools were built from the COMPENSATED
  column. Measured on VoD test, per frame:

      corr(v_comp - v_raw, -(v_ego . u)) = -0.9997, slope -0.998

  so v_raw = v_comp + expected_vr, and sampled clutter v_r is converted back by
  adding the ego term at that clutter point's own direction. Concatenating the
  two streams without this conversion would put two different quantities in one
  column -- silently, since both are small numbers in m/s.

Object motion
-------------
`v_r` carries an object-motion term wherever a box supplies a velocity. Neither
shipped dataset's box type does, so on both this term is always absent and every
generated `v_r` is ego-motion-only. The hook is live for datasets that do carry
one; the consequence for those that do not is that generated `v_r` cannot
reproduce independently moving objects.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from rear.calibration.feature_variants import (F_EXPECTED_VR,
                                               MIN_SPEED_FOR_DIRECTION_MS,
                                               VARIANT_BORESIGHT, VARIANT_DROP,
                                               VARIANT_RAW)
from rear.calibration.platt import (apply_detection_calibration,
                                    load_detection_calibration)
from rear.calibration.rcs_calibration import (apply_rcs_calibration,
                                              load_rcs_calibration)
from rear.calibration.scaling import scale_from_checkpoint
from rear.geometry.candidates import VoxelGridConfig, enumerate_candidates
from rear.materials.features import features_from_composition
from rear.materials.pipeline import point_to_candidate
from rear.materials.taxonomy import load_label_space
from rear.models import clutter_process as _clutter_process_compat  # noqa: F401
from rear.models.base import load_head

REPO_ROOT = Path(__file__).resolve().parents[2]

# Weights are namespaced by which dataset they were FITTED on (see
# rear/data/base.py's fitted_from chain): weights/<name>/ holds one checkpoint
# per architecture, the clutter process, and the two calibrations. BOTH
# calibrations are functions of a specific head's output distribution, so they
# carry a mandatory _<architecture> suffix -- see load_assets(). VoD is the only
# profile shipped fitted, so it is the default for callers that supply no
# adapter and therefore no `fitted.weights_dir` to resolve.
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights" / "vod"

SURFACE = 0
CLUTTER = 1

# Feature column indices, per compute_features()'s docstring.
F_RANGE, F_AZIMUTH, F_ELEVATION = 0, 1, 2

# --- v_r convention contract -----------------------------------------------
# The output column must be RAW v_r. The two streams do not arrive that way, so
# these constants and `assert_vr_conventions` make the conversion a standing
# check rather than a one-time measurement.
VR_OUTPUT_CONVENTION = "raw"
VR_SURFACE_CONVENTION = "raw"
VR_CLUTTER_POOL_CONVENTION = "compensated"
# A compensated pool is ego-motion-removed, so it sits at ~0 (90.2% of real
# clutter is stationary-looking). A raw pool would inherit the ego-speed spread
# and could not. Cheap numeric corroboration of the declared tag.
MAX_COMPENSATED_POOL_MEDIAN_ABS_VR = 0.5


def assert_vr_conventions(clutter_assets: dict) -> None:
    """Fail loudly if the clutter pools are not the convention this module
    converts from, before anything is concatenated.

    Checks the declared tag AND corroborates it numerically, because a tag alone
    can go stale if someone rebuilds the pools from a different column and
    forgets to update it.
    """
    declared = clutter_assets.get("vr_convention")
    if declared is None:
        raise ValueError(
            "clutter_process.pkl has no 'vr_convention' tag. Refit it with "
            "scripts/fit_dataset.py --stage clutter -- the generator will not "
            "guess which v_r convention the pools are in.")
    if declared != VR_CLUTTER_POOL_CONVENTION:
        raise ValueError(
            f"clutter pools declare v_r convention {declared!r}, but this module "
            f"converts from {VR_CLUTTER_POOL_CONVENTION!r} to "
            f"{VR_OUTPUT_CONVENTION!r}. Update the conversion to match; do not "
            f"silently concatenate.")
    med = float(np.median([np.median(np.abs(p)) for p in clutter_assets["process"].bin_vr]))
    if med > MAX_COMPENSATED_POOL_MEDIAN_ABS_VR:
        raise ValueError(
            f"clutter pools are tagged {declared!r} but their median |v_r| is "
            f"{med:.3f} m/s, above the {MAX_COMPENSATED_POOL_MEDIAN_ABS_VR} m/s "
            f"a compensated (ego-motion-removed) pool should show. The tag and "
            f"the data disagree -- check which column they were built from.")


def features_for_checkpoint(feats: np.ndarray, ckpt: dict,
                            ego_velocity: np.ndarray) -> np.ndarray:
    """Reshape the enumerated features into the variant a checkpoint expects.

    Enumeration always emits the raw 9; a head trained on 8 (expected_vr
    dropped, the production variant) or on the speed-normalised boresight cosine
    would otherwise be fed the wrong matrix. Reading the tag off the checkpoint
    keeps one source of truth, exactly as the scaling constants do.
    """
    variant = ckpt.get("feature_variant", VARIANT_RAW)
    if variant == VARIANT_RAW:
        return feats
    if variant == VARIANT_DROP:
        return np.delete(feats, F_EXPECTED_VR, axis=1)
    if variant == VARIANT_BORESIGHT:
        speed = float(np.linalg.norm(ego_velocity))
        out = feats.copy()
        out[:, F_EXPECTED_VR] = (
            np.clip(feats[:, F_EXPECTED_VR] / speed, -1.0, 1.0)
            if speed >= MIN_SPEED_FOR_DIRECTION_MS else 0.0)
        return out
    raise ValueError(f"checkpoint declares unknown feature_variant {variant!r}")


class GeneratedFrame(NamedTuple):
    """One synthetic radar frame.

    `points` is (M, 5) `[x, y, z, RCS, v_r]` in the LiDAR frame -- the whole
    deliverable. `is_clutter` is the parallel (M,) origin label a downstream
    consumer would ignore, but every joint check needs it to tell the two
    streams apart.
    """

    points: np.ndarray      # (M, 5) float64 -- x, y, z, RCS (dBsm), v_r (m/s)
    is_clutter: np.ndarray  # (M,) bool -- False = surface-origin, True = clutter


_ASSETS: dict = {}


def load_assets(out_dir: Path | None = None, device: str = "cpu",
                architecture: str = "prism") -> dict:
    """Load and cache one architecture's head, calibrations and clutter process.

    Cached module-level because the generator is called per frame and these are
    fixed artefacts; reloading a checkpoint 100 times is pure waste.

    BOTH CALIBRATIONS ARE RESOLVED PER ARCHITECTURE AND THERE IS NO FALLBACK.
    The Platt map is fitted so expected per-frame counts match real ones, which
    makes it a property of one head's probability scale; the RCS map bins
    candidates by that head's own predicted quantiles. Applying one head's
    calibration to another is silent -- it cannot move AUC or w-R^2, because
    both maps are monotone and calibration happens after scoring -- but it
    mis-scales the realised frame badly. So a missing file is an error here, not
    a fallback to whatever else is in the directory.
    """
    out_dir = Path(out_dir or DEFAULT_WEIGHTS_DIR)
    key = (str(out_dir), architecture, device)
    if key in _ASSETS:
        return _ASSETS[key]

    head_path = out_dir / f"{architecture}_head.pt"
    if not head_path.exists():
        available = sorted(p.stem.replace("_head", "")
                           for p in out_dir.glob("*_head.pt"))
        raise FileNotFoundError(
            f"no {architecture!r} head at {head_path}. Fitted architectures in "
            f"{out_dir}: {available or 'none'}. Train one with "
            f"scripts/train_heads.py --head {architecture}.")
    head, ck = load_head(head_path, device=device)

    with open(out_dir / "clutter_process.pkl", "rb") as f:
        clutter = pickle.load(f)
    assert_vr_conventions(clutter)      # before anything is concatenated

    det_cal = _require_calibration(out_dir, "detection_calibration", architecture)
    rcs_cal_path = _require_calibration(out_dir, "rcs_calibration", architecture)

    _ASSETS[key] = {
        "head": head, "ckpt": ck, "architecture": architecture,
        "det_cal": load_detection_calibration(det_cal),
        "rcs_cal": load_rcs_calibration(rcs_cal_path),
        "clutter": clutter["process"],
        "clutter_elevation": clutter["bin_elevation"],
        "device": device,
    }
    return _ASSETS[key]


def _require_calibration(out_dir: Path, stem: str, architecture: str) -> Path:
    p = out_dir / f"{stem}_{architecture}.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"no {stem} for architecture {architecture!r} ({p}). Calibrations "
            f"are per-head and are never shared, so this is an error rather "
            f"than a fallback -- refit with `python tools/refit_{stem}.py "
            f"--head {architecture}`.")
    return p


def material_block_for(candidates, lidar_xyz: np.ndarray,
                       point_labels: np.ndarray, label_space,
                       config: VoxelGridConfig, freq_ghz: float | None,
                       n_material: int) -> np.ndarray:
    """The (M, 10) electromagnetic block for one frame's candidates.

    Split out so the generation path and the fitting path build the block the
    same way from the same inputs; see rear/materials/pipeline.py for the
    two-stage rationale and the voxel-map verification.
    """
    from rear.materials.features import aggregate_composition
    vi = point_to_candidate(lidar_xyz, candidates.centres, config)
    mix = label_space.point_mixtures(point_labels)
    comp, frac = aggregate_composition(mix, vi, len(candidates.centres))
    block = features_from_composition(comp, frac, freq_ghz=freq_ghz)
    if block.shape[1] != n_material:
        raise ValueError(
            f"the head expects {n_material} material columns but the block has "
            f"{block.shape[1]}. The checkpoint and the material database "
            f"disagree -- retrain rather than reshaping.")
    return block


def _box_velocities(object_boxes) -> np.ndarray | None:
    """Per-box (3,) velocity if the caller's box type carries one, else None."""
    if not object_boxes:
        return None
    vels = [getattr(b, "velocity", None) for b in object_boxes]
    if any(v is None for v in vels):
        return None
    return np.asarray(vels, dtype=np.float64)


def _object_motion_vr(points_rel, unit_vectors, object_boxes, box_vels):
    """Radial component of object motion for points inside a moving box.

    Only fires when boxes carry velocities. Points outside every box keep 0,
    leaving them with the pure ego-motion term.
    """
    extra = np.zeros(len(points_rel))
    if box_vels is None:
        return extra
    for box, v in zip(object_boxes, box_vels):
        c = np.array([box.x, box.y, box.z], dtype=np.float64)
        half = max(box.length, box.width) / 2.0 + 0.5
        inside = (np.abs(points_rel[:, 0] - c[0]) <= half) & \
                 (np.abs(points_rel[:, 1] - c[1]) <= half) & \
                 (np.abs(points_rel[:, 2] - c[2]) <= box.height / 2.0 + 0.5)
        if inside.any():
            extra[inside] = unit_vectors[inside] @ v
    return extra


def generate_radar_frame(lidar_points: np.ndarray,
                         ego_velocity: np.ndarray,
                         object_boxes=None,
                         seed: int | None = None,
                         config: VoxelGridConfig | None = None,
                         lidar_semantic: np.ndarray | None = None,
                         point_labels: np.ndarray | None = None,
                         label_space=None,
                         material_freq_ghz: float | None = None,
                         out_dir: Path | None = None,
                         device: str = "cpu",
                         architecture: str = "prism",
                         rcs_calibration: bool = True,
                         allow_missing_material: bool = False) -> GeneratedFrame:
    """Generate one synthetic 3+1D radar frame from one LiDAR sweep.

    `lidar_points` is (N, 4); column 3 is interpreted per `config.cos_angle_source`.
    `ego_velocity` is (3,) m/s in LiDAR-frame axes. `object_boxes` is optional and
    used only for the object-motion term in `v_r`.

    MATERIAL. `point_labels` is a per-LiDAR-point semantic id in `label_space`'s
    vocabulary (-1 for unobserved). A head trained with a material branch needs
    it: without labels every candidate would receive the all-zero "no material
    information" row, which is a valid input but throws away the block's whole
    contribution. That degradation is therefore explicit --
    `allow_missing_material=True` -- rather than silent.

    All randomness derives from `seed` through a single Generator, so a given
    (frame, seed) is exactly reproducible: the Bernoulli detection draw and every
    clutter draw share it.
    """
    config = config or VoxelGridConfig()
    assets = load_assets(out_dir, device, architecture)
    head, ck = assets["head"], assets["ckpt"]
    rng = np.random.default_rng(seed)
    radar_position = np.asarray(config.radar_position, dtype=np.float64)

    # --- 1-4. enumerate: voxelise, cull occlusion, FOV-mask, features -------
    if lidar_semantic is None:
        lidar_semantic = np.zeros(len(lidar_points), dtype=np.int64)
    cand = enumerate_candidates(lidar_points, lidar_semantic, ego_velocity, config)
    feats = cand.features.astype(np.float64)
    if len(feats) == 0:
        return GeneratedFrame(np.zeros((0, 5)), np.zeros(0, dtype=bool))

    # --- 5-6. features + material -> the head ------------------------------
    X = scale_from_checkpoint(
        features_for_checkpoint(feats, ck, ego_velocity), ck)
    n_material = int(ck.get("n_material", 0))
    if n_material:
        if point_labels is None:
            if not allow_missing_material:
                raise ValueError(
                    f"the {architecture!r} head was trained with a "
                    f"{n_material}-column material block, but no `point_labels` "
                    f"were supplied. Pass per-point semantic ids (and a "
                    f"`label_space`), use generate_for_frame() which derives "
                    f"them from the dataset adapter, or pass "
                    f"allow_missing_material=True to accept the degraded "
                    f"no-material-information path deliberately.")
            block = np.zeros((len(feats), n_material), dtype=np.float64)
        else:
            space = (label_space if not isinstance(label_space, str)
                     else load_label_space(label_space))
            if space is None:
                raise ValueError("point_labels were supplied without a label_space; "
                                 "the ids cannot be resolved to materials.")
            block = material_block_for(cand, lidar_points[:, :3], point_labels,
                                       space, config, material_freq_ghz, n_material)
        # Appended AFTER scaling: the block is already bounded, and standardising
        # it would hand a near-zero-IQR column an enormous multiplier.
        X = np.hstack([X, block])

    Xs = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = head(Xs)
        p = torch.sigmoid(out["det_logit"]).cpu().numpy()
        rcs_std = out["rcs_mean"].cpu().numpy()

    # --- 7. monotone recalibration, then realisation ------------------------
    # The head is fitted against a candidate population 2.9x larger than the
    # real detection count; without this the realised frame over-produces.
    # Ranking is untouched, so AUC cannot move.
    p = apply_detection_calibration(p, assets["det_cal"])
    fired = rng.random(len(p)) < np.clip(p, 0.0, 1.0)
    n_surface = int(fired.sum())

    surf_xyz = cand.centres[fired].astype(np.float64)
    if n_surface:
        # --- 8. RCS, de-standardised with the checkpoint's own constants ----
        surf_rcs = (rcs_std[fired] * float(ck["y_sigma"]) + float(ck["y_mu"]))
        # Turn conditional-mean predictions into samples from the population
        # real radar actually produces. Without this the surface stream is ~4 dB
        # strong AND 2.3x under-dispersed. Draws from the frame's own rng, so
        # the frame stays reproducible. `rcs_calibration=False` is the measured
        # information-preserving setting -- it fixes per-detection assignment at
        # the cost of the marginal, which is a strictly negative trade
        # end-to-end; see rear/calibration/rcs_calibration.py.
        if rcs_calibration and assets["rcs_cal"] is not None:
            surf_rcs = apply_rcs_calibration(surf_rcs, assets["rcs_cal"], rng)

        # --- 9. v_r analytically. Feature 7 IS -(v_ego . u), computed at
        # enumeration by the same code the head was trained on, so reusing it
        # keeps one definition of ego-radial velocity rather than two.
        surf_vr = feats[fired, F_EXPECTED_VR].copy()
        rel = surf_xyz - radar_position
        u = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
        surf_vr += _object_motion_vr(surf_xyz, u, object_boxes,
                                     _box_velocities(object_boxes))
    else:
        surf_rcs = np.zeros(0)
        surf_vr = np.zeros(0)

    # --- 10. clutter, same FOV, ego-speed-conditioned rate ------------------
    proc = assets["clutter"]
    ego_speed = float(np.linalg.norm(ego_velocity))
    c_rng, c_az, c_rcs, c_vr_comp = proc.sample_frame(rng, ego_speed=ego_speed)

    if len(c_rng):
        # Elevation from the added (NOT validated) per-range-bin pool.
        b = np.clip(np.digitize(c_rng, proc.bin_edges) - 1, 0,
                    len(assets["clutter_elevation"]) - 1)
        c_el = np.radians([rng.choice(assets["clutter_elevation"][i]) for i in b])
        az = np.radians(c_az)
        c_rel = np.stack([c_rng * np.cos(c_el) * np.cos(az),
                          c_rng * np.cos(c_el) * np.sin(az),
                          c_rng * np.sin(c_el)], axis=1)
        clut_xyz = c_rel + radar_position
        # COMPENSATED -> RAW, so both streams share one convention in the output
        # column. Enforced by assert_vr_conventions() at load time.
        cu = c_rel / np.maximum(np.linalg.norm(c_rel, axis=1, keepdims=True), 1e-9)
        clut_vr = c_vr_comp + (-(cu @ np.asarray(ego_velocity, dtype=np.float64)))
    else:
        clut_xyz = np.zeros((0, 3)); c_rcs = np.zeros(0); clut_vr = np.zeros(0)

    points = np.concatenate([
        np.column_stack([surf_xyz, surf_rcs, surf_vr]),
        np.column_stack([clut_xyz, c_rcs, clut_vr]),
    ], axis=0) if (n_surface or len(clut_xyz)) else np.zeros((0, 5))
    is_clutter = np.concatenate([np.zeros(n_surface, dtype=bool),
                                 np.ones(len(clut_xyz), dtype=bool)])
    return GeneratedFrame(points, is_clutter)


# ---------------------------------------------------------------------------
# Adapter-driven entry point. `generate_radar_frame` above stays array-only, so
# it works on any LiDAR sweep from anywhere; this wrapper supplies its arguments
# from a DatasetAdapter + dataset config instead of from constants scattered
# through analysis scripts.
# ---------------------------------------------------------------------------
_EXTRACTORS: dict = {}


def _material_extractor(adapter, device: str):
    """One MaterialExtractor per (dataset, device) -- the segmentation model, if
    the dataset needs one, is loaded once and reused across frames."""
    key = (adapter.name, device)
    if key not in _EXTRACTORS:
        from rear.materials.pipeline import MaterialExtractor
        _EXTRACTORS[key] = MaterialExtractor(adapter, device=device)
    return _EXTRACTORS[key]


def generate_for_frame(adapter, frame, seed=None, device="cpu",
                       architecture=None, rcs_calibration=None,
                       lidar_semantic=None, out_dir=None,
                       material_device=None):
    """Generate one radar frame for any dataset.

    Reads voxel size, volume, FOV, cos-angle source, the material configuration
    and the RCS-calibration switch from the adapter's config, following the
    `fitted_from` chain for everything learned. The radar mount is taken per
    frame from that frame's own calibration where the dataset provides one.

    `material_device` selects where a segmentation model runs (defaults to
    `device`); a dataset shipping per-point semantics never loads one.
    """
    T = frame.calibration.get("radar_to_lidar")
    mount = (np.asarray(T)[:3, 3] if T is not None else adapter.radar_mount_offset)
    cfg = adapter.grid_config(radar_position=tuple(mount))

    if out_dir is None:
        # Follows `fitted_from` exactly like every other fitted value, so a
        # dataset with no `weights_dir` of its own transparently uses its
        # donor's fitted artifacts.
        wd = adapter.fitted("weights_dir", default=None)
        if wd is not None:
            wd = Path(wd)
            out_dir = wd if wd.is_absolute() else REPO_ROOT / wd
    if architecture is None:
        architecture = adapter.fitted("heads", "implementation", default="prism")
    if rcs_calibration is None:
        rcs_calibration = bool(adapter.fitted("rcs_calibration", "enabled",
                                              default=True))

    sem = lidar_semantic
    if sem is None:
        sem = (frame.lidar_semantic if frame.lidar_semantic is not None
               else np.zeros(len(frame.lidar), dtype=np.int64))

    mat = adapter.material_config()
    labels = space = None
    freq = None
    if mat.get("enabled"):
        ex = _material_extractor(adapter, material_device or device)
        labels, _conf = ex.point_labels(frame)
        space, freq = ex.label_space, ex.freq_ghz

    return generate_radar_frame(
        frame.lidar, frame.ego_velocity, frame.boxes or None, seed=seed,
        config=cfg, lidar_semantic=sem, point_labels=labels, label_space=space,
        material_freq_ghz=freq, out_dir=out_dir, device=device,
        architecture=architecture, rcs_calibration=rcs_calibration,
        allow_missing_material=not mat.get("enabled", False))
