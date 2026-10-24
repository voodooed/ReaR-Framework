"""Feature-set variants for the expected_vr controlled comparison.

Fixing the ego-velocity frame made feature 7 (`expected_vr`) the strongest
single correlate of the detection soft label (Spearman -0.2369, up from
-0.0340) while detection test AUC FELL 0.8543 -> 0.8275. Diagnosis: because
`expected_vr = -(v_ego . u)`, its scale is essentially *boresight-ness x ego
speed*, and ego speed is a sequence-level property -- per-sequence
mean|expected_vr| spans 0.71 to 6.49 m/s, and the worst-hit test sequence sits
at 0.71 m/s, outside the entire training range of 1.67-5.20.

Three variants isolate that:

  DROP       7 features. expected_vr removed entirely.
  RAW        8 features, unchanged. The current standing head.
  BORESIGHT  8 features, expected_vr replaced by -(u . v_hat_ego): the cosine
             between the candidate ray and the ego motion DIRECTION, with
             speed magnitude divided out.

WHAT BORESIGHT ALSO REMOVES -- not only confound. Doppler magnitude is real
radar physics: a faster ego produces larger radial rates, which genuinely
change how detectable a scatterer is against ground clutter. BORESIGHT strips
that along with the sequence-level confound. So if BORESIGHT scores below RAW
that does NOT by itself mean it is the worse feature -- it means the magnitude
carried real signal alongside the confound, and the two cannot be separated by
this experiment alone. The per-sequence breakdown, not the aggregate, is what
distinguishes the two readings.

Exactness: BORESIGHT is computed as `expected_vr / ||v_ego||` rather than
re-enumerating. That is algebraically identical to -(u . v_hat_ego) --
expected_vr is -(v_ego . u) and ||v_ego|| is a per-frame scalar -- so it
avoids a second hour-long enumeration pass and any risk of the two paths
drifting apart.
"""

from __future__ import annotations

import numpy as np

VARIANT_DROP = "drop"
VARIANT_RAW = "raw"
VARIANT_BORESIGHT = "boresight"
VARIANTS = (VARIANT_DROP, VARIANT_RAW, VARIANT_BORESIGHT)

F_EXPECTED_VR = 7

# Below this the ego motion direction is undefined (0/0), so the boresight
# cosine has no meaning and is set to 0. This conflates "stationary" with
# "ray perpendicular to motion", which is a real limitation of the variant and
# is reported as an affected-fraction rather than hidden.
MIN_SPEED_FOR_DIRECTION_MS = 0.1


def variant_feature_names(base_names, variant: str):
    """Feature names after the variant transform, in column order."""
    if variant == VARIANT_DROP:
        return [n for j, n in enumerate(base_names) if j != F_EXPECTED_VR]
    if variant == VARIANT_RAW:
        return list(base_names)
    if variant == VARIANT_BORESIGHT:
        return [("boresight_cos" if j == F_EXPECTED_VR else n)
                for j, n in enumerate(base_names)]
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def apply_variant(X: np.ndarray, frame_idx: np.ndarray,
                  speed_by_frame: np.ndarray, variant: str):
    """Transform a feature matrix into the requested variant.

    `frame_idx` is the per-row dataset frame index (the caches' `frame`
    column) and `speed_by_frame` is ||ego_velocity|| indexed by that same
    dataset index. Returns (X_out, n_undefined_rows).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    X = np.asarray(X, dtype=np.float64)
    if variant == VARIANT_RAW:
        return X, 0
    if variant == VARIANT_DROP:
        return np.delete(X, F_EXPECTED_VR, axis=1), 0

    speed = np.asarray(speed_by_frame, dtype=np.float64)[np.asarray(frame_idx)]
    ok = speed >= MIN_SPEED_FOR_DIRECTION_MS
    out = X.copy()
    bore = np.zeros(len(X))
    bore[ok] = X[ok, F_EXPECTED_VR] / speed[ok]
    # Numerical guard: the ratio is a cosine, so it cannot leave [-1, 1].
    # Anything outside is a speed/feature mismatch, not a real value.
    if np.any(np.abs(bore) > 1.0 + 1e-3):
        worst = float(np.abs(bore).max())
        raise ValueError(
            f"boresight cosine reached {worst:.4f}, outside [-1, 1]. "
            f"expected_vr and the ego speed table disagree -- check that "
            f"`frame_idx` indexes the same frames the features were built from.")
    out[:, F_EXPECTED_VR] = np.clip(bore, -1.0, 1.0)
    return out, int((~ok).sum())


def ego_speed_table(root: str, split: str) -> np.ndarray:
    """||ego_velocity|| for every frame in a split, indexed by dataset index.

    Built through VoDDataset's own ego table so it is exactly the velocity the
    features were computed from -- including the carry/interpolate repair --
    rather than a second, possibly divergent, pose-differencing pass.
    """
    from data.vod_loader import VoDDataset

    ds = VoDDataset(root, split=split)
    ds[0]  # forces _build_ego_table
    return np.linalg.norm(np.asarray(ds._ego_table, dtype=np.float64), axis=1)
