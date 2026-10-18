"""Shared feature standardisation for both heads.

Standing infrastructure fix, not tied to any one feature bug.

Mean/std standardisation has a failure mode this project hit for real: the
scaling constants are fit ONCE over every training candidate, so a handful of
extreme values in one feature move sigma for that feature globally, and every
candidate's standardised value shifts -- including the ones that were fine.
Measured instance: corrupted ego velocity put `expected_vr` up to 935 m/s on
3.2% of rows, inflating its sigma from 2.92 to 10.32. Fixing the 3.2%
rescaled the standardised feature ~3.5x for 100% of candidates, and the
resulting prediction change was almost as large on uncorrupted candidates as
on corrupted ones (concentration ratio only 1.6-2.0x).

Median/IQR is the standard remedy: both statistics are order-based, so a
value at 935 or at 9350 shifts them by the same negligible amount as long as
the contaminated fraction stays below ~25%. That bounds the blast radius of
any future undiscovered outlier to the rows that actually contain it.

Targets are deliberately NOT rescaled this way -- see train_rcs_head.py. The
RCS target keeps mean/std because the Huber delta is expressed in units of
target sigma, so changing the target scale would silently redefine the loss.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-9

# Standardised values are clipped after scaling. Median/IQR fixes the
# CONSTANTS but not the VALUES, and for a heavy-tailed feature IQR << std, so
# the same raw tail becomes a LARGER standardised number. Measured on the RCS
# feature set: path_loss (= 1/range^4) has std/IQR = 440, and robust scaling
# alone pushed its |z|max from 55 to 24,437, which cost 0.023 test w-R2.
# Clipping restores that (0.2300 -> 0.2490) while keeping the detection gain.
# The clip is persisted with each checkpoint so inference applies the same
# transform as training -- an unclipped inference path would be train/serve skew.
STANDARDISED_CLIP = 10.0


def fit_robust_scaler(X: np.ndarray) -> tuple:
    """Per-feature median and IQR over the training candidates.

    Returns (center, scale) with the same shape as a feature row, ready for
    apply_scaler. Degenerate features (IQR of 0, e.g. a constant column) get
    a scale of 1.0 so they pass through unchanged rather than exploding.
    """
    X = np.asarray(X, dtype=np.float64)
    center = np.median(X, axis=0)
    q75, q25 = np.percentile(X, [75, 25], axis=0)
    scale = q75 - q25
    scale[scale < EPS] = 1.0
    return center, scale


def fit_standard_scaler(X: np.ndarray) -> tuple:
    """Mean/std, kept only for reproducing pre-robust-scaling results and for
    the side-by-side outlier demonstration. Not used in training."""
    X = np.asarray(X, dtype=np.float64)
    center = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < EPS] = 1.0
    return center, scale


def apply_scaler(X: np.ndarray, center: np.ndarray, scale: np.ndarray,
                 clip: float | None = STANDARDISED_CLIP) -> np.ndarray:
    """Standardise and (by default) clip. Pass clip=None only to reproduce
    an unclipped baseline."""
    z = (np.asarray(X, dtype=np.float64) - center) / scale
    return np.clip(z, -clip, clip) if clip is not None else z


def scale_from_checkpoint(X: np.ndarray, ckpt: dict) -> np.ndarray:
    """Apply exactly the transform a checkpoint was trained with. Use this in
    every inference path so training and scoring cannot drift apart."""
    return apply_scaler(X, ckpt["mu"], ckpt["sigma"], ckpt.get("clip", None))
