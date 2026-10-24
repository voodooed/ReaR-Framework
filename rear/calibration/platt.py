"""Monotone recalibration of the detection head's output probabilities.

Raising the candidate supply (occlusion depth <= 3) multiplied the candidate
population by 2.9x. The head's per-candidate probabilities were fitted against
the old, smaller population, so at unchanged probabilities the realised frame
now carries ~434 surface detections against real radar's ~154-181 -- a count
ratio of 1.90 against real.

WHY CALIBRATION AND NOT RETRAINING. Retraining with a loss modified to hit a
target count would optimise the model toward a downstream summary statistic,
which this project has rejected before (the constant-offset RCS "fix", the
scattering-centre offset). A post-hoc monotone map leaves the learned model
untouched and corrects only the probability scale.

THE MAP is Platt scaling in logit space,

    g(p) = sigmoid(a * logit(p) + b),

fitted on the TRAIN split only by least squares between the predicted
per-frame expected count, sum_i g(p_i), and the real per-frame count of
FOV-masked surface-explained (associated) detections. Two parameters: `a`
sharpens or flattens the probability ordering, `b` shifts the overall rate.

Being strictly monotone in p, the map CANNOT change the ranking of candidates
and therefore cannot change AUC. That is used as a correctness check on the
implementation rather than assumed -- if measured AUC moves, the map is not
being applied where it is believed to be.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

EPS = 1e-6


def _apply(p, a, b):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1.0 - p)) + b)))


def fit_detection_calibration(p, frame_idx, target_per_frame) -> dict:
    """Fit (a, b) so expected per-frame counts match the real counts.

    `p` are the head's probabilities, `frame_idx` the frame each belongs to,
    and `target_per_frame` a dict {frame_idx: real count}. Frames with no
    target are skipped rather than assumed zero.
    """
    p = np.asarray(p, dtype=np.float64)
    frames = np.asarray(frame_idx)
    uniq = np.array(sorted(set(np.unique(frames)) & set(target_per_frame)))
    if len(uniq) == 0:
        raise ValueError("no frames in common between predictions and targets")
    order = {f: k for k, f in enumerate(uniq)}
    slot = np.array([order.get(f, -1) for f in frames])
    keep = slot >= 0
    p, slot = p[keep], slot[keep]
    target = np.array([target_per_frame[f] for f in uniq], dtype=np.float64)

    def resid(theta):
        g = _apply(p, theta[0], theta[1])
        pred = np.bincount(slot, weights=g, minlength=len(uniq))
        return pred - target

    sol = least_squares(resid, x0=np.array([1.0, 0.0]), method="lm", max_nfev=200)
    a, b = float(sol.x[0]), float(sol.x[1])
    pred = np.bincount(slot, weights=_apply(p, a, b), minlength=len(uniq))
    return {"a": a, "b": b, "n_frames": int(len(uniq)),
            "fit_pred_mean": float(pred.mean()),
            "fit_target_mean": float(target.mean())}


def apply_detection_calibration(p, cal: dict | None):
    """Monotone map; returns p unchanged when no calibration is present."""
    if cal is None:
        return np.asarray(p, dtype=np.float64)
    return _apply(p, cal["a"], cal["b"])


def load_detection_calibration(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
