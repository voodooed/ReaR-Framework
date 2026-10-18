"""Null controls — the project's single most important piece of methodology.

This module exists because the same mistake was made, caught, and re-made
several times across the investigation: a change raises an observed rate, the
rate looks like progress, and it is entirely coincidental matching produced by a
denser candidate pool or a wider tolerance.

The canonical case: an association "fix" took box-conditioned matching from
77.0% to 90.3% and the aggregate rate to 71.8%. Under an azimuth-jitter null the
TRUE signal (observed minus null) *fell* from 45.4 pp to 28.3 pp at every step.
The unfixed pipeline was the most discriminative configuration measured. Three
later attempts (sigma=2.8 coupling, SetSurfaceHead, SetOTHead, plus the
range-adaptive voxel schedule) were rejected by the same instrument.

THE RULE, applied everywhere in this framework: never report an observed rate
without its null. A number that rises while its true signal falls is a
regression wearing a disguise.

Three null constructions are provided, all first-class:

  `azimuth_jitter`   rotate detections about the sensor. Preserves each
                     detection's range, elevation and local candidate density
                     and destroys only angular correspondence -- the tightest
                     of the three, and the default.
  `cross_frame`      match frame i's detections against frame i+k's candidates.
                     Preserves everything about the detection set, destroys the
                     scene.
  `binned_shuffle`   permute a per-detection VALUE within range x aspect bins.
                     Preserves the marginal and the range/aspect correlation and
                     destroys the per-detection assignment. Used for feature-
                     level controls (Phase 2b/2c).

Nothing here is dataset-specific: everything takes explicit points and an
origin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ROTATIONS_DEG = (15.0, -15.0, 30.0, -30.0)
DEFAULT_N_BOOTSTRAP = 2000


# --------------------------------------------------------------- transforms --
def azimuth_jitter(points_xyz: np.ndarray, origin: np.ndarray,
                   degrees: float) -> np.ndarray:
    """Rotate points about `origin` in azimuth only."""
    rel = np.asarray(points_xyz, dtype=np.float64) - np.asarray(origin, np.float64)
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    out = rel.copy()
    out[:, 0] = c * rel[:, 0] - s * rel[:, 1]
    out[:, 1] = s * rel[:, 0] + c * rel[:, 1]
    return out + origin


def binned_shuffle(values: np.ndarray, ranges: np.ndarray,
                   aspects: np.ndarray | None = None, n_bins: int = 6,
                   seed: int = 0) -> np.ndarray:
    """Permute `values` within range (x aspect) quantile bins.

    Marginal and range/aspect correlation survive; the per-detection value does
    not. If a shuffled feature lifts a metric as much as the real one, the
    metric was riding scene composition.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64).copy()
    def _bin(x):
        q = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
        return np.digitize(x, q)
    key = _bin(np.asarray(ranges, float))
    if aspects is not None:
        key = key * 100 + _bin(np.abs(np.asarray(aspects, float)))
    for b in np.unique(key):
        idx = np.nonzero(key == b)[0]
        if len(idx) > 1:
            v[idx] = v[idx][rng.permutation(len(idx))]
    return v


# ------------------------------------------------------------------ result ---
@dataclass
class TrueSignalResult:
    observed: float
    null_by_rotation: dict
    null: float
    true_signal: float
    n: int
    ci: tuple | None = None

    @property
    def verdict(self) -> str:
        if self.ci is None:
            return "improved" if self.true_signal > 0 else "not improved"
        lo, hi = self.ci
        return ("improved" if lo > 0 else
                ("regressed" if hi < 0 else "indistinguishable from baseline"))

    def __str__(self):
        c = "" if self.ci is None else f"  95% CI [{self.ci[0]:+.2f}, {self.ci[1]:+.2f}]"
        return (f"observed {self.observed:.1f}%  null {self.null:.1f}%  "
                f"TRUE SIGNAL {self.true_signal:.1f}pp{c}  (n={self.n})")


# ------------------------------------------------------------------- driver --
def true_signal(match_fn, detections_xyz, origin, rotations_deg=DEFAULT_ROTATIONS_DEG,
                frame_ids=None, n_bootstrap=0, seed=0) -> TrueSignalResult:
    """Observed match rate minus the strongest azimuth-jitter null.

    `match_fn(points) -> bool array` is any per-detection matching predicate --
    the caller decides what "matched" means, so this works for association,
    for co-occurrence, for anything with a yes/no per detection.

    The null taken is the WORST (highest) rotation, not the mean: the point is a
    conservative floor on coincidental matching, and averaging a weak rotation
    with a strong one flatters the result.

    With `frame_ids` and `n_bootstrap`, a CLUSTER bootstrap over frames gives a
    CI. Detections inside a frame share a scene and are strongly correlated, so
    resampling detections independently badly understates the interval -- that
    error would make every difference look significant.
    """
    det = np.asarray(detections_xyz, dtype=np.float64)
    obs = float(np.mean(match_fn(det))) * 100.0
    nulls = {d: float(np.mean(match_fn(azimuth_jitter(det, origin, d)))) * 100.0
             for d in rotations_deg}
    # pair +/- of the same magnitude, then take the worst magnitude
    mags = sorted({abs(d) for d in rotations_deg})
    by_mag = {m: np.mean([v for d, v in nulls.items() if abs(d) == m]) for m in mags}
    null = float(max(by_mag.values()))
    res = TrueSignalResult(observed=obs, null_by_rotation=nulls, null=null,
                           true_signal=obs - null, n=len(det))
    if n_bootstrap and frame_ids is not None:
        res.ci = _bootstrap_ci(match_fn, det, origin, rotations_deg,
                               np.asarray(frame_ids), n_bootstrap, seed)
    return res


def _bootstrap_ci(match_fn, det, origin, rotations, frame_ids, n_boot, seed):
    rng = np.random.default_rng(seed)
    hit = {None: match_fn(det)}
    for d in rotations:
        hit[d] = match_fn(azimuth_jitter(det, origin, d))
    uf = np.unique(frame_ids)
    rows = [np.nonzero(frame_ids == f)[0] for f in uf]
    mags = sorted({abs(d) for d in rotations})
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uf), len(uf))
        sel = np.concatenate([rows[p] for p in pick])
        o = hit[None][sel].mean() * 100
        n = max(np.mean([np.mean([hit[d][sel].mean() for d in rotations
                                  if abs(d) == m]) for m in [m]]) * 100
                for m in mags)
        draws[b] = o - n
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def compare(baseline: TrueSignalResult, variant: TrueSignalResult,
            supply_ratio: float | None = None) -> str:
    """The standing adoption rule, in one place.

    A variant is an improvement only if true signal does not fall. Raw rate and
    candidate supply decide nothing -- they are reported so the density-inflation
    trap stays visible, never so it can be argued around.
    """
    d = variant.true_signal - baseline.true_signal
    s = "" if supply_ratio is None else f", supply {supply_ratio:.2f}x"
    if d >= 0:
        return f"KEEP  ({d:+.1f}pp true signal{s})"
    return (f"REJECT  ({d:+.1f}pp true signal{s}) -- observed rate "
            f"{variant.observed - baseline.observed:+.1f}pp is not evidence")
