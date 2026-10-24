"""Clutter process -- Component 2 of the two-component radar model.

Component 1 (the surface process: detection + RCS heads) explains radar
returns that correspond to LiDAR-visible geometry. It structurally cannot
explain the rest: multipath ghosts, ground bounce and CFAR false alarms
arise from propagation physics and detector thresholds, not from a surface
in the candidate grid. Those detections are this component's job.

Per the research plan this component is fit ENTIRELY from real VoD data and
carried into the simulated domain as a fitted statistical prior. There is no
geometric transfer question here -- clutter is not expected to transfer as a
function of scene geometry, and this module makes no such claim.

Two pieces:

  lambda(range, azimuth)  -- an inhomogeneous spatial intensity, fit by 2D
      kernel density estimation over real clutter detections. Inhomogeneous
      because clutter is emphatically not uniform: ground bounce and
      multipath have range- and angle-dependent structure.

  RCS and v_r              -- NOT fit parametrically. Real clutter RCS is
      skewed and heavy-tailed, and a fitted Gaussian/Gamma would smooth away
      exactly the tails that make clutter look like clutter. Instead they are
      drawn by STRATIFIED EMPIRICAL RESAMPLING: bin real clutter by range,
      and sample a synthetic point's RCS/v_r from the real pool in its own
      range bin. That preserves the real marginal shape and the real
      range-conditional structure without assuming a form.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import gaussian_kde

DEFAULT_RANGE_BIN_M = 5.0
MIN_SAMPLES_PER_BIN = 30    # below this a bin is merged with its neighbour


@dataclass
class ClutterProcess:
    """Fitted clutter model. Built by `fit_clutter_process`."""

    kde: gaussian_kde                 # over (range, azimuth_deg)
    rate_mean: float                  # mean clutter detections per frame
    rate_std: float
    bin_edges: np.ndarray             # range-bin edges after merging
    bin_rcs: list = field(default_factory=list)   # per-bin empirical RCS pools
    bin_vr: list = field(default_factory=list)    # per-bin empirical v_r pools
    range_limits: tuple = (0.0, 100.0)
    azimuth_limits: tuple = (-60.0, 60.0)
    # Ego-speed conditioning of the RATE only (a multiplicative factor on
    # lambda, not a change to the spatial shape). Populated only when the
    # measured dependence clears the pre-registered evidence threshold --
    # see fit_clutter_process. None means the model is unconditioned.
    speed_bin_edges: np.ndarray | None = None
    speed_rate_factor: np.ndarray | None = None

    def rate_for_speed(self, ego_speed: float | None) -> float:
        """Expected clutter count for a frame, optionally conditioned on ego
        speed. Falls back to the unconditional rate when no conditioning was
        fitted or no speed is supplied."""
        if self.speed_bin_edges is None or ego_speed is None:
            return self.rate_mean
        b = int(np.clip(np.digitize(ego_speed, self.speed_bin_edges) - 1,
                        0, len(self.speed_rate_factor) - 1))
        return self.rate_mean * float(self.speed_rate_factor[b])

    def sample_frame(self, rng: np.random.Generator, n_points: int | None = None,
                     ego_speed: float | None = None):
        """Sample one frame's clutter: (range, azimuth_deg, rcs, v_r).

        Count is Poisson about the fitted per-frame rate unless `n_points` is
        given -- clutter is a point process, so the count is itself random.
        """
        if n_points is None:
            n_points = int(rng.poisson(self.rate_for_speed(ego_speed)))
        if n_points <= 0:
            empty = np.zeros(0)
            return empty, empty, empty, empty

        # Rejection-sample the KDE back into the physically valid window;
        # a Gaussian KDE can otherwise place mass at negative range or
        # outside the sensor's azimuth coverage.
        keep_r, keep_a = [], []
        while len(keep_r) < n_points:
            draw = self.kde.resample(max(n_points * 2, 64), seed=int(rng.integers(1 << 31)))
            r, a = draw[0], draw[1]
            ok = ((r >= self.range_limits[0]) & (r <= self.range_limits[1])
                  & (a >= self.azimuth_limits[0]) & (a <= self.azimuth_limits[1]))
            keep_r.extend(r[ok].tolist()); keep_a.extend(a[ok].tolist())
        rng_out = np.array(keep_r[:n_points])
        az_out = np.array(keep_a[:n_points])

        idx = np.clip(np.digitize(rng_out, self.bin_edges) - 1, 0, len(self.bin_rcs) - 1)
        rcs_out = np.array([rng.choice(self.bin_rcs[b]) for b in idx])
        vr_out = np.array([rng.choice(self.bin_vr[b]) for b in idx])
        return rng_out, az_out, rcs_out, vr_out


def fit_clutter_process(
    clutter_range: np.ndarray,
    clutter_azimuth: np.ndarray,
    clutter_rcs: np.ndarray,
    clutter_vr: np.ndarray,
    per_frame_counts: np.ndarray,
    range_bin_m: float = DEFAULT_RANGE_BIN_M,
    min_per_bin: int = MIN_SAMPLES_PER_BIN,
    azimuth_limits: tuple = (-60.0, 60.0),
    per_frame_ego_speed: np.ndarray | None = None,
    speed_conditioning: bool = False,
    n_speed_bins: int = 5,
) -> tuple:
    """Fit the intensity KDE and build the stratified resampling pools.

    Returns (ClutterProcess, report dict describing the binning).
    """
    lo, hi = float(clutter_range.min()), float(clutter_range.max())
    kde = gaussian_kde(np.vstack([clutter_range, clutter_azimuth]))

    # Range bins, merging any bin too sparse to resample from honestly.
    raw_edges = np.arange(np.floor(lo), np.ceil(hi) + range_bin_m, range_bin_m)
    edges, pools_rcs, pools_vr, merged = [raw_edges[0]], [], [], 0
    cur_rcs: list = []
    cur_vr: list = []
    for a, b in zip(raw_edges[:-1], raw_edges[1:]):
        m = (clutter_range >= a) & (clutter_range < b)
        cur_rcs.extend(clutter_rcs[m].tolist())
        cur_vr.extend(clutter_vr[m].tolist())
        if len(cur_rcs) >= min_per_bin:
            edges.append(b); pools_rcs.append(np.array(cur_rcs)); pools_vr.append(np.array(cur_vr))
            cur_rcs, cur_vr = [], []
        else:
            merged += 1
    if cur_rcs:      # tail remainder folds into the last bin
        if pools_rcs:
            pools_rcs[-1] = np.concatenate([pools_rcs[-1], np.array(cur_rcs)])
            pools_vr[-1] = np.concatenate([pools_vr[-1], np.array(cur_vr)])
            edges[-1] = raw_edges[-1]
        else:
            edges.append(raw_edges[-1]); pools_rcs.append(np.array(cur_rcs))
            pools_vr.append(np.array(cur_vr))

    # Ego-speed conditioning of the RATE, fitted only when the caller says the
    # measured dependence cleared the evidence threshold. Quantile bins keep
    # every bin populated regardless of how speed is distributed.
    speed_edges = speed_factor = None
    if speed_conditioning and per_frame_ego_speed is not None:
        q = np.linspace(0, 100, n_speed_bins + 1)
        speed_edges = np.unique(np.percentile(per_frame_ego_speed, q))
        idx = np.clip(np.digitize(per_frame_ego_speed, speed_edges) - 1,
                      0, len(speed_edges) - 2)
        overall = float(np.mean(per_frame_counts))
        speed_factor = np.array([
            float(np.mean(per_frame_counts[idx == b]) / overall) if np.any(idx == b) else 1.0
            for b in range(len(speed_edges) - 1)])

    process = ClutterProcess(
        kde=kde,
        rate_mean=float(np.mean(per_frame_counts)),
        rate_std=float(np.std(per_frame_counts)),
        bin_edges=np.array(edges),
        bin_rcs=pools_rcs, bin_vr=pools_vr,
        range_limits=(max(0.0, lo), hi),
        azimuth_limits=azimuth_limits,
        speed_bin_edges=speed_edges, speed_rate_factor=speed_factor,
    )
    report = {
        "n_bins": len(pools_rcs),
        "bins_merged": merged,
        "bin_sizes": [len(p) for p in pools_rcs],
        "min_bin_size": min(len(p) for p in pools_rcs) if pools_rcs else 0,
        "edges": np.array(edges),
        "speed_conditioned": speed_factor is not None,
        "speed_edges": speed_edges,
        "speed_factor": speed_factor,
    }
    return process, report


# `weights/clutter_process.pkl` was pickled back when this class lived at
# `physics.clutter_process.ClutterProcess` (the pre-refactor layout). Pickle
# resolves classes by that recorded module path, not by re-importing content,
# so unpickling needs `physics.clutter_process` to exist and expose the same
# class object. Rather than ship the old `physics/` package, alias it here.
import sys as _sys                                                    # noqa: E402
_sys.modules.setdefault("physics", _sys.modules[__name__])
_sys.modules["physics.clutter_process"] = _sys.modules[__name__]
