"""The swap test — isolate a CHANNEL's contribution to a real detector.

Generalised from the Phase 1c / 2a / 2b harness, which was VoD-path-specific.

The idea: hold geometry EXACTLY fixed (real detections at their real
coordinates) and replace one channel's values. Any AP change is then
attributable to that channel alone. It is the sharpest instrument this project
built, and it produced the central result:

    real RCS   0.0536      shuffled RCS 0.0026      constant RCS 0.0011
    generator  0.0217*     generator v_r 0.0535
    (*tightened support: only detections with a candidate within 1.0 m)

i.e. the detector is ~98% RCS-dependent, v_r is free, and the generator's RCS
recovers a small fraction of the usable information.

REQUIREMENTS: an adapter with `has_real_radar() == True`. Without paired real
radar there is nothing to hold fixed and nothing to swap into, which is exactly
why this cannot run on KITTI-CARLA.

TWO CONTROLS ARE MANDATORY and are provided here rather than left to callers:
  * `shuffled`  real values, permuted -- the marginal is right, the assignment
                is destroyed. Bounds how much of a swap's damage is "wrong
                values" versus "any mis-assignment".
  * `constant`  a single value everywhere -- how much the detector uses the
                channel at all.
A swap result without those two is uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rear.validation.null_control import binned_shuffle

DEFAULT_NEAR_CAND_M = 1.0


@dataclass
class SwapSpec:
    """One arm: which channel, and what to put in it."""
    name: str
    channel: str                      # 'rcs' | 'v_r'
    values: np.ndarray | None = None  # None for the control kinds
    kind: str = "values"              # values | shuffled | constant | real
    meta: dict = field(default_factory=dict)


def tightened_support(real_xyz, candidate_centres, near_m=DEFAULT_NEAR_CAND_M):
    """Mask of real detections that HAVE a generator prediction nearby.

    Phase 1's swap mixed generator error with substitution noise: the RCS head
    scores candidates, so a detection with no candidate near it borrowed a value
    from up to 6.3 m away (p90). Restricting to `near_m` isolates the question
    actually being asked. Detections outside it keep their real value rather
    than being dropped -- dropping them would change the cloud's support and
    stop the arms being comparable.
    """
    from scipy.spatial import cKDTree
    if len(candidate_centres) == 0:
        return np.zeros(len(real_xyz), bool), np.zeros(len(real_xyz), int)
    d, j = cKDTree(np.asarray(candidate_centres, float)).query(
        np.asarray(real_xyz, float), k=1)
    return d <= near_m, j


def build_channel(real_values, swap_values, mask, kind="values",
                  ranges=None, seed=0):
    """The swapped channel for one arm, with the two controls implemented here."""
    out = np.asarray(real_values, dtype=np.float64).copy()
    if kind == "real":
        return out
    if kind == "constant":
        out[mask] = float(np.median(real_values))
        return out
    if kind == "shuffled":
        r = ranges if ranges is not None else np.arange(len(out))
        out[mask] = binned_shuffle(np.asarray(real_values)[mask],
                                   np.asarray(r)[mask], seed=seed)
        return out
    out[mask] = np.asarray(swap_values, dtype=np.float64)[mask]
    return out


def summarise(results: dict, baseline_key="real", metric="Car") -> str:
    """Swap arms as a table, always against the two controls."""
    base = results[baseline_key][metric]
    lines = [f"  {'arm':<34}{metric + ' AP':>9}{'vs real':>10}{'share':>9}",
             "  " + "-" * 62]
    floor = min(v[metric] for v in results.values())
    span = max(base - floor, 1e-9)
    for k, v in results.items():
        lines.append(f"  {k:<34}{v[metric]:>9.4f}{v[metric] - base:>+10.4f}"
                     f"{100 * (v[metric] - floor) / span:>8.0f}%")
    return "\n".join(lines)


def requires_real_radar(adapter):
    if not adapter.has_real_radar():
        raise RuntimeError(
            f"swap test needs paired real radar; dataset {adapter.name!r} has "
            f"none (has_real_radar() is False). This is a property of the "
            f"dataset, not a missing feature.")
