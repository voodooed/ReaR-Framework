"""The null-control machinery must behave correctly on cases with a KNOWN answer.

If this instrument is wrong, every adoption decision in the project is wrong --
so it gets tested against constructed data where the right answer is not in
doubt, not only against real frames where it cannot be checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rear.geometry.association import cone_match  # noqa: E402
from rear.validation.null_control import (  # noqa: E402
    azimuth_jitter, binned_shuffle, compare, true_signal)

ORIGIN = np.zeros(3)


def _scene(n_objects=12, per_object=40, seed=0):
    """Object-like clusters inside the FOV wedge, not uniform noise.

    The clustering matters: a uniformly dense cloud makes the cone match almost
    anything, so the azimuth-jitter null saturates near the observed rate and
    the test can no longer tell signal from coincidence. Real candidates lie on
    surfaces, and that sparsity between surfaces is what gives the null control
    its discriminating power in the first place.
    """
    rng = np.random.default_rng(seed)
    az = rng.uniform(-np.pi / 3, np.pi / 3, n_objects)
    r = rng.uniform(8, 55, n_objects)
    centres = np.c_[r * np.cos(az), r * np.sin(az), rng.uniform(-1, 1.5, n_objects)]
    return (np.repeat(centres, per_object, axis=0)
            + rng.normal(0, 0.6, (n_objects * per_object, 3)))


def test_azimuth_jitter_preserves_range_and_height():
    p = _scene()
    q = azimuth_jitter(p, ORIGIN, 30.0)
    assert np.allclose(np.linalg.norm(p, axis=1), np.linalg.norm(q, axis=1))
    assert np.allclose(p[:, 2], q[:, 2])          # elevation untouched
    assert not np.allclose(p[:, :2], q[:, :2])    # angle destroyed


def test_true_signal_is_high_when_correspondence_is_real():
    cand = _scene(seed=1)
    det = cand[::5] + 0.05                         # genuinely corresponding
    r = true_signal(lambda p: cone_match(p, cand, ORIGIN, 1.0, 5.0)[0],
                    det, ORIGIN)
    assert r.observed > 95.0
    assert r.true_signal > 50.0, f"real correspondence scored only {r}"


def test_true_signal_is_near_zero_for_pure_coincidence():
    """Detections drawn independently of the candidates carry NO signal, however
    high the raw match rate gets. This is the trap the module exists to catch."""
    cand = _scene(seed=2)
    det = _scene(seed=99)[::5]                     # independent draw
    r = true_signal(lambda p: cone_match(p, cand, ORIGIN, 1.0, 5.0)[0],
                    det, ORIGIN)
    assert abs(r.true_signal) < 15.0, \
        f"coincidence scored {r.true_signal:.1f}pp of true signal: {r}"


def test_null_uses_the_worst_rotation_not_the_mean():
    cand = _scene(seed=3)
    det = cand[::5] + 0.05
    r = true_signal(lambda p: cone_match(p, cand, ORIGIN, 1.0, 5.0)[0],
                    det, ORIGIN, rotations_deg=(15.0, -15.0, 30.0, -30.0))
    by_mag = {m: np.mean([v for d, v in r.null_by_rotation.items() if abs(d) == m])
              for m in (15.0, 30.0)}
    assert r.null == max(by_mag.values())
    assert r.null >= np.mean(list(by_mag.values()))   # conservative, never flattering


def test_cluster_bootstrap_interval_is_wider_than_naive():
    """Detections in one frame share a scene. Resampling them independently
    understates the interval -- which would make every difference look real."""
    cand = _scene(seed=4)
    det = np.vstack([cand[i * 40:(i + 1) * 40:4] + 0.05 for i in range(10)])
    fids = np.repeat(np.arange(10), 10)
    r = true_signal(lambda p: cone_match(p, cand, ORIGIN, 1.0, 5.0)[0],
                    det, ORIGIN, frame_ids=fids, n_bootstrap=200)
    assert r.ci is not None and r.ci[0] < r.true_signal < r.ci[1] + 1e-9
    assert r.verdict in ("improved", "regressed", "indistinguishable from baseline")


def test_binned_shuffle_keeps_marginal_and_destroys_assignment():
    rng = np.random.default_rng(0)
    rngs = rng.uniform(5, 60, 3000)
    vals = -0.4 * rngs + rng.normal(0, 2, 3000)
    sh = binned_shuffle(vals, rngs, n_bins=6, seed=0)
    assert np.allclose(np.sort(vals), np.sort(sh))                 # marginal kept
    assert abs(np.corrcoef(rngs, sh)[0, 1] -
               np.corrcoef(rngs, vals)[0, 1]) < 0.10               # trend kept
    assert np.corrcoef(vals, sh)[0, 1] < 0.9                       # assignment gone


def test_adoption_rule_rejects_a_rate_rise_that_costs_true_signal():
    """The canonical historical failure, reproduced as a unit test:
    observed 77.0 -> 90.3 while true signal fell 45.4 -> 28.3. REJECT."""
    from rear.validation.null_control import TrueSignalResult
    base = TrueSignalResult(77.0, {}, 31.6, 45.4, 1000)
    variant = TrueSignalResult(90.3, {}, 62.0, 28.3, 1000)
    assert compare(base, variant, supply_ratio=2.9).startswith("REJECT")
    assert compare(base, base).startswith("KEEP")
