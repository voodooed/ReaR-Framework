"""Generative calibration for the RCS head.

THE PROBLEM, measured on the full training set (in-FOV):

  population                          n          mean     std    skew
  argmax-WINNING (head targets)   1,283,720    -7.507  10.270  -0.274
  ALL ASSOCIATED (target pop)       720,125   -11.509  11.634  -0.556

The head was validated as a REGRESSOR and is being used as a GENERATOR of a
population. Two distinct defects compound:

1. SELECTION. Targets are the argmax-winning detection per matched candidate.
   Strong detections win repeatedly across many candidates; weak ones never
   win, so the low tail is truncated. This is NOT a location shift: the Q-Q
   slope of winning against associated is 0.8358, not 1.0, and the gap runs
   8.37 dB at p05, 4.20 at p25, 3.13 at p50, ~3.0 above that. A constant
   -4.0 dB offset -- the fix this project has rejected before for the
   scattering-centre offset -- would over-correct the upper half and
   under-correct the weak tail, fabricating agreement in the mean while
   leaving the shape wrong.

2. DISPERSION COLLAPSE. A conditional-mean regressor cannot produce a
   population. Predicted std on VoD test is 5.12 against a target-population
   std of 11.63 -- under-dispersed by 2.3x. Retraining with importance
   weights would fix defect 1 only; the model would still emit means, so the
   generated distribution's shape would stay wrong. That is why retraining is
   NOT the chosen fix here.

THE FIX -- two composed pieces, both estimated on TRAIN only:

  a) CONDITIONAL RESAMPLING. Train candidates are binned by the head's
     predicted value (quantile bins). Each bin stores the empirical
     distribution of its winning targets. At generation a candidate draws a
     SAMPLE from its bin rather than emitting the bin's mean. This restores
     conditional variance and keeps the head's ranking of candidates, which
     is the signal it actually learned (test w-R2 0.2765).

  b) MONOTONE QUANTILE MAP, winning -> associated. Estimated from the two
     marginals above and applied after (a). Being monotone it preserves the
     ordering established by (a), and it corrects the selection effect at
     every quantile rather than on average.

Composing them, the emitted marginal is (winning distribution) pushed through
(winning -> associated), i.e. the target population, while conditional spread
survives. Both pieces are fitted on TRAIN and evaluated on held-out frames.

KNOWN LIMITATION, stated rather than hidden: (a) assumes the within-bin spread
of WINNING targets is a reasonable stand-in for the conditional spread of
associated detections. The caches pair each candidate only with its winning
detection, so the (candidate, all-associated-detections) pairing needed to
estimate that directly does not exist without rebuilding the RCS dataset.
The marginal is corrected exactly; the conditional spread is approximated.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

N_PRED_BINS = 40        # quantile bins over the head's predicted value
N_POOL_KNOTS = 256      # quantiles stored per bin, instead of raw pools
N_MAP_KNOTS = 1001      # knots of the winning -> associated quantile map


def fit_rcs_calibration(pred_train: np.ndarray, winning_train: np.ndarray,
                        associated_pop: np.ndarray) -> dict:
    """Fit the two-piece calibration.

    `pred_train` and `winning_train` are the head's predictions and their
    paired winning targets on TRAIN matched candidates. `associated_pop` is
    the RCS of ALL associated in-FOV detections -- the population the
    generator is supposed to reproduce.
    """
    pred_train = np.asarray(pred_train, dtype=np.float64)
    winning_train = np.asarray(winning_train, dtype=np.float64)
    associated_pop = np.asarray(associated_pop, dtype=np.float64)

    # (a) conditional pools, binned by predicted value
    edges = np.quantile(pred_train, np.linspace(0, 1, N_PRED_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    qs = np.linspace(0, 1, N_POOL_KNOTS)
    pools = []
    idx = np.clip(np.digitize(pred_train, edges) - 1, 0, N_PRED_BINS - 1)
    for b in range(N_PRED_BINS):
        vals = winning_train[idx == b]
        if len(vals) < 10:                     # too thin to resample honestly
            vals = winning_train
        pools.append(np.quantile(vals, qs))
    pools = np.asarray(pools)                  # (N_PRED_BINS, N_POOL_KNOTS)

    # (b) monotone quantile map, winning -> associated
    mq = np.linspace(0, 1, N_MAP_KNOTS)
    src = np.quantile(winning_train, mq)
    dst = np.quantile(associated_pop, mq)

    return {"pred_edges": edges, "pools": pools, "pool_qs": qs,
            "map_src": src, "map_dst": dst,
            "fit_n_pred": int(len(pred_train)),
            "fit_n_assoc": int(len(associated_pop))}


def apply_rcs_calibration(pred: np.ndarray, cal: dict,
                          rng: np.random.Generator) -> np.ndarray:
    """Turn conditional-mean predictions into samples from the target population.

    Seeded through `rng`, so a generated frame stays exactly reproducible.
    """
    pred = np.asarray(pred, dtype=np.float64)
    if len(pred) == 0:
        return pred
    b = np.clip(np.digitize(pred, cal["pred_edges"]) - 1, 0,
                cal["pools"].shape[0] - 1)
    # Sample each candidate from its own bin's empirical quantile function.
    u = rng.random(len(pred))
    drawn = np.array([np.interp(u[i], cal["pool_qs"], cal["pools"][b[i]])
                      for i in range(len(pred))])
    # Push through the monotone winning -> associated map.
    return np.interp(drawn, cal["map_src"], cal["map_dst"])


def load_rcs_calibration(path: Path) -> dict | None:
    """Load the calibration, or None if it has not been fitted."""
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
