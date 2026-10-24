"""Distributional comparison: Wasserstein marginals + the classifier two-sample test.

Dataset-agnostic. Two disciplines are baked in because both were learned the
hard way:

1. FOV-MASK BOTH SIDES. Generated output is FOV-masked by construction; real
   radar is not (2.19% of VoD test sits outside the wedge). Comparing masked
   against unmasked biases every count, marginal and density map.

2. MATCH SAMPLE SIZES. Wasserstein and KS are sample-size biased, and the sets
   here differ by an order of magnitude. Every distance is computed at a common
   subsample size over repeated draws; full-sample values are reported alongside
   so the effect of matching is visible rather than assumed away.

And one interpretive rule: at n=60,000 the KS p-value rejects identity for the
real-vs-real null too. p-values carry no information at this scale -- only
effect sizes and the ratio against the null do.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

PASS_RATIO = 2.0          # model-vs-real / real-vs-real
N_MATCH_CAP, N_REPS = 60_000, 40


def fov_mask(azimuth_deg, fov_deg):
    return np.abs(np.asarray(azimuth_deg)) <= fov_deg


def matched_distance(a, b, n=None, reps=N_REPS, seed=0) -> dict:
    """W1 and KS at a common subsample size; mean +- sd over `reps` draws."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return dict(w1=np.nan, w1_sd=np.nan, ks=np.nan, w1_full=np.nan, n=0)
    n = int(min(n or N_MATCH_CAP, len(a), len(b), N_MATCH_CAP))
    rng = np.random.default_rng(seed)
    w, k = [], []
    for _ in range(reps):
        sa = a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]
        sb = b if len(b) <= n else b[rng.choice(len(b), n, replace=False)]
        w.append(wasserstein_distance(sa, sb)); k.append(ks_2samp(sa, sb).statistic)
    return dict(w1=float(np.mean(w)), w1_sd=float(np.std(w)), ks=float(np.mean(k)),
                w1_full=float(wasserstein_distance(a, b)), n=n)


def ratio_verdict(model_w1, null_w1, pass_ratio=PASS_RATIO):
    r = model_w1 / null_w1 if null_w1 > 0 else np.inf
    return r, ("PASS" if r <= pass_ratio else "FAIL")


def classifier_two_sample(Xa, Xb, fit_a, fit_b, seed=0):
    """AUC separating two frame-level feature sets, plus per-feature univariate AUC.

    Splits MUST be sequence-level (the caller supplies the masks). Report this
    against a real-vs-real null of the same construction: real scenes from
    different sequences are themselves separable, so an AUC near 1 means little
    without that baseline. The informative quantity is the per-feature EXCESS
    over the null -- a statistic that separates generated from real but does not
    separate one set of real sequences from another is a generator artefact.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    Xtr = np.vstack([Xa[fit_a], Xb[fit_b]])
    ytr = np.r_[np.zeros(fit_a.sum()), np.ones(fit_b.sum())]
    Xte = np.vstack([Xa[~fit_a], Xb[~fit_b]])
    yte = np.r_[np.zeros((~fit_a).sum()), np.ones((~fit_b).sum())]
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med)
    Xte = np.where(np.isfinite(Xte), Xte, med)
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    uni = np.array([max(roc_auc_score(yte, Xte[:, j]), 1 - roc_auc_score(yte, Xte[:, j]))
                    if np.std(Xte[:, j]) > 0 else 0.5 for j in range(Xte.shape[1])])
    return auc, uni
