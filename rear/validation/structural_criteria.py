"""The 7 pre-registered structural criteria, evaluated on ANY adapter.

Thresholds are properties of real radar, so they live in the dataset config,
not here. The evaluation protocol is fixed and deliberately subset-independent:
score the WHOLE test split, both sequences. The recorded 4/7 and 5/7 figures in
the archive were measured on `range(100)` -- the first 100 frames, all from one
sequence -- and a later evenly-spaced 100 gave materially different values for
the same generator. Any before/after across two different subsets is
meaningless, which is why `n_frames=None` (all) is the default.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

DEFAULT_THRESHOLDS = {
    "count ratio": ("in", 0.85, 1.20),
    "count W":     ("le", 1.532),
    "extent W":    ("le", 0.408),
    "VMR":         ("in", 2.44, 9.74),
    "frac >=6":    ("ge", 0.096),
    "hit rate %":  ("ge", 77.0214),   # HARD; see note below
    "RCS W":       ("le", 2.7),
}
HARD_CONSTRAINT = "hit rate %"
# The hit-rate bar was historically 83.1%, taken from a single-sequence subset.
# On the full 413-frame test split REAL radar itself scores 81.12% and the
# production generator 77.02%, so the literal 83.1% rejects both the incumbent
# and reality. The operative rule is "must not regress below the incumbent",
# which preserves the intent (never buy dispersion by emptying objects) with a
# threshold that is actually attainable. Both are reported.
LITERAL_HIT_RATE_BAR = 83.1


def verdict(name, value, thresholds=None) -> str:
    r = (thresholds or DEFAULT_THRESHOLDS)[name]
    if r[0] == "in":
        return "PASS" if r[1] <= value <= r[2] else "FAIL"
    if r[0] == "le":
        return "PASS" if value <= r[1] else "FAIL"
    return "PASS" if value >= r[1] else "FAIL"


def per_object_stats(gen_pts, real_pts, boxes, in_box_fn, eval_classes):
    """Counts and extents per annotated object, for both streams."""
    cnt, ext, rcnt, rext = [], [], [], []
    for b in boxes:
        if getattr(b, "class_name", None) not in eval_classes:
            continue
        for pts, C, E in ((gen_pts, cnt, ext), (real_pts, rcnt, rext)):
            m = in_box_fn(pts, b)
            C.append(int(m.sum()))
            if m.sum() >= 2:
                q = pts[m]
                E.append(float(np.linalg.norm(q.max(axis=0) - q.min(axis=0))))
    return map(np.asarray, (cnt, ext, rcnt, rext))


def evaluate(counts_gen, counts_real, extents_gen, extents_real,
             rcs_gen, rcs_real, n_gen_per_frame, n_real_per_frame,
             hit_gen, hit_total, thresholds=None) -> dict:
    """All seven criteria from already-collected per-object/per-frame arrays."""
    c, rc = np.asarray(counts_gen, float), np.asarray(counts_real, float)
    e, re_ = np.asarray(extents_gen, float), np.asarray(extents_real, float)
    out = {
        "count ratio": float(np.mean(n_gen_per_frame) / np.mean(n_real_per_frame)),
        "count W": float(wasserstein_distance(c, rc)),
        "extent W": float(wasserstein_distance(e, re_)),
        "VMR": float(c.var() / c.mean()) if c.mean() > 0 else float("nan"),
        "frac >=6": float((c >= 6).mean()),
        "hit rate %": 100.0 * hit_gen / max(hit_total, 1),
        "RCS W": float(wasserstein_distance(np.asarray(rcs_gen),
                                            np.asarray(rcs_real))),
    }
    out["_verdicts"] = {k: verdict(k, v, thresholds) for k, v in out.items()
                        if not k.startswith("_")}
    out["_n_pass"] = sum(v == "PASS" for v in out["_verdicts"].values())
    out["_hard_ok"] = out["_verdicts"][HARD_CONSTRAINT] == "PASS"
    out["_literal_hit_rate_pass"] = out["hit rate %"] >= LITERAL_HIT_RATE_BAR
    return out
