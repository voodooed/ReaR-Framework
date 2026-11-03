"""The chart palette, and the computable checks it has to pass.

Chart colour is not a taste question — the parts that matter are measurable, so
they are measured here rather than eyeballed. This is a Python port of the
data-viz skill's `validate_palette.js` (the bundled Node script needs a newer
runtime than this box has), holding the same thresholds and the same
Machado-Oliveira-Fernandes (2009) colour-vision-deficiency transforms.

Run it directly to see the report:

    python tools/_palette.py

WHAT THE SLOTS MEAN HERE
------------------------
Every chart in `tools/plot_distributions.py` uses colour for one of three jobs,
and each job has one rule:

  IDENTITY (categorical)  real vs generated, or VoD vs KITTI-CARLA. Fixed slot
                          order, never cycled. Real is always slot 1, generated
                          always slot 2 -- across every figure, so the reader
                          learns the mapping once.
  MAGNITUDE (sequential)  one hue, light -> dark. Density maps.
  POLARITY (diverging)    two hues either side of a NEUTRAL GREY midpoint, for
                          `generated - real` differences where zero is the
                          answer you want and the sign carries meaning.

Identity is never carried by colour alone: every figure with two series also
carries a legend, and the two are separated in the mark itself (fill vs
outline, or side-by-side rather than overlaid).
"""

from __future__ import annotations

import math

# ---- the palette ------------------------------------------------------------
# Categorical slots, in fixed order. Only the first three are used; the skill's
# reference palette validates its first three all-pairs in both modes.
SERIES = {
    "real": "#2a78d6",        # slot 1, blue   -- ground truth / measurement
    "generated": "#eb6834",   # slot 2, orange -- this framework's output
    "third": "#1baf7a",       # slot 3, aqua   -- a second generated condition
}
# Sequential ramp (blue, light -> dark), for continuous magnitude.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]
# Diverging pair with a NEUTRAL grey midpoint. Blue<->red: warm/cool poles that
# read as opposite. (Blue<->aqua was rejected upstream -- both cool, so the
# midpoint does not read as "nothing".)
DIVERGING = ("#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a09f", "#e34948", "#8f1f1e")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"

# ---- thresholds, from the skill --------------------------------------------
BAND = (0.43, 0.77)     # OKLCH L, light mode
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def _srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def _lin(h):
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in _srgb(h)]


def _rel_lum(h):
    r, g, b = _lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([_rel_lum(a), _rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = _oklab_from_lin(_lin(h))
    return L, math.hypot(a, b)


def _simulate(h, kind):
    r, g, b = _lin(h)
    M = MACHADO[kind]
    return [min(1.0, max(0.0, M[i][0] * r + M[i][1] * g + M[i][2] * b))
            for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = _oklab_from_lin(_simulate(h1, kind) if kind else _lin(h1))
    b = _oklab_from_lin(_simulate(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def validate(palette=None, surface=SURFACE, pairs="all"):
    """Return (rows, ok). Hard-fails on lightness band, chroma floor, the
    normal-vision floor, and CVD below the 6.0 floor."""
    palette = palette or list(SERIES.values())
    rows, ok = [], True
    for h in palette:
        L, C = oklch(h)
        cr = contrast(h, surface)
        band = BAND[0] <= L <= BAND[1]
        chroma = C >= CHROMA_FLOOR
        ok &= band and chroma
        rows.append({"hex": h, "L": round(L, 3), "C": round(C, 3),
                     "contrast": round(cr, 2),
                     "band": "PASS" if band else "FAIL",
                     "chroma": "PASS" if chroma else "FAIL",
                     # sub-3:1 is a WARN, not a fail: legal with the relief rule
                     # (visible direct labels or a table view), which these
                     # charts satisfy via axis labels + the metrics JSON.
                     "vs surface": "PASS" if cr >= CONTRAST_MIN else "WARN"})

    idx = ([(i, i + 1) for i in range(len(palette) - 1)] if pairs == "adjacent"
           else [(i, j) for i in range(len(palette)) for j in range(i + 1, len(palette))])
    pair_rows = []
    for i, j in idx:
        a, b = palette[i], palette[j]
        cvd = min(delta_e(a, b, "protan"), delta_e(a, b, "deutan"))
        nor = delta_e(a, b)
        hard = cvd >= CVD_FLOOR and nor >= NORMAL_FLOOR
        ok &= hard
        pair_rows.append({"pair": f"{a} vs {b}",
                          "CVD dE": round(cvd, 1),
                          "normal dE": round(nor, 1),
                          "verdict": ("PASS" if cvd >= CVD_TARGET and hard
                                      else "WARN" if hard else "FAIL")})
    return rows, pair_rows, ok


if __name__ == "__main__":
    rows, pairs, ok = validate()
    print(f"{'hex':10}{'L':>7}{'C':>7}{'contr':>7}  {'band':<6}{'chroma':<8}{'vs surface'}")
    for r in rows:
        print(f"{r['hex']:10}{r['L']:>7}{r['C']:>7}{r['contrast']:>7}  "
              f"{r['band']:<6}{r['chroma']:<8}{r['vs surface']}")
    print()
    print(f"{'pair':30}{'CVD dE':>9}{'normal dE':>11}  verdict")
    for p in pairs:
        print(f"{p['pair']:30}{p['CVD dE']:>9}{p['normal dE']:>11}  {p['verdict']}")
    print(f"\nall-pairs gate (CVD>={CVD_FLOOR}, normal>={NORMAL_FLOOR}): "
          f"{'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
