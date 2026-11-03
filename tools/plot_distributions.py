#!/usr/bin/env python
"""Distributional charts: what the generated radar looks like against the real thing.

    python tools/plot_distributions.py --head prism
    python tools/plot_distributions.py --head mlp

Writes PNGs to `output/<head>/quantitative/distribution/`.

WHY CHARTS AND NOT MORE NUMBERS. `metrics.json` already reports a Wasserstein
distance and a pass/fail per marginal. A distance says HOW FAR two distributions
are; it does not say in WHICH DIRECTION or WHERE. A 2.5 dBsm RCS gap could be a
uniform offset, a missing tail, or a bimodality the generator flattened -- three
different bugs with three different fixes, and the same number. These figures
show the shape, so the failure mode is identifiable rather than merely detected.

WHAT IS BEING COMPARED, IN EACH CASE
------------------------------------
Real vs generated on VoD is a PAIRED comparison: the same 413 frames, the same
scenes, real radar on one side and this framework's output on the other. That is
the only place ground truth exists.

VoD vs KITTI-CARLA is a different question -- there is no real radar on the
simulated side, so nothing can be scored there. Those figures show the COVARIATE
SHIFT the deployment actually faces, which is the honest thing to plot when the
answer is unavailable.

EVERYTHING HERE READS CACHED DATA. The collection `evaluate_quantitative.py`
wrote is reused, so re-plotting after a chart change costs seconds and cannot
disagree with the metrics file it sits beside.

COLOUR. Three jobs, three rules (tools/_palette.py, which computes its own
checks rather than asserting them): identity -> fixed categorical slots, real
always slot 1 and generated always slot 2 across every figure; magnitude -> one
hue light-to-dark; polarity -> a diverging pair either side of a neutral grey,
used only for `generated - real` where zero is the wanted answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from tools._palette import (DIVERGING, GRID, INK, INK_SOFT, SEQUENTIAL,  # noqa: E402
                            SERIES, SURFACE)

REAL, GEN, THIRD = SERIES["real"], SERIES["generated"], SERIES["third"]
SEQ = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
DIV = LinearSegmentedColormap.from_list("div", DIVERGING)

# Range bands used throughout, matching the conditional RCS tests in
# metrics.json so a chart and its number refer to the same slice.
RANGE_BINS = [(0, 15), (15, 30), (30, 50), (50, 75), (75, 110)]
EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")


# ------------------------------------------------------------------ styling ---
def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 9.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 0.9,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.size": 9.5, "figure.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def finish(fig, path, caption=None):
    """One caption line under every figure. A chart in a results directory is
    read without its generating code beside it, so what it shows and what to
    conclude have to travel with the image."""
    if caption:
        fig.text(0.008, 0.006, caption, ha="left", va="bottom",
                 fontsize=8, color=INK_SOFT, wrap=True)
        if fig.get_layout_engine() is None:
            fig.subplots_adjust(bottom=0.155)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  wrote {path.name}")


def step_hist(ax, real, gen, bins, xlabel, *, logy=False, density=True):
    """Two distributions as OUTLINES, not filled bars.

    Overlaid translucent fills are the classic two-histogram mistake: the
    overlap region becomes a third colour that means nothing, and neither
    distribution can be read where they cross. Outlines stay legible.
    """
    ax.hist(real, bins=bins, density=density, histtype="step",
            color=REAL, linewidth=2.0, label="real radar")
    ax.hist(gen, bins=bins, density=density, histtype="step",
            color=GEN, linewidth=2.0, label="generated")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density" if density else "count")
    if logy:
        ax.set_yscale("log")


def annotate(ax, text, loc="upper right"):
    ax.text(0.97 if "right" in loc else 0.03, 0.96,
            text, transform=ax.transAxes, ha="right" if "right" in loc else "left",
            va="top", fontsize=8.5, color=INK_SOFT,
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, lw=0.8))


def w1(a, b, cap=40000, seed=0):
    """Wasserstein distance at a common subsample size.

    W1 is sample-size biased and these two sets differ in size, so both sides
    are drawn down to the same n before comparing -- the same convention
    evaluate_quantitative.py uses, so the figure and the metrics agree.
    """
    rng = np.random.default_rng(seed)
    a = rng.choice(a, min(len(a), cap), replace=False) if len(a) > cap else a
    b = rng.choice(b, min(len(b), cap), replace=False) if len(b) > cap else b
    return float(wasserstein_distance(a, b))


# --------------------------------------------------------------------- data ---
def load_collection(head, out_dir):
    hits = sorted(out_dir.glob(f"_collected_*_{head}.npz"))
    if not hits:
        raise SystemExit(
            f"no cached collection for head={head!r} in {out_dir}. Run\n"
            f"  python tools/evaluate_quantitative.py --dataset vod --split test "
            f"--head {head} --out {out_dir}")
    z = np.load(hits[-1], allow_pickle=True)
    return {k: z[k] for k in z.files}


def in_boxes(xyz, boxes, margin=0.5):
    """Mask of points inside any box (axis-aligned + margin, as the project's
    other per-object measurements use)."""
    m = np.zeros(len(xyz), dtype=bool)
    for b in boxes:
        x, y, z_, l, w, h = b[:6]
        m |= ((np.abs(xyz[:, 0] - x) <= l / 2 + margin) &
              (np.abs(xyz[:, 1] - y) <= w / 2 + margin) &
              (np.abs(xyz[:, 2] - z_) <= h / 2 + margin))
    return m



def density_triptych(xb, yb, real, gen, *, xlabel, ylabel, title, caption, path,
                     labels=("real radar", "generated"), equal=False):
    """Two densities on ONE shared colour scale, plus their difference.

    THE SHARED SCALE IS NOT COSMETIC. Two sequential maps side by side with
    independent scales look comparable and are not: the same ink means a
    different number in each, so a reader comparing them reaches a conclusion
    the figure never supported. One scale for the pair; the difference panel
    gets its own diverging scale because it answers a different question.

    The colour bars sit in their own gridspec row rather than being stolen from
    the panels' space -- automatic placement put them through the neighbouring
    panel's axis labels.
    """
    fig = plt.figure(figsize=(15.5, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.045], hspace=0.42,
                          wspace=0.30, left=0.05, right=0.985,
                          top=0.87, bottom=0.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    hr, _, _ = np.histogram2d(real[0], real[1], bins=[xb, yb])
    hg, _, _ = np.histogram2d(gen[0], gen[1], bins=[xb, yb])
    hr, hg = hr / hr.sum(), hg / hg.sum()
    vmax = np.percentile(np.concatenate([hr[hr > 0], hg[hg > 0]]), 99)
    for ax, h, lab in zip(axes[:2], (hr, hg), labels):
        im = ax.pcolormesh(xb, yb, h.T, cmap=SEQ, shading="auto", vmin=0, vmax=vmax)
        ax.set_title(lab)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(False)
        if equal:
            ax.set_aspect("equal")

    d = hg - hr
    lim = np.percentile(np.abs(d), 99.5)
    im2 = axes[2].pcolormesh(xb, yb, d.T, cmap=DIV, shading="auto",
                             norm=TwoSlopeNorm(0, -lim, lim))
    axes[2].set_title(f"{labels[1]} − {labels[0]}")
    axes[2].set_xlabel(xlabel); axes[2].set_ylabel(ylabel); axes[2].grid(False)
    if equal:
        axes[2].set_aspect("equal")

    cax1 = fig.add_subplot(gs[1, 0:2])
    cb = fig.colorbar(im, cax=cax1, orientation="horizontal")
    cb.set_label("fraction of detections — ONE scale across both panels", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cax2 = fig.add_subplot(gs[1, 2])
    cb2 = fig.colorbar(im2, cax=cax2, orientation="horizontal")
    cb2.set_label("density difference (diverging, neutral at 0)", fontsize=8)
    cb2.ax.tick_params(labelsize=7.5)
    cb2.locator = matplotlib.ticker.MaxNLocator(5)
    cb2.update_ticks()

    fig.suptitle(title, fontsize=12.5, color=INK, y=0.965)
    fig.text(0.05, 0.035, caption, ha="left", va="top", fontsize=8, color=INK_SOFT)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ------------------------------------------------------------------ figures ---
def fig_rcs(C, out, head):
    """RCS is the channel the downstream detector is ~98% dependent on, so it
    gets the most attention."""
    r, g = C["real_rcs"].astype(float), C["gen_rcs"].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    bins = np.linspace(-45, 30, 80)
    step_hist(axes[0], r, g, bins, "RCS (dBsm)")
    axes[0].set_title("RCS distribution")
    axes[0].legend(loc="upper left")
    annotate(axes[0], f"W1 = {w1(r, g):.2f} dBsm\n"
                      f"real  μ {r.mean():+.1f}  σ {r.std():.1f}\n"
                      f"gen   μ {g.mean():+.1f}  σ {g.std():.1f}")

    for d, c, lab in ((r, REAL, "real radar"), (g, GEN, "generated")):
        xs = np.sort(d)
        axes[1].plot(xs, np.linspace(0, 1, len(xs)), color=c, lw=2.0, label=lab)
    axes[1].set_xlabel("RCS (dBsm)"); axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Cumulative distribution")
    axes[1].legend(loc="upper left")
    axes[1].set_xlim(-45, 30)

    # Q-Q: the shape comparison a histogram cannot make precise. Deviation from
    # the diagonal localises WHICH quantiles disagree.
    q = np.linspace(0.5, 99.5, 199)
    axes[2].plot([-45, 30], [-45, 30], color=INK_SOFT, lw=1.0, ls="--",
                 label="perfect agreement")
    axes[2].plot(np.percentile(r, q), np.percentile(g, q), color=GEN, lw=2.2,
                 label="generated vs real")
    axes[2].set_xlabel("real RCS quantile (dBsm)")
    axes[2].set_ylabel("generated RCS quantile (dBsm)")
    axes[2].set_title("Q–Q: which quantiles disagree")
    axes[2].legend(loc="upper left")
    axes[2].set_xlim(-45, 30); axes[2].set_ylim(-45, 30)
    axes[2].set_aspect("equal", adjustable="box")

    fig.suptitle(f"RCS — real vs generated · {head.upper()} · 413 VoD test frames",
                 fontsize=12.5, color=INK, y=1.02)
    finish(fig, out / "01_rcs_distribution.png",
           "The RCS channel carries almost all of the downstream signal. The Q–Q panel is the "
           "diagnostic one: points on the dashed line mean the generated marginal matches real "
           "at that quantile; departures localise the disagreement to a part of the range.")


def fig_rcs_by_range(C, out, head):
    r_rcs, r_rng = C["real_rcs"].astype(float), C["real_rng"]
    g_rcs, g_rng = C["gen_rcs"].astype(float), C["gen_rng"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(bottom=0.24, top=0.84, wspace=0.24, left=0.06, right=0.985)

    pos, labels, stats = [], [], []
    for k, (lo, hi) in enumerate(RANGE_BINS):
        rr = r_rcs[(r_rng >= lo) & (r_rng < hi)]
        gg = g_rcs[(g_rng >= lo) & (g_rng < hi)]
        if len(rr) < 50 or len(gg) < 50:
            continue
        for j, (d, c) in enumerate(((rr, REAL), (gg, GEN))):
            bp = axes[0].boxplot([d], positions=[k * 3 + j], widths=0.85,
                                 showfliers=False, patch_artist=True,
                                 medianprops=dict(color=SURFACE, lw=1.8),
                                 whiskerprops=dict(color=c, lw=1.2),
                                 capprops=dict(color=c, lw=1.2))
            for b in bp["boxes"]:
                b.set(facecolor=c, edgecolor=c, linewidth=0)
        pos.append(k * 3 + 0.5); labels.append(f"{lo}–{hi}")
        stats.append((f"{lo}–{hi} m", w1(rr, gg), rr.mean(), gg.mean()))
    axes[0].set_xticks(pos); axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("range from radar (m)"); axes[0].set_ylabel("RCS (dBsm)")
    axes[0].set_title("RCS by range band")
    axes[0].legend(handles=[plt.Line2D([], [], color=REAL, lw=7, label="real radar"),
                            plt.Line2D([], [], color=GEN, lw=7, label="generated")],
                   loc="lower right")

    # A DUMBBELL, not bars. Mean RCS is negative everywhere, so bars hang down
    # from zero and their length encodes distance-from-zero rather than the
    # thing being compared. Here the CONNECTOR is the comparison: its length is
    # the gap between real and generated, read directly off the axis.
    x = np.arange(len(stats))
    for i, sdat in enumerate(stats):
        axes[1].plot([sdat[2], sdat[3]], [i, i], color=INK_SOFT, lw=1.4, zorder=1,
                     alpha=0.55)
    axes[1].scatter([s_[2] for s_ in stats], x, s=90, color=REAL, zorder=3,
                    label="real radar")
    axes[1].scatter([s_[3] for s_ in stats], x, s=90, color=GEN, zorder=3,
                    label="generated")
    for i, sdat in enumerate(stats):
        axes[1].text(max(sdat[2], sdat[3]) + 0.7, i, f"W1 {sdat[1]:.2f}",
                     va="center", ha="left", fontsize=8, color=INK_SOFT)
    axes[1].set_yticks(x); axes[1].set_yticklabels([s_[0] for s_ in stats])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean RCS (dBsm)"); axes[1].set_ylabel("range band")
    axes[1].set_title("Mean RCS per band — the gap IS the error")
    axes[1].set_xlim(min(s_[2] for s_ in stats) - 2.5,
                     max(max(s_[2], s_[3]) for s_ in stats) + 6.5)
    axes[1].legend(loc="upper right")

    fig.suptitle(f"RCS conditioned on range · {head.upper()}", fontsize=12.5,
                 color=INK, y=0.965)
    finish(fig, out / "02_rcs_by_range.png",
           "A matching overall RCS marginal can still hide a wrong range dependence. These are the "
           "same bands metrics.json scores conditionally, so a failing band here is the same "
           "failure the numbers report.")


def fig_rcs_range_density(C, out, head):
    density_triptych(
        np.linspace(0, 105, 90), np.linspace(-45, 25, 80),
        (C["real_rng"], C["real_rcs"].astype(float)),
        (C["gen_rng"], C["gen_rcs"].astype(float)),
        xlabel="range (m)", ylabel="RCS (dBsm)",
        title=f"Where in (range, RCS) the two clouds differ · {head.upper()}",
        caption="Diverging scale, neutral at zero: blue = too few generated detections in that cell, "
                "red = too many, grey = right. The two left panels share one colour scale so they "
                "can be compared directly.",
        path=out / "03_rcs_vs_range_density.png")


def fig_spatial(C, out, head):
    density_triptych(
        np.linspace(-10, 100, 120), np.linspace(-45, 45, 100),
        (C["real_xyz"][:, 0], C["real_xyz"][:, 1]),
        (C["gen_xyz"][:, 0], C["gen_xyz"][:, 1]),
        xlabel="x, forward (m)", ylabel="y, left (m)", equal=True,
        title=f"Spatial distribution in the radar's own frame · {head.upper()}",
        caption="Bird's-eye view, all 413 frames stacked, both sides FOV-masked at 60° so any "
                "structure here is a real disagreement rather than a masking artefact.",
        path=out / "04_spatial_bev.png")


def fig_marginals(C, out, head):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    fig.subplots_adjust(bottom=0.13, top=0.90, hspace=0.35, wspace=0.20,
                        left=0.075, right=0.98)
    panels = [
        ("rng", np.linspace(0, 105, 70), "range (m)", "Range"),
        ("az", np.linspace(-62, 62, 70), "azimuth (deg)", "Azimuth"),
        ("el", np.linspace(-25, 20, 70), "elevation (deg)", "Elevation"),
        ("vr", np.linspace(-22, 12, 70), "radial velocity (m/s)", "v_r"),
    ]
    for ax, (key, bins, xlabel, title) in zip(axes.ravel(), panels):
        r = C[f"real_{key}"].astype(float)
        g = C[f"gen_{key}"].astype(float)
        step_hist(ax, r, g, bins, xlabel)
        ax.set_title(f"{title}   (W1 {w1(r, g):.3f})")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"The four output marginals · {head.upper()}", fontsize=12.5,
                 color=INK, y=0.965)
    finish(fig, out / "05_marginals.png",
           "v_r is COMPUTED from ego motion rather than predicted, so its agreement is a check on "
           "the geometry, not a learned result. Elevation is the weakest of the four: clutter "
           "elevation is an empirical per-range pool that never went through the null control.")


def fig_counts(C, out, head):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))
    fig.subplots_adjust(bottom=0.24, top=0.84, wspace=0.28, left=0.055, right=0.985)
    rn, gn = C["real_n_per_frame"].astype(float), C["gen_n_per_frame"].astype(float)

    bins = np.linspace(0, max(rn.max(), gn.max()) * 1.02, 45)
    step_hist(axes[0], rn, gn, bins, "detections per frame")
    axes[0].set_title("Detections per frame")
    # Legend left, statistics right: the default corner for both is the same one.
    axes[0].legend(loc="upper left")
    annotate(axes[0], f"real  μ {rn.mean():.0f}\ngen   μ {gn.mean():.0f}\n"
                      f"ratio  {gn.mean() / rn.mean():.3f}")

    # Per-object counts: the supply side of whether a detector can fit a box.
    bframe, bpar = C["meta_box_frame"], C["meta_box_par"]
    per = {"real": [], "gen": []}
    for key in ("real", "gen"):
        xyz, fr = C[f"{key}_xyz"], C[f"{key}_frame"]
        for f in np.unique(bframe):
            bs = bpar[bframe == f]
            pts = xyz[fr == f]
            if not len(pts):
                per[key] += [0] * len(bs)
                continue
            for b in bs:
                per[key].append(int(in_boxes(pts, [b]).sum()))
    pr = np.array(per["real"], float); pg = np.array(per["gen"], float)
    b2 = np.arange(0, 26) - 0.5
    step_hist(axes[1], np.clip(pr, 0, 25), np.clip(pg, 0, 25), b2,
              "detections on an annotated object", density=True)
    axes[1].set_title("Detections per object (clipped at 25)")
    axes[1].legend(loc="upper right")

    # Variance-to-mean ratio: real radar is strongly over-dispersed. A
    # Poisson-like ~1 means objects are too uniform -- the failure mode where
    # per-object counts "improve" by being flattened.
    vmr = lambda v: v.var() / max(v.mean(), 1e-9)
    hit = lambda v: 100 * (v > 0).mean()
    labels = ["mean pts\nper object", "VMR\n(over-dispersion)", "hit rate %\n(≥1 detection)"]
    rv = [pr.mean(), vmr(pr), hit(pr)]
    gv = [pg.mean(), vmr(pg), hit(pg)]
    x = np.arange(3)
    axes[2].bar(x - 0.19, rv, 0.36, color=REAL, label="real radar")
    axes[2].bar(x + 0.19, gv, 0.36, color=GEN, label="generated")
    for i, (a, b) in enumerate(zip(rv, gv)):
        axes[2].text(i - 0.19, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8, color=INK_SOFT)
        axes[2].text(i + 0.19, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8, color=INK_SOFT)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_yscale("log"); axes[2].set_ylabel("value (log scale)")
    axes[2].set_title("Per-object supply")
    axes[2].legend(loc="upper left")

    fig.suptitle(f"How many detections, and where they land · {head.upper()}",
                 fontsize=12.5, color=INK, y=0.965)
    finish(fig, out / "06_point_counts.png",
           "Object hit rate is the project's HARD constraint: per-object dispersion can always be "
           "'improved' by emptying objects, so the hit rate is what stops that trade.")


def fig_object_wise(C, out, head):
    bframe, bcls, bpar = C["meta_box_frame"], C["meta_box_cls"], C["meta_box_par"]
    rows = {c: {"real": [], "gen": []} for c in EVAL_CLASSES}
    rcs_on = {c: {"real": [], "gen": []} for c in EVAL_CLASSES}
    for key in ("real", "gen"):
        xyz, fr = C[f"{key}_xyz"], C[f"{key}_frame"]
        rcs = C[f"{key}_rcs"].astype(float)
        for f in np.unique(bframe):
            sel = bframe == f
            pts, pr = xyz[fr == f], rcs[fr == f]
            for b, c in zip(bpar[sel], bcls[sel]):
                if c not in EVAL_CLASSES:
                    continue
                if not len(pts):
                    rows[c][key].append(0)
                    continue
                m = in_boxes(pts, [b])
                rows[c][key].append(int(m.sum()))
                if m.any():
                    rcs_on[c][key].append(pr[m])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(bottom=0.26, top=0.78, wspace=0.26, left=0.055, right=0.985)
    x = np.arange(len(EVAL_CLASSES))
    for j, (stat, title, fn) in enumerate([
            ("hit", "Object hit rate", lambda v: 100 * (np.array(v) > 0).mean()),
            ("mean", "Mean detections per object", lambda v: float(np.mean(v)))]):
        rv = [fn(rows[c]["real"]) for c in EVAL_CLASSES]
        gv = [fn(rows[c]["gen"]) for c in EVAL_CLASSES]
        axes[j].bar(x - 0.19, rv, 0.36, color=REAL, label="real radar")
        axes[j].bar(x + 0.19, gv, 0.36, color=GEN, label="generated")
        for i, (a, b) in enumerate(zip(rv, gv)):
            axes[j].text(i - 0.19, a, f"{a:.1f}", ha="center", va="bottom",
                         fontsize=8, color=INK_SOFT)
            axes[j].text(i + 0.19, b, f"{b:.1f}", ha="center", va="bottom",
                         fontsize=8, color=INK_SOFT)
        axes[j].set_xticks(x)
        axes[j].set_xticklabels([f"{c}\n(n={len(rows[c]['real'])})" for c in EVAL_CLASSES])
        axes[j].set_title(title)
        axes[j].set_ylabel("%" if stat == "hit" else "detections")
        # Headroom so the value labels never touch the frame.
        axes[j].set_ylim(0, max(max(rv), max(gv)) * 1.18)

    for k, c in enumerate(EVAL_CLASSES):
        for j, (key, col) in enumerate((("real", REAL), ("gen", GEN))):
            v = rcs_on[c][key]
            if not v:
                continue
            d = np.concatenate(v)
            bp = axes[2].boxplot([d], positions=[k * 3 + j], widths=0.85,
                                 showfliers=False, patch_artist=True,
                                 medianprops=dict(color=SURFACE, lw=1.8),
                                 whiskerprops=dict(color=col, lw=1.2),
                                 capprops=dict(color=col, lw=1.2))
            for b in bp["boxes"]:
                b.set(facecolor=col, edgecolor=col, linewidth=0)
    axes[2].set_xticks([k * 3 + 0.5 for k in range(len(EVAL_CLASSES))])
    axes[2].set_xticklabels(EVAL_CLASSES)
    axes[2].set_ylabel("RCS (dBsm)")
    axes[2].set_title("RCS of detections ON each class")

    # ONE legend for the figure: all three panels carry the same two series, so
    # three legends would be three chances to collide with a value label for no
    # added information.
    fig.legend(handles=[plt.Line2D([], [], color=REAL, lw=7, label="real radar"),
                        plt.Line2D([], [], color=GEN, lw=7, label="generated")],
               loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2,
               frameon=False)
    fig.suptitle(f"Object-wise behaviour · {head.upper()}", fontsize=12.5,
                 color=INK, y=0.975)
    finish(fig, out / "07_object_wise.png",
           "The right panel is the class-discriminative RCS contrast a detector actually uses. If "
           "the generated boxes overlap where the real ones separate, the cloud looks right and "
           "still carries no class information — which is this project's standing downstream result.")


def fig_neighbourhood(C, out, head):
    """Local spacing. The classifier two-sample test found nearest-neighbour
    distance the single most separating statistic, so it is worth seeing."""
    from scipy.spatial import cKDTree
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    nn = {}
    for key in ("real", "gen"):
        xyz, fr = C[f"{key}_xyz"], C[f"{key}_frame"]
        d = []
        for f in np.unique(fr)[:200]:          # 200 frames is ample for a shape
            p = xyz[fr == f]
            if len(p) < 4:
                continue
            dd, _ = cKDTree(p).query(p, k=2)
            d.append(dd[:, 1])
        nn[key] = np.concatenate(d) if d else np.zeros(0)
    bins = np.linspace(0, 6, 70)
    step_hist(axes[0], nn["real"], nn["gen"], bins, "distance to nearest detection (m)")
    axes[0].set_title(f"Nearest-neighbour spacing   (W1 {w1(nn['real'], nn['gen']):.3f} m)")
    axes[0].legend(loc="upper right")

    for key, col, lab in (("real", REAL, "real radar"), ("gen", GEN, "generated")):
        xs = np.sort(nn[key])
        axes[1].plot(xs, np.linspace(0, 1, len(xs)), color=col, lw=2.0, label=lab)
    axes[1].set_xlim(0, 6)
    axes[1].set_xlabel("distance to nearest detection (m)")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].set_title("Cumulative")
    axes[1].legend(loc="lower right")

    fig.suptitle(f"Local point spacing · {head.upper()}", fontsize=12.5, color=INK, y=1.02)
    finish(fig, out / "08_nearest_neighbour.png",
           "This is the statistic the classifier two-sample test found most separating, so it is "
           "where the generated cloud is most identifiable as synthetic. A generator sitting on "
           "the real curve here would be hard to tell apart from real by local structure alone.")


def fig_vr_structure(C, out, head):
    """v_r against azimuth. For a static scatterer v_r = -(v_ego . u), so the
    real data traces a cosine in azimuth scaled by ego speed. This checks the
    ANALYTIC channel, not a learned one."""
    density_triptych(
        np.linspace(-62, 62, 90), np.linspace(-22, 12, 80),
        (C["real_az"], C["real_vr"].astype(float)),
        (C["gen_az"], C["gen_vr"].astype(float)),
        xlabel="azimuth (deg)", ylabel="radial velocity (m/s)",
        title=f"v_r structure against azimuth · {head.upper()}",
        caption="For a static scatterer v_r = -(v_ego · u): a cosine in azimuth scaled by ego speed. "
                "v_r is COMPUTED analytically and never predicted, so agreement here validates the "
                "geometry rather than a model. Vertical spread is the mix of ego speeds.",
        path=out / "09_vr_structure.png")


def fig_head_prediction(head, out):
    """Predicted vs measured RCS on the matched test candidates.

    The most direct ground-truth-vs-prediction view available: this is the
    head's own regression, before any calibration or sampling.
    """
    from rear.calibration.scaling import apply_scaler
    from rear.materials.features import features_from_composition
    from rear.models.base import load_head
    import torch

    cache = REPO_ROOT / "outputs" / "fit_vod" / "cache" / "labeled_test_all.npz"
    ck_path = REPO_ROOT / "weights" / "vod" / f"{head}_head.pt"
    if not (cache.exists() and ck_path.exists()):
        print("  (skipping head-prediction figure: cache or checkpoint missing)")
        return
    z = np.load(cache)
    model, ck = load_head(ck_path)
    X = np.delete(z["X"].astype(np.float64), 7, axis=1)      # the `drop` variant
    Xs = apply_scaler(X, ck["mu"], ck["sigma"], ck.get("clip"))
    if ck["n_material"]:
        Xs = np.hstack([Xs, features_from_composition(
            z["mat_comp"].astype(np.float64), z["mat_frac"].astype(np.float64))])
    with torch.no_grad():
        pred = np.concatenate([model(torch.tensor(Xs[i:i + 262144], dtype=torch.float32)
                                     )["rcs_mean"].numpy()
                               for i in range(0, len(Xs), 262144)])
    pred = pred * float(ck["y_sigma"]) + float(ck["y_mu"])
    m = z["winning_mask"]
    y = z["winning_rcs"].astype(np.float64)
    p, w = pred[m], z["y"][m].astype(np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.subplots_adjust(wspace=0.30, bottom=0.24, top=0.84, left=0.055, right=0.985)
    xb = np.linspace(-40, 25, 90); yb = np.linspace(-40, 25, 90)
    h, _, _ = np.histogram2d(y, p, bins=[xb, yb])
    # No colour bar: on a predicted-vs-measured density the absolute cell value
    # carries nothing a reader acts on -- the shape against the diagonal is the
    # whole content -- and the bar only collided with the next panel's labels.
    axes[0].pcolormesh(xb, yb, (h / h.sum()).T, cmap=SEQ, shading="auto",
                       vmin=0, vmax=np.percentile((h / h.sum())[h > 0], 99))
    axes[0].plot([-40, 25], [-40, 25], color=INK_SOFT, lw=1.2, ls="--")
    axes[0].set_xlabel("measured RCS (dBsm)"); axes[0].set_ylabel("predicted RCS (dBsm)")
    axes[0].set_title("Predicted vs measured (density)")
    axes[0].set_aspect("equal"); axes[0].grid(False)
    wm = np.average(y, weights=w)
    wr2 = 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - wm) ** 2)
    annotate(axes[0], f"w-R² = {wr2:.4f}\nn = {len(y):,}")

    # The regression-to-the-mean the w-R2 number cannot show on its own.
    qs = np.linspace(0, 100, 21)
    edges = np.percentile(y, qs)
    cx, cy, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (y >= a) & (y < b)
        if s.sum() < 30:
            continue
        cx.append(y[s].mean()); cy.append(p[s].mean())
        lo.append(np.percentile(p[s], 25)); hi.append(np.percentile(p[s], 75))
    axes[1].plot([-40, 25], [-40, 25], color=INK_SOFT, lw=1.2, ls="--",
                 label="perfect")
    axes[1].fill_between(cx, lo, hi, color=GEN, alpha=0.22, lw=0,
                         label="predicted IQR")
    axes[1].plot(cx, cy, color=GEN, lw=2.2, marker="o", ms=4.5,
                 label="mean prediction")
    axes[1].set_xlabel("measured RCS (dBsm)"); axes[1].set_ylabel("predicted RCS (dBsm)")
    axes[1].set_title("Conditional mean — regression to the mean")
    axes[1].legend(loc="upper left")

    res = p - y
    # One series: the title names it, so no legend box (which would only fight
    # the statistics annotation for the same corner).
    axes[2].hist(res, bins=np.linspace(-40, 40, 90), density=True,
                 histtype="step", color=GEN, lw=2.0)
    axes[2].axvline(0, color=INK_SOFT, lw=1.2, ls="--")
    axes[2].set_xlabel("predicted − measured (dBsm)")
    axes[2].set_ylabel("density")
    axes[2].set_title("Residuals")
    annotate(axes[2], f"bias {res.mean():+.2f} dBsm\nsd {res.std():.2f}")

    fig.suptitle(f"Head-level RCS regression · {head.upper()} · matched test candidates",
                 fontsize=12.5, color=INK, y=0.965)
    finish(fig, out / "10_head_prediction.png",
           "This is the head ALONE — no calibration, no sampling. The middle panel shows the "
           "compression a conditional-mean regressor always has: it under-predicts the strong "
           "returns and over-predicts the weak ones, which is exactly what the RCS quantile "
           "calibration exists to undo at generation time.")


def fig_sim_vs_real(C, out, head):
    """The deployment comparison. There is no real radar on the simulated side,
    so this is a distribution SHIFT, not an error."""
    kc = REPO_ROOT / "downstream" / "cache" / f"kc_{head}.npz"
    if not kc.exists():
        print("  (skipping sim-vs-real figure: no KITTI-CARLA cache)")
        return
    z = np.load(kc)
    pts, ptr = z["points"], z["points_ptr"]
    k_rcs = pts[:, 3].astype(float)
    k_rng = np.linalg.norm(pts[:, :3].astype(float), axis=1)
    k_n = np.diff(ptr).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    series = [("real radar (VoD)", C["real_rcs"].astype(float), REAL),
              ("generated (VoD)", C["gen_rcs"].astype(float), GEN),
              ("generated (KITTI-CARLA)", k_rcs, THIRD)]
    for lab, d, c in series:
        axes[0].hist(d, bins=np.linspace(-45, 30, 80), density=True,
                     histtype="step", color=c, lw=2.0, label=lab)
    axes[0].set_xlabel("RCS (dBsm)"); axes[0].set_ylabel("density")
    axes[0].set_title("RCS across the domain gap")
    axes[0].legend(loc="upper left")

    for lab, d, c in [("real radar (VoD)", C["real_rng"], REAL),
                      ("generated (VoD)", C["gen_rng"], GEN),
                      ("generated (KITTI-CARLA)", k_rng, THIRD)]:
        axes[1].hist(d, bins=np.linspace(0, 105, 70), density=True,
                     histtype="step", color=c, lw=2.0, label=lab)
    axes[1].set_xlabel("range (m)"); axes[1].set_ylabel("density")
    axes[1].set_title("Range occupancy")
    axes[1].legend(loc="upper right")

    counts = [("real radar\n(VoD)", C["real_n_per_frame"].astype(float), REAL),
              ("generated\n(VoD)", C["gen_n_per_frame"].astype(float), GEN),
              ("generated\n(KITTI-CARLA)", k_n, THIRD)]
    parts = axes[2].violinplot([c[1] for c in counts], showmeans=True, widths=0.8)
    for b, (_, _, col) in zip(parts["bodies"], counts):
        b.set_facecolor(col); b.set_edgecolor(col); b.set_alpha(0.75)
    for k in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[k].set_color(INK_SOFT); parts[k].set_linewidth(1.1)
    axes[2].set_xticks([1, 2, 3]); axes[2].set_xticklabels([c[0] for c in counts])
    axes[2].set_ylabel("detections per frame")
    axes[2].set_title("Per-frame yield")
    for i, (_, d, _) in enumerate(counts, start=1):
        # Above the mean line, not on it -- on it the label and the rule
        # overprint each other and neither is readable.
        axes[2].text(i, d.mean(), f"mean {d.mean():.0f}", ha="center", va="bottom",
                     fontsize=8.5, color=INK_SOFT,
                     bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none",
                               alpha=0.85))

    fig.suptitle(f"Simulated vs real domain · {head.upper()}", fontsize=12.5,
                 color=INK, y=1.02)
    finish(fig, out / "11_sim_vs_real.png",
           "KITTI-CARLA has NO real radar, so nothing here is an error measurement — it is the "
           "distribution shift the deployment faces. The generated-VoD curve is the one with a "
           "ground truth beside it; the KITTI-CARLA curve shows how far the same model drifts "
           "when the input domain changes.")


def fig_covariate_shift(C, out, head, n_frames=120):
    """The input-side shift, which is where the deployment risk actually lives."""
    from rear.data.base import load_adapter
    from rear.materials.features import (MATERIAL_FEATURE_NAMES,
                                         features_from_composition)
    from rear.materials.pipeline import MaterialExtractor

    cache = REPO_ROOT / "outputs" / "fit_vod" / "cache" / "labeled_test_all.npz"
    kc_cache = REPO_ROOT / "outputs" / "kc_features.npz"
    if not cache.exists():
        print("  (skipping covariate-shift figure: no VoD feature cache)")
        return
    if not kc_cache.exists():
        print(f"  building KITTI-CARLA feature sample ({n_frames} frames) ...")
        # Candidate enumeration ONLY -- no association, no soft labels.
        # KITTI-CARLA has no real radar, so `build_labeled_frame` cannot run
        # there and does not need to: this figure compares the head's INPUTS,
        # which exist on both datasets, not its targets, which exist on one.
        from _fit_common import candidates_for_frame
        a = load_adapter("kitti_carla", split="Town01")
        ids = a.sequence_splits()["Town01"]
        pick = np.linspace(0, len(ids) - 1, n_frames).astype(int)
        ex = MaterialExtractor(a, device="cpu")
        Xs, comps, fracs = [], [], []
        for i in pick:
            fr = a.load_frame(ids[int(i)])
            cand, cfg, sem = candidates_for_frame(a, fr)
            if not len(cand.centres):
                continue
            d = ex.composition_for_frame(fr, cand, cfg, point_semantic=sem)
            Xs.append(cand.features.astype(np.float64))
            comps.append(d["composition"]); fracs.append(d["frac_labelled"])
        kc_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(kc_cache, X=np.concatenate(Xs),
                            mat_comp=np.concatenate(comps),
                            mat_frac=np.concatenate(fracs))
    zk = np.load(kc_cache)
    zv = np.load(cache)

    names = ["range", "azimuth", "elevation", "cos_angle", "height_above_ground",
             "point_density", "path_loss", "occlusion_depth"]
    Xv = np.delete(zv["X"].astype(np.float64), 7, axis=1)
    Xk = np.delete(zk["X"].astype(np.float64), 7, axis=1)
    Mv = features_from_composition(zv["mat_comp"].astype(np.float64),
                                   zv["mat_frac"].astype(np.float64))
    Mk = features_from_composition(zk["mat_comp"].astype(np.float64),
                                   zk["mat_frac"].astype(np.float64))
    show_mat = ["frac_labelled", "eps_real_mean", "gamma_pow_mean", "metal_fraction"]
    mi = [MATERIAL_FEATURE_NAMES.index(n) for n in show_mat]

    cols = [(names[i], Xv[:, i], Xk[:, i]) for i in range(8)] + \
           [(show_mat[j], Mv[:, k], Mk[:, k]) for j, k in enumerate(mi)]

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.48, wspace=0.16,
                        left=0.03, right=0.985)
    shifts = []
    for ax, (nm, v, k) in zip(axes.ravel(), cols):
        mu, sd = v.mean(), v.std() + 1e-12
        lo = min(np.percentile(v, 0.5), np.percentile(k, 0.5))
        hi = max(np.percentile(v, 99.5), np.percentile(k, 99.5))
        bins = np.linspace(lo, hi, 60)
        ax.hist(v, bins=bins, density=True, histtype="step", color=REAL, lw=2.0,
                label="VoD (fitting)")
        ax.hist(k, bins=bins, density=True, histtype="step", color=THIRD, lw=2.0,
                label="KITTI-CARLA (deployment)")
        d = abs(k.mean() - mu) / sd
        shifts.append((nm, d))
        ax.set_title(f"{nm}   Δμ = {d:.2f} sd", fontsize=9.5)
        ax.set_yticks([])
    axes[0, 0].legend(loc="upper right", fontsize=8)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    worst = max(shifts, key=lambda s: s[1])
    fig.suptitle(f"Covariate shift, fitting → deployment · {head.upper()} "
                 f"(largest: {worst[0]} at {worst[1]:.2f} VoD-sd)",
                 fontsize=12.5, color=INK, y=0.965)
    finish(fig, out / "12_covariate_shift.png",
           "Every input the head sees, on the dataset it was FITTED on against the dataset it is "
           "DEPLOYED on, each shift in units of the fitting set's own sd. `frac_labelled` is the "
           "known one: material comes from predicted semantics on VoD and simulator ground truth "
           "on KITTI-CARLA, so coverage differs by construction. There is no real radar on the "
           "simulated side to check the consequence against — this is the open risk, drawn.")


def fig_link_budget(head, out):
    """PRISM's additive decomposition, plotted. An unstructured head has no
    equivalent — this is what the link-budget factorisation buys."""
    if head != "prism":
        return
    from rear.calibration.scaling import apply_scaler
    from rear.materials.features import features_from_composition
    from rear.models.base import load_head
    import torch

    cache = REPO_ROOT / "outputs" / "fit_vod" / "cache" / "labeled_test_all.npz"
    ck_path = REPO_ROOT / "weights" / "vod" / "prism_head.pt"
    if not (cache.exists() and ck_path.exists()):
        return
    z = np.load(cache)
    model, ck = load_head(ck_path)
    X = np.delete(z["X"].astype(np.float64), 7, axis=1)
    Xs = apply_scaler(X, ck["mu"], ck["sigma"], ck.get("clip"))
    block = features_from_composition(z["mat_comp"].astype(np.float64),
                                      z["mat_frac"].astype(np.float64))
    Xf = np.hstack([Xs, block])
    idx = np.random.default_rng(0).choice(len(Xf), min(120000, len(Xf)), replace=False)
    with torch.no_grad():
        contrib = model.branch_contributions(torch.tensor(Xf[idx], dtype=torch.float32))
    raw = X[idx]

    groups = [("propagation", "range", 0), ("beam", "azimuth", 1),
              ("surface", "cos_angle", 3), ("occlusion", "occlusion_depth", 7),
              ("material", "gamma_pow_mean", None)]
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 4.2))
    fig.subplots_adjust(bottom=0.30, top=0.80, wspace=0.30, left=0.05, right=0.99)
    for ax, (br, xname, col) in zip(axes, groups):
        s_ = contrib[br][:, 1].numpy()          # task 1 = the RCS read-out
        xv = (block[idx][:, 0] if col is None else raw[:, col])

        # DISCRETE features get their own values as bins. occlusion_depth is a
        # small integer count (0..3 under this config), and forcing it through
        # 26 continuous bins leaves most of them under the minimum-count floor
        # and draws nothing at all -- an empty panel that looks like the branch
        # learned nothing, when it simply was not binned correctly.
        uniq = np.unique(xv)
        if len(uniq) <= 12:
            mid = uniq
            groups_idx = [np.isclose(xv, u) for u in uniq]
        else:
            lo, hi = np.percentile(xv, [1, 99])
            edges = np.linspace(lo, hi, 26)
            mid = 0.5 * (edges[:-1] + edges[1:])
            groups_idx = [(xv >= a) & (xv < b) for a, b in zip(edges[:-1], edges[1:])]

        keep = [i for i, m in enumerate(groups_idx) if m.sum() > 40]
        if not keep:
            ax.text(0.5, 0.5, "too few samples", transform=ax.transAxes,
                    ha="center", va="center", color=INK_SOFT, fontsize=9)
        else:
            mid = np.asarray(mid)[keep]
            sel = [groups_idx[i] for i in keep]
            mean = np.array([s_[m].mean() for m in sel])
            q1 = np.array([np.percentile(s_[m], 25) for m in sel])
            q3 = np.array([np.percentile(s_[m], 75) for m in sel])
            ax.fill_between(mid, q1, q3, color=GEN, alpha=0.2, lw=0)
            marker = "o" if len(mid) <= 12 else None
            ax.plot(mid, mean, color=GEN, lw=2.2, marker=marker, ms=5)
        ax.axhline(0, color=INK_SOFT, lw=0.9, ls="--")
        ax.set_title(f"{br} branch", fontsize=10)
        ax.set_xlabel(xname)
        ax.set_ylabel("additive contribution\nto RCS (standardised)" if br == "propagation" else "")

    fig.suptitle("PRISM-2's link budget, learned and separately plottable "
                 "(shaded = interquartile spread)", fontsize=12.5, color=INK, y=0.95)
    finish(fig, out / "13_link_budget.png",
           "Each branch reads a DISJOINT feature group and emits one additive scalar, so the model "
           "is a neural GAM in the five physics groups and each learned term can be drawn on its "
           "own. An unstructured head has no such decomposition; this is what the factorisation "
           "buys beyond accuracy.")



FIGURE_INDEX = [
    ("01_rcs_distribution.png", "RCS marginal, CDF and Q–Q",
     "The channel the detector is ~98% dependent on. Q–Q localises which quantiles disagree."),
    ("02_rcs_by_range.png", "RCS conditioned on range",
     "A matching overall marginal can hide a wrong range dependence. Same bands metrics.json scores."),
    ("03_rcs_vs_range_density.png", "(range, RCS) joint density and its difference",
     "Where in the joint distribution the two clouds disagree, and in which direction."),
    ("04_spatial_bev.png", "Bird's-eye spatial density and its difference",
     "Where detections physically land, both sides FOV-masked."),
    ("05_marginals.png", "The four output marginals",
     "Range, azimuth, elevation, v_r — each with its Wasserstein distance."),
    ("06_point_counts.png", "Detections per frame and per object",
     "Yield, per-object supply, over-dispersion and the hard hit-rate constraint."),
    ("07_object_wise.png", "Object-wise behaviour by class",
     "Hit rate, detections per object, and the class-discriminative RCS contrast."),
    ("08_nearest_neighbour.png", "Local point spacing",
     "The statistic the classifier two-sample test found most separating."),
    ("09_vr_structure.png", "v_r against azimuth",
     "The ego-motion cosine. v_r is computed, never predicted — a geometry check."),
    ("10_head_prediction.png", "Head-level RCS regression",
     "GROUND TRUTH vs PREDICTED, the head alone: no calibration, no sampling."),
    ("11_sim_vs_real.png", "Simulated vs real domain",
     "Generated-on-KITTI-CARLA against generated-on-VoD and real VoD."),
    ("12_covariate_shift.png", "Covariate shift, fitting → deployment",
     "Every model input on both datasets, in units of the fitting set's own sd."),
    ("13_link_budget.png", "PRISM's learned link budget",
     "Each branch's additive contribution, plotted separately. PRISM only."),
]


def write_index(out, head, C):
    """A README beside the figures. A directory of PNGs with no reading order
    is a pile; this says what each one answers and in what order to look."""
    L = [f"# Distribution charts — {head.upper()}\n",
         f"Generated 3+1D radar against real paired radar, **{int(C['meta_n_frames'])} "
         f"VoD test frames**, {len(C['real_rcs']):,} real and {len(C['gen_rcs']):,} "
         f"generated in-FOV detections.\n",
         "These are the SHAPES behind the numbers in "
         "[`../metrics.json`](../metrics.json) and "
         "[`../QUANTITATIVE_RESULTS.md`](../QUANTITATIVE_RESULTS.md). A Wasserstein "
         "distance says how far two distributions are; it does not say in which "
         "direction or where. That is what these show.\n",
         "| # | figure | what it answers |", "|---|---|---|"]
    for i, (fn, title, desc) in enumerate(FIGURE_INDEX, start=1):
        if not (out / fn).exists():
            continue
        L.append(f"| {i} | [{title}]({fn}) | {desc} |")
    L += ["",
          "## How to read them\n",
          "**Colour means the same thing in every figure.** Blue is always the real "
          "measurement, orange always this framework's output, green a second "
          "generated condition. Density maps use one blue ramp light→dark; "
          "difference maps use a diverging blue↔red scale that is NEUTRAL GREY at "
          "zero, so grey means agreement.\n",
          "**Two panels side by side always share one colour scale** where they are "
          "meant to be compared — independent scales would make the same ink mean "
          "different numbers in each.\n",
          "**What has a ground truth and what does not.** Real-vs-generated on VoD is "
          "a paired comparison on the same frames, and is the only place error can "
          "be measured. KITTI-CARLA has no real radar, so figure 11 and 12 show "
          "distribution SHIFT, not error — the honest thing to plot when the answer "
          "is unavailable.\n",
          "Regenerate with `python tools/plot_distributions.py --head "
          f"{head}` (reads the cached collection; seconds, not a re-run).\n"]
    (out / "README.md").write_text("\n".join(L))
    print(f"  wrote README.md")


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, help="architecture to plot (mlp | prism)")
    ap.add_argument("--out", default=None,
                    help="default: output/<head>/quantitative/distribution")
    ap.add_argument("--kc-frames", type=int, default=120,
                    help="KITTI-CARLA frames sampled for the covariate-shift figure")
    args = ap.parse_args()

    qdir = REPO_ROOT / "output" / args.head / "quantitative"
    out = Path(args.out) if args.out else qdir / "distribution"
    out.mkdir(parents=True, exist_ok=True)
    style()

    C = load_collection(args.head, qdir)
    print(f"plotting {args.head}: {int(C['meta_n_frames'])} frames, "
          f"{len(C['real_rcs']):,} real / {len(C['gen_rcs']):,} generated detections")

    fig_rcs(C, out, args.head)
    fig_rcs_by_range(C, out, args.head)
    fig_rcs_range_density(C, out, args.head)
    fig_spatial(C, out, args.head)
    fig_marginals(C, out, args.head)
    fig_counts(C, out, args.head)
    fig_object_wise(C, out, args.head)
    fig_neighbourhood(C, out, args.head)
    fig_vr_structure(C, out, args.head)
    fig_head_prediction(args.head, out)
    fig_sim_vs_real(C, out, args.head)
    fig_covariate_shift(C, out, args.head, args.kc_frames)
    fig_link_budget(args.head, out)
    write_index(out, args.head, C)
    print(f"\n{len(list(out.glob('*.png')))} figures in {out}")


if __name__ == "__main__":
    main()
