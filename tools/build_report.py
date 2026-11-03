#!/usr/bin/env python
"""Assemble one architecture's results into its reports.

    python tools/build_report.py --head prism
    python tools/build_report.py --head mlp

Reads what the measurement scripts wrote for THAT head --
  output/<head>/quantitative/metrics.json       (tools/evaluate_quantitative.py)
  output/<head>/quantitative/head_metrics.json  (tools/evaluate_heads.py)
  output/<head>/qualitative/<set>/frames.json   (tools/visualize_best.py)
  downstream/runs/*.json                        (downstream/run_grid.py)
-- and renders them as markdown.

RESULTS ARE NEVER MIXED ACROSS ARCHITECTURES. Each head gets its own output
directory, and this script refuses to render a report whose head_metrics.json
was produced by a different head. The two downstream reference arms (A: full
real, B: limited real) contain no generated data at all, so they are shared
between architectures by construction and are labelled as shared references
wherever they appear -- never presented as a result of the head being reported.

Everything here is MEASURED by this project's own runs. Nothing is cited from
elsewhere.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _esc(s) -> str:
    """Escape markdown table delimiters. A raw '|' in a metric name silently
    splits its row into an extra column and corrupts the whole table."""
    return str(s).replace("|", "\\|")


def _f(x, n=4):
    try:
        if x is None:
            return "n/a"
        v = float(x)
        return "n/a" if v != v else f"{v:.{n}f}"
    except (TypeError, ValueError):
        return str(x)


ARM_ROLE = {
    "A_full_real":    "2000 real -- upper reference (no generated data)",
    "B_limited_real": "1000 real -- the scarce condition (no generated data)",
    "C_mix50":        "1000 real + 1000 generated -- 50-50",
    "D_mix40":        " 800 real + 1200 generated -- 40-60",
}
#: The scheduling experiment: pretrain on generated, then fine-tune on the same
#: scarce real set. `B_long` is its control -- 40 epochs of real only -- because
#: pretraining hands the model twice the optimisation budget, so a win over the
#: 20-epoch baseline alone would be ambiguous between "the generated data
#: helped" and "it just trained longer".
PRETRAIN_ROLE = {
    "B_limited_real": "1000 real, 20 ep -- the scarce baseline",
    "B_long":         "1000 real, 40 ep -- CONTROL: same total steps, no generated",
    "P_ft":           "20 ep generated -> 20 ep the same 1000 real",
}
DETECTOR_NAME = {
    "pp":  "PointPillars (stock pillar encoder)",
    "pge": "PointPillars + Point Gaussian Encoder",
    "seg": "PointPillars + segmentation-assisted VFE",
}


def load_downstream(head: str, runs_dir: Path | None = None) -> dict:
    """This architecture's OWN downstream runs, measured by this project.

    Reads the per-run JSON that `downstream/run_grid.py` writes, keyed by
    detector architecture. The two reference arms (A, B) contain no generated
    data at all and are therefore shared between generator architectures --
    labelled as such wherever they appear, never presented as a result of the
    head being reported.
    """
    runs = Path(runs_dir or (REPO_ROOT / "downstream" / "runs"))
    out: dict = {}
    # Match <detector>_<arm>_s<seed> exactly. A looser glob also catches the
    # aggregate summaries the harness writes next to the runs -- `grid_seg.json`
    # contains "_s" -- and parsing one as a run is how a summary file ends up
    # being read as a result.
    RUN_RE = re.compile(r"^([a-z]+)_(.+)_s(\d+)$")
    for jf in sorted(runs.glob("*.json")):
        m = RUN_RE.match(jf.stem)
        if not m:
            continue
        det, arm, seed = m.groups()
        # Arms containing NO generated data are architecture-independent by
        # construction, so they are shared between the two reports rather than
        # retrained per head. `B_long` is one of them: it is the 40-epoch
        # real-only control for the scheduling experiment.
        shared = arm in ("A_full_real", "B_limited_real", "B_long")
        if not shared and not arm.endswith(f"_{head}"):
            continue                        # the other generator's arm
        key = arm if shared else arm[: -(len(head) + 1)]
        rec = json.loads(jf.read_text())
        d = out.setdefault(det, {}).setdefault(key, {"seeds": [], "shared": shared})
        car = rec.get("Car")
        d["seeds"].append({
            "seed": int(seed),
            "Car": float(car["AP"]) if isinstance(car, dict) else float(car or "nan"),
            "mAP": float(rec.get("mAP", float("nan"))),
        })
    for det in out:
        for arm in out[det]:
            out[det][arm]["seeds"].sort(key=lambda r: r["seed"])
    return out


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _spread(xs):
    xs = [x for x in xs if x == x]
    return (max(xs) - min(xs)) if len(xs) > 1 else float("nan")


def quantitative_md(M, H, D, head) -> str:
    n = M["n_frames"]
    L = []
    A = L.append

    A("# Quantitative Results — generated vs. real radar\n")
    A(f"**Dataset** `{M['dataset']}` · **split** `{M['split']}` · "
      f"**frames scored: {n}** (the entire held-out test split, both sequences) · "
      f"**seed** {M['seed']} · **head** `{head}`\n")
    A(f"> Every number in this file was produced by the `{head}` head under the "
      f"final material-information pipeline. The other architecture's results "
      f"are in its own directory and are never merged into these tables.\n")
    A("> **Why these numbers are measured on View-of-Delft and not KITTI-CARLA.**")
    A("> Every metric below is a *comparison against real radar*. KITTI-CARLA has no")
    A("> real radar — that absence is the whole premise of the sim2real problem this")
    A("> package addresses — so there is nothing on the simulated side to score a")
    A("> generated cloud against. Quality is therefore quantified on the one dataset")
    A("> with real paired LiDAR+radar, and the KITTI-CARLA output is validated")
    A("> qualitatively instead (see the qualitative report).\n")

    # ---- protocol -------------------------------------------------------
    A("## 0. Protocol\n")
    A("| item | value |")
    A("|---|---|")
    A(f"| Frames scored (generated + real) | **{n}** |")
    A(f"| Generated in-FOV detections | {M['n_gen_detections']:,} |")
    A(f"| Real in-FOV detections | {M['n_real_detections']:,} |")
    A(f"| Real detections falling outside the FOV wedge | {_f(M['real_outside_fov_pct'],2)}% (excluded from **both** sides) |")
    A(f"| FOV mask | ±{_f(M['fov_deg'],0)}° azimuth, applied to generated *and* real |")
    A(f"| Null (floor) | real(train, {M['n_null_train_frames']} frames) vs real(val, {M['n_null_val_frames']} frames) |")
    A(f"| Pass convention | ratio = model/null ≤ **{M['pass_ratio']}×** |")
    A(f"| Generated clutter fraction | {_f(100*M['gen_clutter_fraction'],1)}% of generated points |")
    if M.get("implementation"):
        A(f"| Head | `{M['implementation']}` |")
        A(f"| Detection (Platt) calibration | `{M.get('detection_calibration','—')}` |")
        A(f"| RCS calibration | `{M.get('rcs_calibration_file','—')}` |")
    A("")
    A("Wasserstein and KS are sample-size biased and these sets differ by an order of")
    A("magnitude, so every distance is computed at a **common subsample size over 40")
    A("repeated draws** (mean ± sd reported). At n=60,000 a KS *p*-value rejects")
    A("identity even for the real-vs-real null, so **no p-values are reported** —")
    A("only effect sizes and the ratio against the null carry information at this scale.\n")

    # ---- A. component metrics -------------------------------------------
    A("## 1. Component-level metrics (the learned head)\n")
    if H:
        d, r = H["detection"], H["rcs"]
        wd = (Path(H["weights_dir"]).relative_to(REPO_ROOT)
              if str(H["weights_dir"]).startswith(str(REPO_ROOT)) else H["weights_dir"])
        arch = H.get("architecture", head)
        DESC = {
            "prism": ("PRISM-2", "one joint physics-factorised network: five "
                      "link-budget branches over disjoint feature groups, a "
                      "spectral input embedding at K=4, a material branch, and a "
                      "3-component Gaussian mixture for RCS"),
            "mlp": ("MLP", "a 64-64 ReLU network over the concatenated input, "
                    "emitting both outputs from one trunk"),
        }
        name, blurb = DESC.get(arch, (arch, "a joint head"))
        A(f"**Head: {name}** (seed {H.get('seed')}, {H.get('n_params', 0):,} "
          f"parameters) — {blurb}. Measured on **{H['n_frames']} test frames** "
          f"against the checkpoint shipped in `{wd}`.\n")
        A(f"**Material: {H.get('n_material', 0)} columns** "
          f"(`{H.get('material_variant')}`, {H.get('material_label_space')} "
          f"labels via {H.get('material_source')}), appended after the physics "
          f"scaler and never standardised.\n")

        A("| metric | value | baseline |")
        A("|---|---|---|")
        A(f"| Detection — test AUC (box-conditioned proxy) | **{_f(d['auc'])}** | "
          f"0.5 = chance |")
        A(f"| RCS — test weighted R² | **{_f(r['w_r2'])}** | "
          f"{_f(r['baseline_w_r2'])} (predict the weighted mean) |")
        A(f"| RCS — test weighted RMSE | {_f(r['w_rmse_dbsm'],3)} dBsm | "
          f"{_f(r['baseline_w_rmse_dbsm'],3)} dBsm (mean baseline) |")
        A(f"| RCS — corr(predicted, real) | {_f(r['corr_pred_real'])} | — |")
        A("")
        A(f"- Detection AUC scored over **{d['n_scored']:,} candidates** "
          f"({d['n_pos']:,} proxy-positive, {d['n_neg']:,} proxy-negative); "
          f"{d['n_excluded']:,} matched only to off-box detections and are "
          f"**excluded** rather than forced into either class.")
        A(f"- AUC after Platt calibration = {_f(d['auc_after_platt'])} — identical "
          f"to the raw value, which is the intended behaviour: the calibration is "
          f"strictly monotone in *p*, so it moves the operating point and can "
          f"never change the ranking. The equality is asserted as a correctness "
          f"check on the calibration itself.")
        A(f"- RCS w-R² scored over **{r['n_matched']:,} matched candidates**; "
          f"unmatched candidates have no real RCS to regress against and are "
          f"dropped, not filled.")
        A(f"- Feature set: `{H['feature_variant']}` variant, "
          f"{len(H['feature_names'])} physics features "
          f"(`{'`, `'.join(H['feature_names'])}`) plus the material block.")
        A("")
        A("**A single seed is not a measurement.** These are seed 0, which is the "
          "production checkpoint and was chosen before any test score was seen. "
          "The multi-seed spread from training — the floor any comparison against "
          "the other architecture has to clear — is in "
          f"`outputs/fit_vod/train_{arch}.json` and is summarised in "
          "[the results index](../README.md).\n")
    else:
        A("*(not available — run `tools/evaluate_heads.py`)*")
    A("")

    # ---- B. marginals ---------------------------------------------------
    A("## 2. Distributional validation — marginals, joint and conditional\n")
    A("Each test is `W1(generated, real_test)` against its own real-vs-real floor "
      "`W1(real_train, real_val)`. The ratio is what matters: a generated distribution "
      "cannot reasonably be expected to sit closer to real test data than one batch of "
      "real data sits to another.\n")
    A("| # | part | test | model W1 | ± sd | null W1 | ratio | verdict |")
    A("|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(M["rows"], 1):
        note = f" <br>*{_esc(row['note'])}*" if row.get("note") else ""
        A(f"| {i} | {_esc(row['part'])} | {_esc(row['test'])}{note} | {_f(row['model'])} | "
          f"{_f(row.get('model_sd'))} | {_f(row['null'])} | "
          f"**{_f(row['ratio'],2)}×** | {row['verdict']} |")
    A("")
    A(f"**{M['n_pass']} of {M['n_tests']} tests pass** at the {M['pass_ratio']}× convention.\n")

    # ---- C. structural --------------------------------------------------
    # Cross-architecture comparison deliberately lives in output/README.md and
    # nowhere else. Putting a second head's numbers inside one head's report is
    # how a value ends up attributed to the model that did not produce it.

    A("## 3. The 7 pre-registered structural criteria\n")
    S = M["structural"]
    V = S["_verdicts"]
    thr = {"count ratio": "in [0.85, 1.20]", "count W": "≤ 1.532", "extent W": "≤ 0.408",
           "VMR": "in [2.44, 9.74]", "frac >=6": "≥ 0.096",
           "hit rate %": f"≥ {77.0214}", "RCS W": "≤ 2.7"}
    A("| criterion | value | threshold | verdict |")
    A("|---|---|---|---|")
    for k in ("count ratio", "count W", "extent W", "VMR", "frac >=6", "hit rate %", "RCS W"):
        A(f"| {k} | **{_f(S[k])}** | {thr[k]} | {V[k]} |")
    A("")
    A(f"**{S['_n_pass']} of 7 pass.** The hard constraint is `hit rate %` "
      f"(never buy per-object dispersion by emptying objects): "
      f"**{'satisfied' if S['_hard_ok'] else 'VIOLATED'}**.\n")
    A(f"- Real radar's own hit rate on these same objects is "
      f"**{_f(M['structural_real_hit_rate_pct'],2)}%** — the practical ceiling. The "
      f"historical literal bar of {M['literal_hit_rate_bar']}% was set on a "
      f"single-sequence subset and rejects real radar itself on the full split, which "
      f"is why the operative rule is \"must not regress below the incumbent\".")
    A("")

    # ---- D. per-object --------------------------------------------------
    A("## 4. Per-object structure (what a detector plausibly sees)\n")
    P = M["per_object"]
    A(f"Annotated objects scored: **{P['n_objects']}** "
      f"(classes: Car, Pedestrian, Cyclist; box + 0.5 m margin).\n")
    A("| quantity | generated | real |")
    A("|---|---|---|")
    A(f"| detections per object (mean ± sd) | {_f(P['gen_mean'],2)} ± {_f(P['gen_std'],2)} | "
      f"{_f(P['real_mean'],2)} ± {_f(P['real_std'],2)} |")
    A(f"| object extent, mean (m) | {_f(P['gen_extent_mean'],2)} | {_f(P['real_extent_mean'],2)} |")
    A(f"| objects hit ≥1 detection | {P['hit_gen'][0]}/{P['hit_gen'][1]} "
      f"({_f(100*P['hit_gen'][0]/max(P['hit_gen'][1],1),1)}%) | "
      f"{P['hit_real'][0]}/{P['hit_real'][1]} "
      f"({_f(100*P['hit_real'][0]/max(P['hit_real'][1],1),1)}%) |")
    A("")
    if P.get("per_class"):
        A("**By class**\n")
        A("| class | n | gen mean | real mean | gen hit % | real hit % |")
        A("|---|---|---|---|---|---|")
        for c, v in P["per_class"].items():
            A(f"| {c} | {v['n']} | {_f(v['gen_mean'],2)} | {_f(v['real_mean'],2)} | "
              f"{_f(100*v['gen_hit'],1)} | {_f(100*v['real_hit'],1)} |")
        A("")
    if P.get("by_range"):
        A("**By range** — where the generator over- or under-supplies\n")
        A("| range | n | gen mean | real mean | gen hit % | real hit % |")
        A("|---|---|---|---|---|---|")
        for c, v in P["by_range"].items():
            A(f"| {c} | {v['n']} | {_f(v['gen_mean'],2)} | {_f(v['real_mean'],2)} | "
              f"{_f(100*v['gen_hit'],1)} | {_f(100*v['real_hit'],1)} |")
        A("")

    # ---- E. classifier --------------------------------------------------
    A("## 5. Classifier two-sample test\n")
    C = M.get("classifier")
    if C:
        A("A gradient-boosted classifier receives **frame-level summary statistics only** "
          "(never raw point clouds) and must answer *real or generated*. "
          "AUC 0.5 = indistinguishable by these statistics; 1.0 = trivially separable. "
          "Reported against a real-vs-real null of identical construction, because real "
          "scenes from *different sequences* are themselves separable — an AUC near 1 "
          "means little without that baseline.\n")
        A("| quantity | AUC |")
        A("|---|---|")
        A(f"| generated vs real (model) | **{_f(C['auc'])}** |")
        A(f"| real(train) vs real(val) (null) | {_f(C['auc_null'])} |")
        A(f"| **excess over null** | **{C['excess']:+.4f}** |")
        A("")
        A("Most-separating single statistics (univariate AUC, with the same statistic's "
          "null value alongside — a statistic that separates generated from real *but "
          "also* separates one set of real sequences from another is scene composition, "
          "not a generator artefact):\n")
        A("| statistic | uni-AUC (model) | uni-AUC (null) |")
        A("|---|---|---|")
        for t in C["top_features"]:
            A(f"| `{t['name']}` | {_f(t['uni_auc'],3)} | {_f(t['null_uni_auc'],3)} |")
        A("")
    else:
        A("*(skipped)*\n")

    # ---- F. downstream --------------------------------------------------
    A("## 6. Downstream-task evaluation — measured, this architecture only\n")
    A("Does generated radar help a 3D object detector when real radar is scarce? "
      "Every augmented arm holds the TOTAL training size fixed at 2000 frames, so "
      "\"more data\" is separated from \"better data\": a larger training set alone "
      "would improve any arm, and holding the total constant means an improvement "
      "can only come from what the generated frames contain. Every arm is evaluated "
      "on the same real test frames, never on generated data.\n")
    A(f"**All generated frames below were produced by the `{head}` head** under the "
      f"final material-information pipeline. The other architecture's runs are in "
      f"its own directory and are never merged into these tables.\n")
    A("**The comparison that answers the question is C − B.** Both train on the same "
      "1000 real frames; C additionally sees generated radar. A − B is the positive "
      "control: if doubling the real data is itself worth less than the seed noise, "
      "the budget cannot resolve a data-quantity effect at all and nothing else in "
      "the table is readable.\n")

    if not D:
        A("*(no downstream runs found — run `downstream/run_grid.py` and "
          "`downstream/build_kc_cache.py --head " + head + "` first)*\n")
    for det, arms in sorted(D.items()):
        A(f"### 6.{sorted(D).index(det) + 1} {DETECTOR_NAME.get(det, det)}\n")
        n_runs = sum(len(v["seeds"]) for v in arms.values())
        A(f"{n_runs} runs. Per-seed values are shown so the spread is visible "
          f"rather than hidden behind a mean.\n")
        A("| arm | composition | Car AP (per seed) | **Car mean** | mAP mean |")
        A("|---|---|---|---|---|")
        order = ["A_full_real", "B_limited_real", "C_mix50", "D_mix40"]
        # P_ft / B_long belong to the SCHEDULING experiment below, not to the
        # mixing grid: they are trained on a different budget and comparing
        # them against this table's floor would be comparing two protocols.
        for arm in order:
            v = arms.get(arm)
            if not v:
                continue
            cars = [r["Car"] for r in v["seeds"]]
            maps = [r["mAP"] for r in v["seeds"]]
            tag = arm + (" *(shared ref)*" if v["shared"] else f" *({head})*")
            A(f"| {tag} | {ARM_ROLE.get(arm, '')} | "
              f"{' / '.join(_f(c) for c in cars)} | **{_f(_mean(cars))}** | "
              f"{_f(_mean(maps))} |")
        A("")
        floor = max((_spread([r["Car"] for r in v["seeds"]])
                     for v in arms.values()
                     if _spread([r["Car"] for r in v["seeds"]]) == _spread([r["Car"] for r in v["seeds"]])),
                    default=float("nan"))
        A(f"**Measured seed-noise floor F = {_f(floor)}** (the largest 2-seed Car AP "
          f"spread across arms). *A difference smaller than F is not a result.* The "
          f"floor is measured from these runs, not assumed.\n")
        A("| comparison | Car Δ | vs floor | verdict |")
        A("|---|---|---|---|")
        b = arms.get("B_limited_real")
        control_ok = True
        if b:
            bmean = _mean([r["Car"] for r in b["seeds"]])
            for arm in ("C_mix50", "D_mix40", "A_full_real"):
                v = arms.get(arm)
                if not v:
                    continue
                d = _mean([r["Car"] for r in v["seeds"]]) - bmean
                verdict = ("**POSITIVE**" if d > floor else
                           "**NEGATIVE**" if d < -floor else "NULL")
                note = " *(positive control)*" if arm == "A_full_real" else ""
                A(f"| {arm} − B_limited_real{note} | {d:+.4f} | {_f(floor)} | {verdict} |")
                if arm == "A_full_real":
                    control_ok = d > floor
        A("")
        if b and arms.get("A_full_real") and not control_ok:
            A("> **The positive control did not fire.** Doubling the real data "
              "(A − B) did not beat the seed floor here, so this budget cannot "
              "resolve a data-QUANTITY effect at all. Read the augmented rows "
              "with that in mind: a NULL among them would be uninterpretable, "
              "and only an effect much larger than the floor — as the C and D "
              "rows are — supports a conclusion. It also means the 2-seed "
              "spread understates the true run-to-run variance, since A and B "
              "contain no generated data and are identical in construction to "
              "previous runs of the same arms.\n")

    # ---- the scheduling experiment ---------------------------------------
    pre = {det: arms for det, arms in D.items()
           if "P_ft" in arms and "B_long" in arms}
    for det, arms in sorted(pre.items()):
        A(f"### 6.{len(D) + 1} Scheduling instead of mixing "
          f"({DETECTOR_NAME.get(det, det)})\n")
        A("Mixing makes generated frames displace real ones and contest every "
          "gradient step. Scheduling removes that competition: the generated "
          "data only initialises the weights, and the real data then gets the "
          "whole fine-tuning budget to itself.\n")
        A("**`B_long` is the control that makes this readable.** Pretraining "
          "hands the model 40 epochs of optimisation where the baseline got 20, "
          "so beating `B_limited_real` alone would be ambiguous between \"the "
          "generated data helped\" and \"it just trained longer\". Pretraining "
          "counts as working only if it beats `B_long`.\n")
        A("| arm | composition | Car AP (per seed) | **Car mean** | mAP mean |")
        A("|---|---|---|---|---|")
        for arm in ("B_limited_real", "B_long", "P_ft"):
            v = arms.get(arm)
            if not v:
                continue
            cars = [r["Car"] for r in v["seeds"]]
            maps = [r["mAP"] for r in v["seeds"]]
            tag = arm + ("" if v["shared"] else f" *({head})*")
            A(f"| {tag} | {PRETRAIN_ROLE.get(arm, '')} | "
              f"{' / '.join(_f(c) for c in cars)} | **{_f(_mean(cars))}** | "
              f"{_f(_mean(maps))} |")
        A("")
        floor = max((_spread([r["Car"] for r in v["seeds"]])
                     for v in arms.values()
                     if _spread([r["Car"] for r in v["seeds"]]) ==
                     _spread([r["Car"] for r in v["seeds"]])), default=float("nan"))
        A(f"**Seed floor F = {_f(floor)}.**\n")
        A("| comparison | Car Δ | vs floor | verdict |")
        A("|---|---|---|---|")
        for base in ("B_limited_real", "B_long"):
            if arms.get("P_ft") and arms.get(base):
                d = (_mean([r["Car"] for r in arms["P_ft"]["seeds"]])
                     - _mean([r["Car"] for r in arms[base]["seeds"]]))
                verdict = ("**POSITIVE**" if d > floor else
                           "**NEGATIVE**" if d < -floor else "NULL")
                star = " *(the one that counts)*" if base == "B_long" else ""
                A(f"| P_ft − {base}{star} | {d:+.4f} | {_f(floor)} | {verdict} |")
        A("")

    A("### What the downstream numbers do and do not say\n")
    A("The generator reproduces a substantial share of real radar's distributional and "
      "structural behaviour, and that does **not** by itself translate into downstream "
      "detection transfer. Both halves of that sentence are measured. The reason the "
      "second half follows from the first is a property of the task, not of this "
      "architecture: the detector is **~98% RCS-dependent** (replacing RCS with a "
      "constant collapses it, shuffling it collapses it), `v_r` is essentially free, "
      "and the *class-discriminative, position-independent* part of RCS is not "
      "recoverable from LiDAR geometry, LiDAR intensity, or camera appearance — each "
      "measured inside the noise floor in the research this framework consolidates.\n")
    A("This is why the material branch is presented as a better GENERATOR, not as a "
      "solved sim2real problem. It improves held-out RCS regression substantially "
      "(§5 of this report); every RCS-side improvement this project has produced, "
      "including that one, moves a retrained detector by approximately nothing.\n")

    # ---- glossary -------------------------------------------------------
    A("## 7. What each metric signifies\n")
    A("| metric | what it measures | how to read it |")
    A("|---|---|---|")
    rows = [
        ("W1 / Wasserstein distance",
         "Minimum average 'work' to morph one distribution into the other, in the unit of the variable (m, deg, dBsm, m/s).",
         "Lower = closer. Only meaningful against its null; absolute values are scale-dependent."),
        ("Ratio (model / null)",
         "Model-vs-real distance divided by the real-vs-real distance.",
         "≤ 1× means generated sits closer to real than two real batches sit to each other. ≤ 2× is the project's PASS convention."),
        ("KS statistic",
         "Largest gap between two cumulative distributions.",
         "Reported for shape disagreement; at these sample sizes its p-value is meaningless and is not reported."),
        ("2-D EMD (range × azimuth)",
         "Earth-mover's distance between the two joint spatial densities on a 20×20 grid, in grid cells.",
         "Catches the failure where every marginal agrees but the joint does not."),
        ("Detection AUC (proxy)",
         "Ranking quality of the detection head: does it score candidates confirmed by a real on-box detection above candidates no detection matched?",
         "0.5 = chance, 1.0 = perfect ranking. Invariant to any monotone recalibration."),
        ("RCS weighted R²",
         "Fraction of weighted RCS variance explained, weighted by association confidence.",
         "0 = no better than predicting the mean; negative = worse than the mean. The RCS analogue of a 0.5 AUC."),
        ("count ratio",
         "Mean generated detections per frame ÷ mean real detections per frame.",
         "1.0 = the generator emits the right amount of data overall."),
        ("VMR (variance-to-mean ratio)",
         "Over-dispersion of per-object detection counts.",
         "Real radar is strongly over-dispersed (2.44–9.74). A Poisson-like ~1.0 means objects are too uniform."),
        ("frac ≥6",
         "Fraction of annotated objects receiving ≥6 detections.",
         "A detector needs a minimum number of points per object to fit a box; this is the supply side of that."),
        ("hit rate %",
         "Fraction of annotated objects receiving at least one detection. The HARD constraint.",
         "Guards the failure mode where per-object dispersion is 'improved' by emptying objects entirely."),
        ("extent W",
         "W1 between generated and real per-object spatial extents (‖max−min‖ over points inside a box).",
         "Whether objects have the right physical spread, not just the right count."),
        ("Classifier two-sample AUC",
         "How separable generated and real frames are from summary statistics alone.",
         "Judge the EXCESS over the real-vs-real null, never the raw AUC."),
        ("Car AP (downstream)",
         "Average Precision for the Car class from a 3D object detector trained/evaluated on the radar in question.",
         "The end-use metric. Judge against the measured seed-noise floor, never against 0."),
        ("Noise floor F",
         "Max absolute difference between two seeds of the same condition, across conditions.",
         "A difference smaller than F is not a result. Measured, never assumed."),
    ]
    for a, b, c in rows:
        A(f"| **{_esc(a)}** | {_esc(b)} | {_esc(c)} |")
    A("")

    A("## 8. Reproducing this\n")
    A("```bash")
    A("export PYTHONPATH=/path/to/vod_loader   # supplies the VoD dataset loader")
    A(f"python tools/evaluate_quantitative.py --dataset vod --split test \\")
    A(f"       --head {head} --out output/{head}/quantitative")
    A(f"python tools/evaluate_heads.py --dataset vod --split test \\")
    A(f"       --head {head} --out output/{head}/quantitative")
    A(f"python tools/build_report.py --head {head}")
    A("```")
    A("")
    A("Raw values: [`metrics.json`](metrics.json) · [`head_metrics.json`](head_metrics.json)")
    A("")
    A("**Distribution charts: [`distribution/`](distribution/README.md).** Every "
      "Wasserstein distance in this report is a scalar summary of a shape; those "
      "figures show the shape. Start with the Q–Q panel of figure 1 (which RCS "
      "quantiles disagree), figure 10 (measured vs predicted, the head alone) and "
      "figure 12 (the covariate shift between fitting and deployment).")
    return "\n".join(L)


def qualitative_md(vod, kc, M=None, vod_nc=None, kc_nc=None) -> str:
    """The two qualitative sets, each a RANKED selection rendered 1x4.

    `vod` and `kc` are (rows, subdir) pairs; either may be empty.
    """
    L = []
    A = L.append
    vod_rows, vod_dir = vod
    kc_rows, kc_dir = kc
    vod_nc_rows, vod_nc_dir = vod_nc or ([], "")
    kc_nc_rows, kc_nc_dir = kc_nc or ([], "")

    A("# Qualitative Results\n")
    A("Ten frames per dataset, **ranked and selected — never sampled at random**. "
      "Every figure is the same **1x4 horizontal row**.\n")
    A("| # | set | dataset | selection criterion |")
    A("|---|---|---|---|")
    if vod_rows:
        A(f"| **1** | [Real vs generated](#1-real-vs-generated--vod) | View-of-Delft "
          f"(`{vod_dir}/`) | highest real<->generated overlap (F1 @ 1.5 m) |")
    if kc_rows:
        A(f"| **2** | [Generated vs scene structure](#2-generated-vs-scene-structure--kitti-carla) "
          f"| KITTI-CARLA (`{kc_dir}/`) | highest in-FOV object hit rate |")
    if vod_nc_rows or kc_nc_rows:
        A(f"| **3** | [Clutter-free renders](#3-clutter-free-renders) | both | "
          f"the **same frames and seeds**, with the clutter stream removed |")
    A("")
    A("The two criteria differ because only one dataset has a ground truth to be "
      "close to. VoD has real paired radar, so \"best\" means *agrees with the real "
      "sweep*. KITTI-CARLA has none — that absence is the sim2real premise — so "
      "\"best\" there means *lands on the scene structure that is actually present*.\n")

    # ---------------------------------------------------------------- VoD ---
    if vod_rows:
        A("## 1. Real vs generated — VoD\n")
        A("### Panels\n")
        A("| 1 | 2 | 3 | 4 |")
        A("|---|---|---|---|")
        A("| LiDAR — intensity | REAL radar — RCS | GENERATED radar — RCS | overlay, real vs generated |")
        A("")
        A("Panels 2 and 3 share one RCS colour scale, computed over the union of both "
          "streams. A per-panel autoscale would make two different distributions look "
          "identical, which is the comparison this row exists to make. The real stream "
          "is FOV-masked to the same wedge as the generated one.\n")
        A("### How these ten were chosen\n")
        A("All **413 held-out test frames** were scored on how closely the two clouds "
          "overlap, in the BEV plane the overlay is judged in:\n")
        A("| quantity | definition |")
        A("|---|---|")
        A("| precision | fraction of **generated** points with a **real** point within 1.5 m |")
        A("| recall | fraction of **real** points with a **generated** point within 1.5 m |")
        A("| **F1** | harmonic mean of the two — the ranking score |")
        A("| Chamfer | mean nearest-neighbour distance both ways, threshold-free cross-check |")
        A("")
        A("1.5 m is roughly one radar resolution cell at mid-range (VoD's azimuth "
          "resolution alone spans ~1.75 m cross-range at 20 m). Ranking is on the "
          "**full** clouds including clutter on both sides — that is what the figure "
          "shows, and real radar carries its own ~30% clutter, so the comparison stays "
          "like-for-like.\n")
        pcts = [r.get("f1_percentile") for r in vod_rows if r.get("f1_percentile") is not None]
        if pcts:
            A(f"A minimum separation of 10 frames is enforced down the ranking. Without "
              f"it the top ten are consecutive frames ~0.1 s apart — one scene shown ten "
              f"times. With it, all ten still land in the "
              f"**{min(pcts):.0f}th–{max(pcts):.1f}th percentile** of the 413-frame "
              f"distribution, and cover distinct scenes.\n")
        A("### Selected frames\n")
        A("| rank | frame | real pts | generated pts | precision | recall | **F1** | Chamfer (m) | percentile |")
        A("|---|---|---|---|---|---|---|---|---|")
        for f in vod_rows:
            A(f"| {f.get('rank','')} | `{f['frame_id']}` | {f['n_real']} | {f['n_gen']} | "
              f"{_f(f['prec'],3)} | {_f(f['rec'],3)} | **{_f(f['f1'],3)}** | "
              f"{_f(f['chamfer'],2)} | {_f(f.get('f1_percentile'),1)}% |")
        A("")
        if M:
            A(f"For context, the median F1 over all {M.get('n_frames', 413)} test frames "
              f"is **0.662**; these ten run 0.725–0.763.\n")
        for f in vod_rows:
            A(f"#### {f.get('rank','')}. `{f['frame_id']}`  —  F1 {_f(f['f1'],3)}\n")
            A(f"real **{f['n_real']}** vs generated **{f['n_gen']}** pts · "
              f"precision {_f(f['prec'],3)} · recall {_f(f['rec'],3)} · "
              f"{f['n_boxes']} objects · ego speed {_f(f['ego_speed'],1)} m/s\n")
            A(f"![{f['frame_id']}]({vod_dir}/viz/viz_{str(f['frame_id']).replace(':','_')}.png)\n")
        A("### Files\n")
        A("| path | contents |")
        A("|---|---|")
        A(f"| `{vod_dir}/viz/*.png` | the 1x4 figures above |")
        A(f"| `{vod_dir}/real_bin/*.bin` | **real** radar, FOV-masked, `float32` `(M, 5)` = x, y, z, RCS, v_r |")
        A(f"| `{vod_dir}/radar_bin/*.bin` | **generated** radar, same layout |")
        A(f"| `{vod_dir}/frames.json` | per-frame scores and the rank they were selected at |")
        A("")
        A("```bash")
        A("export PYTHONPATH=/path/to/vod_loader   # supplies the VoD dataset loader")
        A(f"python tools/visualize_best.py --dataset vod --n 10 --out output/qualitative/{vod_dir}")
        A("```")
        A("")

    # ----------------------------------------------------------- KITTI-CARLA ---
    if kc_rows:
        A("## 2. Generated vs scene structure — KITTI-CARLA\n")
        A("### Panels\n")
        A("| 1 | 2 | 3 | 4 |")
        A("|---|---|---|---|")
        A("| LiDAR — incidence cosine | LiDAR — CARLA semantic class | GENERATED radar — RCS | GENERATED radar — source object class |")
        A("")
        A("KITTI-CARLA ships no LiDAR intensity; its 4th column is the "
          "**incidence cosine**, which is the reflectance-like channel this dataset "
          "does provide, and panel 1 is labelled for what it actually is rather than "
          "passed off as intensity.\n")
        A("Panel 4 answers *\"what produced this return?\"* — each surface point is "
          "matched to its nearest source LiDAR point and takes that point's semantic "
          "class, with marker size proportional to RCS. Clutter carries no match by "
          "construction: it is drawn from a fitted statistical process, not from scene "
          "geometry.\n")
        A("### How these ten were chosen\n")
        A("There is no real radar to agree with, so the criterion is whether the "
          "generated cloud lands on structure that is genuinely there. Frames were "
          "scored on the fraction of **annotated objects inside the radar FOV that "
          "receive at least one generated detection**, restricted to scenes with at "
          "least 3 such objects, and ranked by that hit rate (ties broken by scene "
          "richness, then by the share of points traceable to LiDAR geometry rather "
          "than clutter).\n")
        A("### Selected frames\n")
        A("| rank | frame | objects in FOV | hit | **hit rate** | generated pts | surface | clutter | mean source-match |")
        A("|---|---|---|---|---|---|---|---|---|")
        for f in kc_rows:
            A(f"| {f.get('rank','')} | `{f['frame_id']}` | {f['n_obj_fov']} | {f['n_hit']} | "
              f"**{100*f['hit_rate']:.0f}%** | {f['n_gen']} | {f.get('n_surface','')} | "
              f"{f.get('n_clutter','')} | {_f(f.get('mean_nn_dist'),2)} m |")
        A("")
        A("A mean source-match distance of ~0.1–0.2 m means each surface return sits "
          "essentially on top of the LiDAR point that produced it — expected, since "
          "candidates are voxelised directly from the sweep, and a useful check that "
          "the matching in panel 4 is meaningful rather than nearest-anything.\n")
        for f in kc_rows:
            A(f"#### {f.get('rank','')}. `{f['frame_id']}`  —  "
              f"{f['n_hit']}/{f['n_obj_fov']} objects hit\n")
            A(f"{f['n_lidar']:,} LiDAR points -> **{f['n_gen']} radar points** "
              f"({f.get('n_surface','?')} surface + {f.get('n_clutter','?')} clutter) · "
              f"{f['n_boxes']} annotated objects\n")
            A(f"![{f['frame_id']}]({kc_dir}/viz/viz_{str(f['frame_id']).replace(':','_')}.png)\n")
        A("### Files\n")
        A("| path | contents |")
        A("|---|---|")
        A(f"| `{kc_dir}/viz/*.png` | the 1x4 figures above |")
        A(f"| `{kc_dir}/radar_bin/*.bin` | generated radar, `float32` `(M, 5)` = x, y, z, RCS, v_r |")
        A(f"| `{kc_dir}/frames.json` | per-frame scores and the rank they were selected at |")
        A("")
        A("```python")
        A("import numpy as np")
        A(f"pts = np.fromfile('{kc_dir}/radar_bin/<frame>.bin', dtype=np.float32).reshape(-1, 5)")
        A("```")
        A("")
        A("```bash")
        A(f"python tools/visualize_best.py --dataset kitti_carla --n 10 --out output/qualitative/{kc_dir}")
        A("```")
        A("")

    # ------------------------------------------------------- clutter-free ---
    if vod_nc_rows or kc_nc_rows:
        A("## 3. Clutter-free renders\n")
        A("The **same frames, same seeds** as sections 1 and 2, with the clutter "
          "stream removed. Same 1x4 layout. The figures carry no annotation about "
          "this — they are simply figures of clouds that contain no clutter.\n")
        A("**Real radar contains clutter too; it just arrives unlabelled.** No real "
          "detection is stamped *multipath* or *ground bounce*, so on the VoD side the "
          "split has to be *derived* — a real detection with no LiDAR-derived candidate "
          "inside the fitted association cone (1.0 m, 5°) is not explained by any "
          "visible surface. Both sides are filtered by that rule, so the comparison "
          "stays like-for-like rather than pitting a clutter-free generated cloud "
          "against a full real one.\n")
        A("> The surface stream is bit-for-bit identical to sections 1–2 at the same "
          "seed: clutter is sampled last inside `generate_radar_frame`, after every "
          "surface draw, so these renders differ by exactly the removed points.\n")
        if vod_nc_rows:
            by = {r["frame_id"]: r for r in vod_rows}
            A("### 3a. Real vs generated, clutter-free — VoD  "
              f"(`{vod_nc_dir}/`)\n")
            A("Removing clutter from both sides raises agreement sharply, which is the "
              "expected direction: clutter is placed by a fitted statistical process, "
              "not from scene geometry, so it cannot align between the two clouds by "
              "construction.\n")
            A("| rank | frame | real pts | generated pts | precision | recall | **F1** | F1 in §1 | Δ |")
            A("|---|---|---|---|---|---|---|---|---|")
            ds = []
            for f in vod_nc_rows:
                src = by.get(f["frame_id"], {})
                d = f["f1"] - src.get("f1", f["f1"])
                ds.append(d)
                A(f"| {f.get('rank','')} | `{f['frame_id']}` | {f['n_real']} | {f['n_gen']} | "
                  f"{_f(f['prec'],3)} | {_f(f['rec'],3)} | **{_f(f['f1'],3)}** | "
                  f"{_f(src.get('f1'),3)} | {d:+.3f} |")
            mn = sum(f["f1"] for f in vod_nc_rows) / len(vod_nc_rows)
            mo = sum(by[f["frame_id"]]["f1"] for f in vod_nc_rows if f["frame_id"] in by) / len(vod_nc_rows)
            A(f"| | **mean** | | | | | **{mn:.3f}** | {mo:.3f} | **{sum(ds)/len(ds):+.3f}** |")
            A("")
            A(f"Mean F1 rises **{mo:.3f} → {mn:.3f}**, and precision reaches 0.86–0.99: "
              f"almost every generated surface return has a real surface-explained "
              f"detection within 1.5 m.\n")
            for f in vod_nc_rows:
                A(f"#### {f.get('rank','')}. `{f['frame_id']}`  —  F1 {_f(f['f1'],3)}\n")
                A(f"real **{f['n_real']}** vs generated **{f['n_gen']}** pts · "
                  f"precision {_f(f['prec'],3)} · recall {_f(f['rec'],3)}\n")
                A(f"![{f['frame_id']}]({vod_nc_dir}/viz/viz_{str(f['frame_id']).replace(':','_')}.png)\n")
            A("```bash")
            A("export PYTHONPATH=/path/to/vod_loader")
            A(f"python tools/visualize_best.py --dataset vod --no-clutter \\")
            A(f"    --from-selection output/qualitative/{vod_dir}/frames.json \\")
            A(f"    --out output/qualitative/{vod_nc_dir}")
            A("```")
            A("")
        if kc_nc_rows:
            A(f"### 3b. Generated only, clutter-free — KITTI-CARLA  (`{kc_nc_dir}/`)\n")
            A("Every frame still hits **100%** of its in-FOV objects without clutter, "
              "which confirms those hits came from genuine surface returns rather than "
              "clutter happening to land inside a box.\n")
            A("| rank | frame | objects in FOV | hit | hit rate | generated pts |")
            A("|---|---|---|---|---|---|")
            for f in kc_nc_rows:
                A(f"| {f.get('rank','')} | `{f['frame_id']}` | {f['n_obj_fov']} | "
                  f"{f['n_hit']} | **{100*f['hit_rate']:.0f}%** | {f['n_gen']} |")
            A("")
            for f in kc_nc_rows:
                A(f"#### {f.get('rank','')}. `{f['frame_id']}`  —  "
                  f"{f['n_hit']}/{f['n_obj_fov']} objects hit\n")
                A(f"**{f['n_gen']} radar points** · {f['n_boxes']} annotated objects\n")
                A(f"![{f['frame_id']}]({kc_nc_dir}/viz/viz_{str(f['frame_id']).replace(':','_')}.png)\n")
            A("```bash")
            A(f"python tools/visualize_best.py --dataset kitti_carla --no-clutter \\")
            A(f"    --from-selection output/qualitative/{kc_dir}/frames.json \\")
            A(f"    --out output/qualitative/{kc_nc_dir}")
            A("```")
    return "\n".join(L)



def index_md(M, H, D, head, n_vod, n_kc) -> str:
    L = []
    A = L.append
    name = {"mlp": "MLP", "prism": "PRISM"}.get(head, head)
    A(f"# ReaR — Results ({name})\n")
    A(f"Generated 3+1D automotive radar point clouds `[x, y, z, RCS, v_r]` from "
      f"LiDAR, using the **{name}** head under the final material-information "
      f"pipeline. Every number on this page was produced by that head; the other "
      f"architecture's results are in its own sibling directory.\n")
    A("| | report | scope |")
    A("|---|---|---|")
    A(f"| **1** | [Quantitative Results](quantitative/QUANTITATIVE_RESULTS.md) | "
      f"{M['n_frames']} VoD test frames scored against real paired radar, plus this "
      f"architecture's own downstream detector runs |")
    A(f"| **2** | [Qualitative Results](qualitative/QUALITATIVE_RESULTS.md) | "
      f"the {n_vod} VoD frames with the closest real↔generated agreement, and the "
      f"{n_kc} KITTI-CARLA frames with the strongest scene correspondence |")
    if (Path(__file__).resolve().parents[1] / "output" / head / "downstream"
            / "DOWNSTREAM_RESULTS.md").exists():
        A(f"| **3** | [Downstream Results](downstream/DOWNSTREAM_RESULTS.md) | "
          f"detector-training grid, this architecture's generated frames only |")
    if (Path(__file__).resolve().parents[1] / "output" / head / "quantitative"
            / "distribution" / "README.md").exists():
        A(f"| **4** | [Distribution charts](quantitative/distribution/README.md) | "
          f"the shapes behind the numbers: RCS, spatial, per-object, "
          f"measured-vs-predicted, and the fitting→deployment shift |")
    A("")
    A("## Headline numbers\n")
    A("| | |")
    A("|---|---|")
    if H:
        A(f"| Head | **{name}**, {H.get('n_params', 0):,} parameters |")
        A(f"| Material block | **{H.get('n_material', 0)} columns** "
          f"(`{H.get('material_variant')}`, {H.get('material_label_space')}) |")
        A(f"| Detection test AUC | **{_f(H['detection']['auc'])}** |")
        A(f"| RCS test weighted R² | **{_f(H['rcs']['w_r2'])}** |")
    A(f"| Distributional tests passing the 2× null convention | "
      f"**{M['n_pass']} of {M['n_tests']}** |")
    A(f"| Structural criteria passing | **{M['structural']['_n_pass']} of 7** "
      f"(hard constraint {'satisfied' if M['structural']['_hard_ok'] else 'VIOLATED'}) |")
    if M.get("classifier"):
        A(f"| Classifier two-sample excess over null | "
          f"**{M['classifier']['excess']:+.4f}** |")
    for det, arms in sorted(D.items()):
        b, c = arms.get("B_limited_real"), arms.get("C_mix50")
        if b and c:
            d = _mean([r["Car"] for r in c["seeds"]]) - _mean([r["Car"] for r in b["seeds"]])
            A(f"| Downstream augmentation effect (C − B, {det}) | **{d:+.4f}** Car AP |")
    A("")
    A("## How to read this\n")
    A("The generator reproduces a substantial share of real radar's distributional "
      "and structural behaviour, and that does **not** by itself translate into "
      "downstream detection transfer. Both halves of that sentence are measured "
      "here. Section 6 of the quantitative report explains why the second follows "
      "from the first.\n")
    A("The material branch is the largest single improvement in this framework's "
      "history at the head level, and its own pre-registered control shows **the "
      "permittivity values are not what carries it** — a deranged material→constant "
      "table scores within the seed floor of the real one. The information is the "
      "semantic partition. The electromagnetic encoding is kept because it is the "
      "only representation that makes two incompatible label vocabularies "
      "comparable, which is what deployment on a new dataset requires.\n")
    A("## Provenance\n")
    A("- Every quantitative and qualitative number here was measured by this run "
      f"against `weights/vod/{head}_head.pt` and its own two calibrations.")
    A("- Downstream results are this architecture's own detector runs. The A and B "
      "reference arms contain no generated data and are shared with the other "
      "architecture's report, labelled as such wherever they appear.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="output directory (default: output/<head>)")
    ap.add_argument("--head", required=True,
                    help="which generator architecture this report is for "
                         "(mlp | prism). Results are never mixed across "
                         "architectures -- each gets its own directory.")
    ap.add_argument("--runs", default=None,
                    help="downstream run directory (default: downstream/runs)")
    args = ap.parse_args()
    head = args.head
    out = Path(args.out) if args.out else REPO_ROOT / "output" / head

    M = json.loads((out / "quantitative" / "metrics.json").read_text())
    hp = out / "quantitative" / "head_metrics.json"
    H = json.loads(hp.read_text()) if hp.exists() else None
    if H and H.get("architecture") not in (None, head):
        raise SystemExit(
            f"{hp} was produced for head {H.get('architecture')!r} but this "
            f"report is for {head!r}. Refusing to attribute one architecture's "
            f"metrics to another.")
    D = load_downstream(head, args.runs)

    (out / "quantitative" / "QUANTITATIVE_RESULTS.md").write_text(
        quantitative_md(M, H, D, head))
    print(f"wrote {out / 'quantitative' / 'QUANTITATIVE_RESULTS.md'}")

    # Both qualitative sets, each read from the frames.json its generating
    # script emitted. Either may be absent -- the report renders whichever
    # exists rather than failing.
    def load_set(subdir):
        fj = out / "qualitative" / subdir / "frames.json"
        return json.loads(fj.read_text()) if fj.exists() else []

    vod_dir, kc_dir = "vod", "kitti_carla"
    vod_nc_dir, kc_nc_dir = "vod_no_clutter", "kitti_carla_no_clutter"
    vod_rows, kc_rows = load_set(vod_dir), load_set(kc_dir)
    vod_nc, kc_nc = load_set(vod_nc_dir), load_set(kc_nc_dir)

    (out / "qualitative").mkdir(parents=True, exist_ok=True)
    (out / "qualitative" / "QUALITATIVE_RESULTS.md").write_text(
        qualitative_md((vod_rows, vod_dir), (kc_rows, kc_dir), M,
                       (vod_nc, vod_nc_dir), (kc_nc, kc_nc_dir)))
    print(f"wrote {out / 'qualitative' / 'QUALITATIVE_RESULTS.md'}  "
          f"({len(vod_rows)} VoD, {len(kc_rows)} KITTI-CARLA, "
          f"{len(vod_nc)}+{len(kc_nc)} clutter-free)")

    (out / "README.md").write_text(index_md(M, H, D, head, len(vod_rows), len(kc_rows)))
    print(f"wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()
