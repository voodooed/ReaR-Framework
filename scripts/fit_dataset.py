#!/usr/bin/env python
"""Fit every learnable stage of ReaR's radar model on a NEW dataset that has
real paired LiDAR + radar.

    python scripts/fit_dataset.py --dataset my_radar [--stage <name>] [--dry-run]

Produces the artifacts under `weights/<name>/` and the numbers that go into
`configs/datasets/<name>.yaml`'s `fitted:` block. This -- plus
scripts/train_heads.py for the two gradient-trained stages -- is the ONLY
place anything in this framework is fitted; nothing under `rear/` may contain
a dataset-specific number (see rear/data/base.py's module docstring, enforced
by tests/).

WHAT A NEW DATASET NEEDS, TO BE FIT-ABLE
-----------------------------------------
1. A `rear.data.base.DatasetAdapter` subclass implementing `load_frame`,
   `sequence_splits()` (SEQUENCE-level train/val/test -- frame-level splits
   leak, see null_control.py), `__len__`, `has_real_radar() -> True`, and
   `radar_xyz_in_lidar(frame)` mapping real radar detections into the LiDAR
   frame the generator works in. `rear/data/vod.py` is the reference
   implementation -- read it before writing a new one.
2. A `configs/datasets/<name>.yaml` with a `sensors:` block: at minimum
   `radar_mount_offset`, `radar_fov_degrees`, `lidar_sensor_type`, and
   `radar_columns` naming its real-radar array's columns -- must include
   `rcs` and an ego-motion-COMPENSATED radial velocity column named
   `v_r_comp` (see scripts/_fit_common.py:radar_columns for why the clutter
   stage refuses to run without one). No `fitted:` block yet -- this script
   produces it, stage by stage, printed for you to paste in.

THE DEPENDENCY CHAIN. Stages run in this order because each consumes the one
before it; the driver does not let you skip past a stage whose prerequisite
is missing.

  1  candidates       preflight: enumerate every split, report candidate
                       density and raw feature ranges. Nothing is fitted;
                       this just proves the adapter + config wiring works
                       before the slow stages run.
  2  association       sweep the operating point on VAL; SELECT ON THE
                       NULL-CONTROLLED TRUE SIGNAL, never on raw match rate
                       -- see rear/validation/null_control.py's docstring for
                       the measured case (a "fix" that raised observed match
                       rate 77.0%->90.3% while true signal FELL 45.4pp->28.3pp)
                       this rule exists to prevent repeating.
  3  soft_labels       records the V3 rank-weighted formula's constants.
                       Nothing computed here -- V3's parameters are structural
                       (see rear/calibration/soft_labels.py), but changing the
                       formula invalidates both heads, so it is a stage.
  4  scaling           reports robust median/IQR feature statistics on TRAIN.
                       The constants that actually ship are refit, on exactly
                       the same data, inside each head's own checkpoint by
                       train_heads.py -- this stage is a sanity check, not the
                       source of truth.

  -- at this point, run the gradient-trained stage, ONE LONG JOB PER
     ARCHITECTURE. Each trains a JOINT head emitting both outputs:
         python scripts/train_heads.py --dataset <name> --head prism
         python scripts/train_heads.py --dataset <name> --head mlp
     Both require stage 2's operating point to already be in your YAML (soft
     labels are built from it) --

  5  platt             monotone probability calibration so expected per-frame
                       detection counts match real counts. Cannot change AUC
                       (it is strictly monotone in p) -- reads
                       outputs/fit_<name>/detection_scores_val_<head>.npz.
  6  rcs_calibration   two-piece quantile map correcting the head's RCS output
                       (a conditional MEAN) onto the real RCS population's
                       marginal shape -- reads
                       outputs/fit_<name>/rcs_pred_train_<head>.npz.

     BOTH CALIBRATION STAGES ARE PER ARCHITECTURE and write
     <name>_calibration_<head>.pkl. They are functions of one head's own output
     distribution, so they do not transfer between heads -- and because both
     maps are monotone and applied after scoring, a mismatched pair moves no
     head-level metric and fails silently. Pass --head to select; it defaults
     to the config's fitted.heads.implementation.
  7  clutter           inhomogeneous Poisson process over range x azimuth for
                       every real detection NOT explained by a candidate,
                       speed-conditioned rate, plus the (never
                       null-controlled) per-range elevation pool needed to
                       give clutter a z. See rear/models/clutter_process.py.

This has only ever been run end to end against View-of-Delft. Treat a first
run on a new dataset as exactly that -- sanity-check the printed diagnostics
(candidate density, the true-signal table, matched fraction, material
coverage, clutter rate) before trusting the weights for anything downstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time  # noqa: E402

import numpy as np  # noqa: E402

from _fit_common import (  # noqa: E402
    RAW_FEATURE_NAMES, candidates_for_frame, in_fov_mask, radar_columns,
    radar_rcs_and_vr_comp)
from _fit_common import operating_point as _operating_point  # noqa: E402
from rear.calibration.scaling import fit_robust_scaler  # noqa: E402
from rear.data.base import load_adapter  # noqa: E402
from rear.geometry.association import cone_match  # noqa: E402
from rear.validation.null_control import true_signal  # noqa: E402

STAGES = ("candidates", "association", "soft_labels", "scaling",
         "platt", "rcs_calibration", "clutter")


def _head_of(adapter, args) -> str:
    """Which architecture the per-head stages act on."""
    return (getattr(args, "head", None)
            or adapter.fitted("heads", "implementation", default="prism"))


def _fit_dir(dataset: str) -> Path:
    d = REPO_ROOT / "outputs" / f"fit_{dataset}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _weights_dir(adapter, args) -> Path:
    """Where this stage writes its artifact. `--weights-dir` overrides the
    dataset's own `fitted.weights_dir` -- use it for a smoke test so it
    cannot overwrite a production profile such as weights/vod/."""
    if getattr(args, "weights_dir", None):
        d = Path(args.weights_dir)
    else:
        d = REPO_ROOT / str(adapter.fitted("weights_dir", default=f"weights/{adapter.name}"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rel(path: Path) -> str:
    """Path relative to the repo for the printed/recorded fitted.json summary,
    or the absolute path unchanged when it falls outside the repo (a
    `--weights-dir` smoke-test override, typically)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _match_frame(adapter, frame, tol, ang):
    """Candidates, real detections (LiDAR frame), FOV mask and match mask
    for one frame at a given (tol, ang) operating point."""
    cand, cfg, _ = candidates_for_frame(adapter, frame)
    origin = np.asarray(cfg.radar_position, dtype=np.float64)
    radar_lidar = adapter.radar_xyz_in_lidar(frame)
    fov = in_fov_mask(radar_lidar, origin, adapter.radar_fov_degrees) if len(radar_lidar) \
        else np.zeros(0, dtype=bool)
    matched = np.zeros(len(radar_lidar), dtype=bool)
    if len(cand.centres) and len(radar_lidar):
        matched, _ = cone_match(radar_lidar, cand.centres, origin, tol, ang)
    return cand, origin, radar_lidar, fov, matched



def _association_targets(adapter, ids, tol, ang, fit_dir, split, args):
    """Per-frame count of associated in-FOV real detections, cached.

    THIS TARGET IS HEAD-INDEPENDENT BY CONSTRUCTION -- it counts real
    detections that some candidate explains, which depends on the geometry and
    the association operating point and on nothing the network does. Computing
    it once and reusing it across architectures is therefore not an
    approximation, and it also guarantees both heads are calibrated against
    exactly the same target rather than two independently recomputed ones.

    The cache key includes the operating point, so changing it invalidates the
    cache instead of silently reusing targets from a different association.
    """
    tag = f"{split}_{len(ids)}_tol{tol:g}_ang{ang:g}"
    path = fit_dir / f"assoc_targets_{tag}.npz"
    if path.exists() and not getattr(args, "no_cache", False):
        d = np.load(path)
        print(f"  association targets: {path.name} (cached)")
        return {int(k): int(v) for k, v in zip(d["frame"], d["target"])}, d["assoc_rcs"]

    target, assoc = {}, []
    t0 = time.time()
    for i, fid in enumerate(ids):
        try:
            frame = adapter.load_frame(fid)
        except (KeyError, OSError, ValueError):
            target[i] = 0
            continue
        _, origin, dets, fov, matched = _match_frame(adapter, frame, tol, ang)
        keep = matched & fov
        target[i] = int(keep.sum())
        if keep.any():
            rcs, _vr = radar_rcs_and_vr_comp(adapter, frame)
            assoc.append(rcs[keep])
        if (i + 1) % 200 == 0 or i == len(ids) - 1:
            e = time.time() - t0
            print(f"    association {i + 1}/{len(ids)}  ({e / (i + 1):.2f}s/frame, "
                  f"eta {(len(ids) - i - 1) * e / (i + 1) / 60:.1f} min)", flush=True)
    assoc_rcs = np.concatenate(assoc) if assoc else np.zeros(0)
    np.savez_compressed(path, frame=np.array(sorted(target)),
                        target=np.array([target[k] for k in sorted(target)]),
                        assoc_rcs=assoc_rcs)
    print(f"  association targets -> {path.name}")
    return target, assoc_rcs


# ---------------------------------------------------------------- stage 1 ---
def stage_candidates(adapter, splits, args):
    for split, ids in splits.items():
        ids = ids if args.max_frames is None else ids[:args.max_frames]
        n_cand, ranges = 0, []
        for fid in ids:
            frame = adapter.load_frame(fid)
            cand, _, _ = candidates_for_frame(adapter, frame)
            n_cand += len(cand.centres)
            if len(cand.centres):
                ranges.append(cand.features[:, 0])
        mean_cand = n_cand / max(len(ids), 1)
        if ranges:
            r = np.concatenate(ranges)
            print(f"  [{split}] {len(ids)} frames, {n_cand:,} candidates "
                 f"({mean_cand:.0f}/frame), range [{r.min():.1f}, {r.max():.1f}] m")
        else:
            print(f"  [{split}] {len(ids)} frames, 0 candidates")
    return {"note": "preflight only; nothing persisted"}


# ---------------------------------------------------------------- stage 2 ---
def stage_association(adapter, splits, args,
                      tolerances=(0.5, 1.0, 1.5, 2.0), angles=(3.0, 5.0, 8.0)):
    ids = splits["val"] if args.max_frames is None else splits["val"][:args.max_frames]
    if not ids:
        raise SystemExit(f"{adapter.name} has no val-split frames to sweep against.")
    grid = {}
    for tol in tolerances:
        for ang in angles:
            obs_l, null_l, n = [], [], 0
            for fid in ids:
                frame = adapter.load_frame(fid)
                cand, cfg, _ = candidates_for_frame(adapter, frame)
                origin = np.asarray(cfg.radar_position, dtype=np.float64)
                dets = adapter.radar_xyz_in_lidar(frame)
                if not len(dets) or not len(cand.centres):
                    continue
                r = true_signal(lambda p: cone_match(p, cand.centres, origin, tol, ang)[0],
                                dets, origin)
                obs_l.append(r.observed); null_l.append(r.null); n += r.n
            if not obs_l:
                continue
            grid[(tol, ang)] = dict(observed=float(np.mean(obs_l)),
                                    null=float(np.mean(null_l)),
                                    true_signal=float(np.mean(obs_l) - np.mean(null_l)), n=n)
    if not grid:
        raise SystemExit("no val frame produced both candidates and detections; "
                         "check the adapter and dataset root.")
    best = max(grid, key=lambda k: grid[k]["true_signal"])
    loosest = max(grid, key=lambda k: grid[k]["observed"])
    print(f"  {'tol':>5}{'ang':>6}{'observed':>11}{'null':>9}{'TRUE':>9}")
    for k in sorted(grid):
        g = grid[k]
        flag = "  <- selected" if k == best else ""
        print(f"  {k[0]:>5.1f}{k[1]:>6.1f}{g['observed']:>10.1f}%"
             f"{g['null']:>8.1f}%{g['true_signal']:>8.1f}pp{flag}")
    if loosest != best:
        print(f"  NOTE: the highest raw match rate is at {loosest}, NOT the "
             f"selected cell -- selecting on raw rate would be wrong here too.")
    result = {"matcher": "cone", "range_tolerance_m": best[0], "cone_angle_deg": best[1],
              "true_signal_pp": round(grid[best]["true_signal"], 1)}
    print(f"\n  Paste into configs/datasets/{adapter.name}.yaml under fitted.association:")
    print(f"    range_tolerance_m: {result['range_tolerance_m']}")
    print(f"    cone_angle_deg: {result['cone_angle_deg']}")
    print(f"    true_signal_pp: {result['true_signal_pp']}")
    return result


# ---------------------------------------------------------------- stage 3 ---
def stage_soft_labels(adapter, splits, args):
    return {"formula": "v3_rank_weighted", "epsilon_unmatched": 0.02,
           "max_confidence": 1.0, "out_of_box_discount": 0.7}


# ---------------------------------------------------------------- stage 4 ---
def stage_scaling(adapter, splits, args):
    ids = splits["train"] if args.max_frames is None else splits["train"][:args.max_frames]
    feats = []
    for fid in ids:
        frame = adapter.load_frame(fid)
        cand, _, _ = candidates_for_frame(adapter, frame)
        if len(cand.centres):
            feats.append(cand.features)
    if not feats:
        raise SystemExit("no train candidates to scale.")
    X = np.concatenate(feats).astype(np.float64)
    mu, sigma = fit_robust_scaler(X)
    print(f"  {len(X):,} train candidates, {X.shape[1]} raw features")
    print(f"  {'feature':<20}{'median':>10}{'IQR':>10}")
    for j, name in enumerate(RAW_FEATURE_NAMES[:X.shape[1]]):
        print(f"  {name:<20}{mu[j]:>10.3f}{sigma[j]:>10.3f}")
    return {"method": "median_iqr", "clip": 10.0, "source": "checkpoint",
           "n_features": int(X.shape[1])}


# ---------------------------------------------------------------- stage 5 ---
def stage_platt(adapter, splits, args):
    """Monotone probability calibration, PER ARCHITECTURE.

    g(p) = sigmoid(a*logit(p) + b), fitted so per-frame detection counts match
    the real ones. Because it is monotone it cannot reorder candidates and
    therefore cannot move AUC -- which is exactly why applying one head's map to
    another is silent, and why these are never shared.

    TWO TARGETS, AND THE DIFFERENCE MATTERS. The default fit matches the
    EXPECTED count, sum_i g(p_i). The generator does not emit that: it emits a
    Bernoulli REALISATION of it, and the structural criteria score what is
    emitted. `--match-realized` therefore scales the target by kappa, RUNS the
    generator, and keeps the kappa whose realised count lands closest -- a
    measurement rather than a correction factor taken on faith. It needs an RCS
    calibration to already exist (the generator loads one), so run it after
    stage 6 rather than before.
    """
    import pickle

    from rear.calibration.platt import (apply_detection_calibration,
                                        fit_detection_calibration)
    head = _head_of(adapter, args)
    fit_dir = _fit_dir(adapter.name)
    scores_path = fit_dir / f"detection_scores_val_{head}.npz"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found -- run\n"
                         f"  python scripts/train_heads.py --dataset {adapter.name} --head {head}\n"
                         f"first; it writes this file.")
    d = np.load(scores_path)
    tol, ang = _operating_point(adapter)
    ids = splits["val"] if args.max_frames is None else splits["val"][:args.max_frames]
    target, _assoc = _association_targets(adapter, ids, tol, ang, fit_dir, "val", args)

    result = fit_detection_calibration(d["p"], d["frame"], target)
    p_, fr = d["p"], d["frame"]
    n_fr = len(set(fr.tolist()))
    raw = float(np.mean([p_[fr == k].sum() for k in range(len(ids)) if (fr == k).any()]))
    tgt = float(np.mean([target.get(k, 0) for k in range(len(ids))]))
    exp_of = lambda r: float(np.mean([apply_detection_calibration(p_[fr == k], r).sum()
                                      for k in range(len(ids)) if (fr == k).any()]))
    print(f"\n  mean expected detections per frame ({n_fr} frames):")
    print(f"    uncalibrated head output           {raw:>10.1f}")
    print(f"    with the fitted map                {exp_of(result):>10.1f}")
    print(f"    real associated in-FOV target      {tgt:>10.1f}")

    out_path = _weights_dir(adapter, args) / f"detection_calibration_{head}.pkl"
    if getattr(args, "match_realized", False):
        from rear.generate.generate_radar_frame import _ASSETS, generate_for_frame
        probe = ids[:args.realized_frames]
        print(f"\n  matching the REALISED count on {len(probe)} probe frames "
              f"(target {tgt:.1f} surface/frame)")
        print(f"    {'kappa':>7}{'a':>9}{'b':>9}{'expected':>10}{'realised':>10}")
        best, best_err = None, None
        for kappa in (0.45, 0.55, 0.65, 0.75, 0.90, 1.00):
            r = fit_detection_calibration(p_, fr, {k: v * kappa for k, v in target.items()})
            with open(out_path, "wb") as f:     # the generator reads from disk
                pickle.dump(r, f)
            _ASSETS.clear()                      # drop the cached asset bundle
            n_surf = []
            for fid in probe:
                try:
                    frame = adapter.load_frame(fid)
                except (KeyError, OSError, ValueError):
                    continue
                g = generate_for_frame(adapter, frame, seed=0, architecture=head,
                                       material_device=args.material_device)
                n_surf.append(int((~g.is_clutter).sum()))
            realised = float(np.mean(n_surf)) if n_surf else float("nan")
            print(f"    {kappa:>7.2f}{r['a']:>9.3f}{r['b']:>9.3f}"
                  f"{exp_of(r):>10.1f}{realised:>10.1f}", flush=True)
            err = abs(realised - tgt)
            if best_err is None or err < best_err:
                best, best_err = r, err
        result = best
        print(f"    -> kept a={result['a']:.4f} b={result['b']:.4f} "
              f"(realised within {best_err:.1f} of target)")

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n  a={result['a']:.4f} b={result['b']:.4f} over {result['n_frames']} val frames")
    print(f"  Saved {out_path}")
    return {"a": result["a"], "b": result["b"], "artifact": _rel(out_path),
            "head": head, "matched_realized": bool(getattr(args, "match_realized", False))}


# ---------------------------------------------------------------- stage 6 ---
def stage_rcs_calibration(adapter, splits, args):
    """Two-piece quantile map from the head's conditional MEAN onto the real
    RCS population's marginal shape. PER ARCHITECTURE: piece (a) bins
    candidates by the head's own PREDICTED value, so its bin edges are that
    head's prediction quantiles and do not transfer.

    KEEP IT ENABLED in production. Disabling it recovers +0.0107 swap-test Car
    AP but scores 0.0000 downstream and pushes RCS Wasserstein 2.544 -> 4.759,
    failing a criterion that was passing. The trade is strictly negative.
    """
    import pickle

    from rear.calibration.rcs_calibration import fit_rcs_calibration
    head = _head_of(adapter, args)
    fit_dir = _fit_dir(adapter.name)
    pred_path = fit_dir / f"rcs_pred_train_{head}.npz"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found -- run\n"
                         f"  python scripts/train_heads.py --dataset {adapter.name} --head {head}\n"
                         f"first; it writes this file.")
    d = np.load(pred_path)
    tol, ang = _operating_point(adapter)
    ids = splits["train"] if args.max_frames is None else splits["train"][:args.max_frames]
    _target, associated_pop = _association_targets(adapter, ids, tol, ang,
                                                   fit_dir, "train", args)
    if not len(associated_pop):
        raise SystemExit("no associated in-FOV real detections on train -- "
                         "the RCS marginal cannot be calibrated.")
    cal = fit_rcs_calibration(d["pred"], d["winning"], associated_pop)
    out_path = _weights_dir(adapter, args) / f"rcs_calibration_{head}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(cal, f)
    print(f"  predictions           n={len(d['pred']):,}  mean {d['pred'].mean():+.2f} dBsm")
    print(f"  winning targets       n={cal['fit_n_pred']:,}  mean {d['winning'].mean():+.2f} dBsm")
    print(f"  associated population n={cal['fit_n_assoc']:,}  mean {associated_pop.mean():+.2f} dBsm")
    print(f"  (the gap between the last two is the SELECTION EFFECT the quantile "
          f"map corrects: the head is trained on argmax winners, the generator "
          f"has to reproduce the whole associated population)")
    print(f"  Saved {out_path}")
    return {"enabled": True, "artifact": _rel(out_path), "head": head,
            "fit_n_pred": cal["fit_n_pred"], "fit_n_assoc": cal["fit_n_assoc"]}


# ---------------------------------------------------------------- stage 7 ---
def stage_clutter(adapter, splits, args):
    from rear.models.clutter_process import fit_clutter_process
    tol, ang = _operating_point(adapter)
    ids = splits["train"] if args.max_frames is None else splits["train"][:args.max_frames]
    c_range, c_az, c_rcs, c_vr = [], [], [], []
    per_frame_counts = np.zeros(len(ids))
    per_frame_speed = np.zeros(len(ids))
    for i, fid in enumerate(ids):
        frame = adapter.load_frame(fid)
        _, origin, dets, fov, matched = _match_frame(adapter, frame, tol, ang)
        per_frame_speed[i] = float(np.linalg.norm(frame.ego_velocity))
        clutter = fov & ~matched
        per_frame_counts[i] = int(clutter.sum())
        if not clutter.any():
            continue
        rcs, vr_comp = radar_rcs_and_vr_comp(adapter, frame)
        rel = dets[clutter] - origin
        c_range.append(np.linalg.norm(rel, axis=1))
        c_az.append(np.degrees(np.arctan2(rel[:, 1], rel[:, 0])))
        c_rcs.append(rcs[clutter])
        c_vr.append(vr_comp[clutter])
    if not c_range:
        raise SystemExit("no unmatched in-FOV real detections on train -- "
                         "nothing to fit a clutter process from.")
    c_range = np.concatenate(c_range); c_az = np.concatenate(c_az)
    c_rcs = np.concatenate(c_rcs); c_vr = np.concatenate(c_vr)

    proc, rep = fit_clutter_process(
        c_range, c_az, c_rcs, c_vr, per_frame_counts,
        azimuth_limits=(-adapter.radar_fov_degrees / 2, adapter.radar_fov_degrees / 2),
        per_frame_ego_speed=per_frame_speed, speed_conditioning=True)
    print(f"  rate {proc.rate_mean:.1f} +- {proc.rate_std:.1f} clutter/frame")
    print(f"  range bins {rep['n_bins']} (merged {rep['bins_merged']}), "
         f"min bin size {rep['min_bin_size']}")
    if rep["speed_factor"] is not None:
        print(f"  speed-rate factors: {np.round(rep['speed_factor'], 3).tolist()}")

    # Elevation pools -- see rear/models/clutter_process.py's module docstring:
    # NOT a validated component (never null-controlled), only a per-range-bin
    # empirical pool so the generator has SOME z to draw. Needs each clutter
    # detection's full 3D position (not just range), so this re-walks frames
    # rather than reusing c_range/c_az/c_rcs/c_vr above.
    el_all = []
    idx_all = []
    for i, fid in enumerate(ids):
        frame = adapter.load_frame(fid)
        _, origin, dets, fov, matched = _match_frame(adapter, frame, tol, ang)
        clutter = fov & ~matched
        if not clutter.any():
            continue
        rel = dets[clutter] - origin
        el = np.degrees(np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1])))
        rng = np.linalg.norm(rel, axis=1)
        b = np.clip(np.digitize(rng, proc.bin_edges) - 1, 0, len(proc.bin_rcs) - 1)
        el_all.append(el); idx_all.append(b)
    el_all = np.concatenate(el_all); idx_all = np.concatenate(idx_all)
    el_pools = [el_all[idx_all == b] for b in range(len(proc.bin_rcs))]
    empty = [b for b, p in enumerate(el_pools) if len(p) == 0]
    if empty:
        raise SystemExit(f"empty elevation pool in range bin(s) {empty} -- "
                         f"too little train data in that range to resample from; "
                         f"widen DEFAULT_RANGE_BIN_M or gather more frames.")
    print(f"  elevation pools: {len(el_pools)} bins, "
         f"min {min(len(p) for p in el_pools):,}, max {max(len(p) for p in el_pools):,}")

    out_path = _weights_dir(adapter, args) / "clutter_process.pkl"
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump({"process": proc, "bin_elevation": el_pools, "fit_report": rep,
                    "fov_deg": adapter.radar_fov_degrees, "elevation_is_validated": False,
                    "vr_convention": "compensated"}, f)
    print(f"  Saved {out_path}")
    return {"artifact": _rel(out_path),
           "rate_mean": proc.rate_mean, "rate_std": proc.rate_std,
           "n_range_bins": rep["n_bins"], "speed_conditioned": rep["speed_factor"] is not None,
           "vr_convention": "compensated", "elevation_validated": False}


FITTERS = {"candidates": stage_candidates, "association": stage_association,
          "soft_labels": stage_soft_labels, "scaling": stage_scaling,
          "platt": stage_platt, "rcs_calibration": stage_rcs_calibration,
          "clutter": stage_clutter}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", choices=STAGES, help="run one stage only")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames per split (smoke-testing knob)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit; touches nothing")
    ap.add_argument("--weights-dir", default=None,
                    help="override where platt/rcs_calibration/clutter write "
                         "their artifact (default: this dataset's "
                         "fitted.weights_dir). Use this for a smoke test so "
                         "it cannot overwrite a production profile such as "
                         "weights/vod/.")
    ap.add_argument("--head", default=None,
                    help="architecture the platt and rcs_calibration stages act "
                         "on (default: fitted.heads.implementation). Both are "
                         "per-head artifacts and are never shared between "
                         "architectures.")
    ap.add_argument("--match-realized", action="store_true",
                    help="platt stage: calibrate the REALISED surface count "
                         "rather than the expected one, by running the "
                         "generator. Needs an RCS calibration to exist, so run "
                         "it after --stage rcs_calibration.")
    ap.add_argument("--realized-frames", type=int, default=60,
                    help="probe frames used to measure the realised count")
    ap.add_argument("--material-device", default="cuda",
                    help="where a segmentation model runs during --match-realized")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the per-frame association targets instead "
                         "of reusing the cached (head-independent) ones")
    args = ap.parse_args()

    adapter = load_adapter(args.dataset)
    if not adapter.has_real_radar():
        raise SystemExit(
            f"{args.dataset} has no real radar, so nothing can be fitted on it.\n"
            f"Point its config at a donor with `fitted_from: <name>` instead -- "
            f"that is what configs/datasets/kitti_carla.yaml does, and the "
            f"sim2real gap it creates is this framework's whole subject, not "
            f"an implementation detail.")
    splits = adapter.sequence_splits()
    missing = [s for s in ("train", "val", "test") if s not in splits]
    if missing:
        raise SystemExit(f"{adapter.name}.sequence_splits() is missing {missing} -- "
                         f"fit_dataset.py needs sequence-level train/val/test splits.")

    todo = [args.stage] if args.stage else list(STAGES)
    print(__doc__.split("THE DEPENDENCY")[0])
    print(f"dataset : {adapter}\nstages  : {' -> '.join(todo)}")
    if args.dry_run:
        return

    fit_dir = _fit_dir(args.dataset)
    fitted = {}
    prior = fit_dir / "fitted.json"
    if prior.exists():
        fitted = json.loads(prior.read_text())
    for s in todo:
        print(f"\n=== {s} " + "=" * (66 - len(s)))
        fitted[s] = FITTERS[s](adapter, splits, args)
        prior.write_text(json.dumps(fitted, indent=2, default=str))
    print(f"\nWrote {prior}. Paste the relevant values into the `fitted:` "
         f"block of configs/datasets/{args.dataset}.yaml (see each stage's "
         f"printed instructions above) -- this script never edits your YAML "
         f"for you.")


if __name__ == "__main__":
    main()
