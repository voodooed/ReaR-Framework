#!/usr/bin/env python
"""Re-measure a fitted head's two outputs on held-out real data.

    python tools/evaluate_heads.py --dataset vod --head prism --out output/prism/quantitative
    python tools/evaluate_heads.py --dataset vod --head mlp   --out output/mlp/quantitative

Scores ONE architecture's joint checkpoint -- detection AUC and RCS weighted R^2
-- on the same held-out sequences and with the same definitions every recorded
value for this project used. Run it once per architecture; the two runs share
every stage except the head, which is what makes their numbers comparable.

It verifies that the checkpoint actually shipped in `weights/<dataset>/` is the
one its recorded numbers describe. A mismatch means the weights and the claims
have drifted apart, which is exactly what this script exists to rule out.

THE DETECTION METRIC IS A PROXY AUC, and the proxy is deliberately narrow:

    positive  candidate matched to a detection that itself lies on an
              annotated box -- external evidence of a true object return
    negative  candidate matched to NO detection at all
    excluded  candidates matched only to OFF-box detections. Neither
              confirmed real nor confirmed absent; leaving them in would
              blur the metric rather than sharpen it.

AUC is invariant to any monotone transform, so the Platt calibration
(`detection_calibration_<head>.pkl`) cannot move it -- it is reported raw here, and
that invariance is itself asserted as a correctness check on the calibration
being monotone at all.

The RCS metric is a WEIGHTED R^2 over matched candidates only: an unmatched
candidate has no real RCS to regress against and is excluded rather than
filled. Each example is weighted by its V3 soft label, so a low-confidence
association pulls the score less than a confident one, and the baseline it is
scored against is the weighted mean of the targets being scored (the RCS
analogue of a 0.5 AUC).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from _fit_common import (  # noqa: E402
    candidates_for_frame, operating_point, radar_rcs_and_vr_comp)
from rear.calibration.soft_labels import (  # noqa: E402
    BOX_MARGIN_M, build_soft_labels_v3_rank_weighted, distance_to_box,
    point_counts_from_features)
from rear.data.base import load_adapter  # noqa: E402
from rear.inference import HeadScorer  # noqa: E402
from rear.models.base import available_architectures  # noqa: E402


def box_conditioned_proxy(centres, radar_lidar, boxes, radar_position, tol, ang):
    """(positive, negative) masks -- see the module docstring."""
    cone = np.radians(ang)
    rel_c = np.asarray(centres, np.float64) - radar_position
    cand_range = np.linalg.norm(rel_c, axis=1)
    cand_dir = rel_c / np.maximum(cand_range, 1e-6)[:, None]

    on_box_det = np.zeros(len(radar_lidar), dtype=bool)
    for b in boxes:
        on_box_det |= distance_to_box(np.asarray(radar_lidar, np.float64), b) <= BOX_MARGIN_M

    matched_any = np.zeros(len(centres), dtype=bool)
    matched_box = np.zeros(len(centres), dtype=bool)
    if len(radar_lidar):
        rel_d = np.asarray(radar_lidar, np.float64) - radar_position
        det_range = np.linalg.norm(rel_d, axis=1)
        det_dir = rel_d / np.maximum(det_range, 1e-6)[:, None]
        for i in range(len(radar_lidar)):
            angle = np.arccos(np.clip(cand_dir @ det_dir[i], -1.0, 1.0))
            within = (np.abs(cand_range - det_range[i]) <= tol) & (angle <= cone)
            matched_any |= within
            if on_box_det[i]:
                matched_box |= within
    return matched_box, ~matched_any


def weighted_r2(target, prediction, weight):
    wm = np.average(target, weights=weight)
    ss_res = np.sum(weight * (target - prediction) ** 2)
    ss_tot = np.sum(weight * (target - wm) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vod")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--head", "--implementation", dest="head", default=None,
                    choices=available_architectures(),
                    help="which architecture to score (default: whatever "
                         "fitted.heads.implementation says)")
    ap.add_argument("--weights-dir", default=None,
                    help="read the head from here instead of the dataset's "
                         "fitted.weights_dir (smoke tests, side-by-side runs)")
    ap.add_argument("--material-device", default=None,
                    help="where a segmentation model runs, if this dataset "
                         "needs one to build the material block")
    ap.add_argument("--out", default=str(REPO_ROOT / "output" / "quantitative"))
    args = ap.parse_args()

    adapter = load_adapter(args.dataset, split=args.split)
    if not adapter.has_real_radar():
        raise SystemExit(f"{args.dataset} has no real radar: both head metrics "
                         f"are scored against real detections.")
    tol, ang = operating_point(adapter)
    ids = adapter.sequence_splits()[args.split]
    if args.limit:
        ids = ids[:args.limit]

    scorer = HeadScorer(adapter, architecture=args.head, device=args.device,
                        weights_dir=args.weights_dir,
                        material_device=args.material_device)
    info = scorer.describe()
    impl, wdir = scorer.architecture, scorer.weights_dir

    print("=" * 78)
    print(f"HEAD-LEVEL METRICS -- {adapter.name} / {args.split}")
    print("=" * 78)
    print(f"  frames            : {len(ids)}")
    print(f"  weights           : {wdir}")
    print(f"  head              : {impl}  ({info['n_params']:,} params, "
          f"seed {info['seed']})")
    print(f"  material          : {info['n_material']} columns"
          + (f" ({info['material_variant']}, {info['material_label_space']} via "
             f"{info['material_source']})" if info['n_material'] else " (none)"))
    print(f"  operating point   : {tol} m range tol, {ang} deg cone")
    print(f"  feature variant   : {info['feature_variant']} "
          f"({len(info['feature_names'])} physics features)\n")

    P, Pc, POS, NEG = [], [], [], []
    RP, RT, RW = [], [], []
    t0 = time.time()
    for i, fid in enumerate(ids):
        frame = adapter.load_frame(fid)
        cand, cfg, _ = candidates_for_frame(adapter, frame)
        if len(cand.centres) == 0:
            continue
        origin = np.asarray(cfg.radar_position, np.float64)
        radar_lidar = adapter.radar_xyz_in_lidar(frame)
        scored = scorer.score(frame, cand, cfg)
        p = scored["p_raw"]
        pc = (scored["p_calibrated"] if scored["p_calibrated"] is not None else p)
        pos, neg = box_conditioned_proxy(cand.centres, radar_lidar, frame.boxes,
                                          origin, tol, ang)
        P.append(p); Pc.append(pc)
        POS.append(pos); NEG.append(neg)

        counts = point_counts_from_features(cand.features, cfg.voxel_size)
        res = build_soft_labels_v3_rank_weighted(
            cand.centres, radar_lidar, frame.boxes, origin,
            candidate_point_counts=counts, range_tol=tol, cone_angle_deg=ang)
        m = res.matched
        if m.any():
            RP.append(scored["rcs_dbsm"][m])
            real_rcs, _ = radar_rcs_and_vr_comp(adapter, frame)
            RT.append(real_rcs[res.winning_detection[m]])
            RW.append(res.labels[m])

        if (i + 1) % 50 == 0 or i == len(ids) - 1:
            e = time.time() - t0
            print(f"    frame {i + 1}/{len(ids)}  ({e / (i + 1):.2f}s/frame, "
                  f"eta {(len(ids) - i - 1) * e / (i + 1) / 60:.1f} min)", flush=True)

    p = np.concatenate(P); pc = np.concatenate(Pc)
    pos, neg = np.concatenate(POS), np.concatenate(NEG)
    keep = pos | neg
    auc = float(roc_auc_score(pos[keep], p[keep]))
    auc_cal = float(roc_auc_score(pos[keep], pc[keep]))

    rp, rt, rw = np.concatenate(RP), np.concatenate(RT), np.concatenate(RW)
    r2 = weighted_r2(rt, rp, rw)
    base = weighted_r2(rt, np.full_like(rt, np.average(rt, weights=rw)), rw)
    rmse = float(np.sqrt(np.sum(rw * (rt - rp) ** 2) / rw.sum()))
    brmse = float(np.sqrt(np.sum(rw * (rt - np.average(rt, weights=rw)) ** 2) / rw.sum()))
    corr = float(np.corrcoef(rp, rt)[0, 1])

    print(f"\n  DETECTION HEAD")
    print(f"    candidates scored   : {int(keep.sum()):,} "
          f"({int(pos.sum()):,} proxy-positive, {int(neg.sum()):,} proxy-negative)")
    print(f"    excluded (off-box)  : {int((~keep).sum()):,}")
    print(f"    test AUC (raw)      : {auc:.4f}")
    print(f"    test AUC (Platt)    : {auc_cal:.4f}   "
          f"[must equal raw -- Platt is monotone] "
          f"{'OK' if abs(auc - auc_cal) < 1e-9 else 'MISMATCH'}")
    print(f"    (raw and calibrated AUC must agree: a monotone map cannot "
          f"reorder candidates. A mismatch means the map is not monotone.)")
    print(f"\n  RCS HEAD")
    print(f"    matched candidates  : {len(rt):,}")
    print(f"    test w-R2           : {r2:.4f}   (mean-prediction baseline {base:.4f})")
    print(f"    test w-RMSE (dBsm)  : {rmse:.3f}  (baseline {brmse:.3f})")
    print(f"    corr(pred, real)    : {corr:+.4f}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    res = {"dataset": adapter.name, "split": args.split, "n_frames": len(ids),
           **info,
           "operating_point": {"range_tolerance_m": tol, "cone_angle_deg": ang},
           "detection": {"auc": auc, "auc_after_platt": auc_cal,
                         "n_scored": int(keep.sum()), "n_pos": int(pos.sum()),
                         "n_neg": int(neg.sum()), "n_excluded": int((~keep).sum())},
           "rcs": {"w_r2": float(r2), "baseline_w_r2": float(base),
                   "w_rmse_dbsm": rmse, "baseline_w_rmse_dbsm": brmse,
                   "corr_pred_real": corr, "n_matched": int(len(rt))}}
    (out_dir / "head_metrics.json").write_text(json.dumps(res, indent=1))
    print(f"\n  wrote {out_dir / 'head_metrics.json'}")


if __name__ == "__main__":
    main()
