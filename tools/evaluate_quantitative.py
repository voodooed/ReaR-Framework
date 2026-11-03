#!/usr/bin/env python
"""Quantitative validation of generated radar against REAL paired radar.

    python tools/evaluate_quantitative.py --dataset vod --split test --out output/quantitative

Re-runs, in one pass and against the currently shipped weights, every
distributional and structural metric this project used to judge generated
radar. Nothing here fits, calibrates or selects anything -- it is measurement
only.

WHY THIS RUNS ON VoD AND NOT KITTI-CARLA
-----------------------------------------
Every metric below is a COMPARISON against real radar. KITTI-CARLA has no real
radar (`has_real_radar()` is False) -- that is the entire premise of the
sim2real problem this package exists for -- so there is nothing on the
simulated side to compare a generated cloud to. The generator's quality is
therefore measured on the one dataset with real paired LiDAR+radar (VoD, on
its held-out test sequences), and the KITTI-CARLA output is validated
qualitatively instead. Reporting a "distributional score" for KITTI-CARLA
would require inventing the very ground truth the dataset lacks.

THE PROTOCOL (fixed, and the same one the recorded numbers used)
-----------------------------------------------------------------
* GENERATED on the held-out TEST split, one frame per real frame, seeded.
* REAL is FOV-masked to the same wedge the candidate enumeration masks to.
  Comparing FOV-masked generated against unmasked real biases every count and
  every marginal -- 2.19% of VoD test sits outside the wedge.
* THE NULL is real-vs-real: real(train) against real(val), both real, the same
  statistic. It measures how far two disjoint sets of REAL driving sequences
  sit from each other, which is the natural floor: a generated distribution
  cannot be expected to sit closer to real test data than one batch of real
  data sits to another.
* RATIO = model-vs-real / real-vs-real, judged against the project-wide 2.0x
  convention. A ratio <= 2.0 is a PASS.
* Wasserstein and KS are sample-size biased and these sets differ by an order
  of magnitude, so every distance is computed at a COMMON subsample size over
  repeated draws (mean +- sd reported). Full-sample values are recorded
  alongside so the effect of matching is visible rather than assumed away.

At n=60,000 a KS p-value rejects identity for the real-vs-real null too, so
p-values carry no information at this scale -- only effect sizes and the ratio
against the null do. No p-value is reported anywhere in this script.

WHAT THIS SCRIPT DOES NOT MEASURE
----------------------------------
Downstream detection transfer (the E1-E5 detector-training experiments and the
swap tests). Those need a trained 3D detector and days of GPU time per arm;
they are recorded results, cited in the generated report from the source
project's own result files rather than recomputed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import linprog  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import skew, wasserstein_distance  # noqa: E402

from rear.data.base import load_adapter  # noqa: E402
from rear.generate.generate_radar_frame import generate_for_frame  # noqa: E402
from rear.validation.distributional import (  # noqa: E402
    PASS_RATIO, classifier_two_sample, matched_distance, ratio_verdict)
from rear.models.base import available_architectures  # noqa: E402
from rear.validation.structural_criteria import (  # noqa: E402
    DEFAULT_THRESHOLDS, LITERAL_HIT_RATE_BAR, evaluate as eval_structural)

SEED = 0
BOX_MARGIN_M = 0.5           # same margin the soft labels and every prior
                             # per-object measurement in this project use
EVAL_CLASSES = ("Car", "Pedestrian", "Cyclist")
HIT_CLASSES = ("Car", "Pedestrian", "Cyclist", "rider", "motor")

# 2-D joint-density EMD protocol.
EMD_BINS, EMD_N, EMD_REPS = 20, 20_000, 15
# Real VoD radar reaches ~100 m (23.7% of in-FOV detections sit beyond 55 m),
# so the joint grid must too -- a 55 m grid would pile a quarter of both
# distributions into one edge row and hide any disagreement out there.
EMD_RANGE_MAX = 100.0

RANGE_BINS = np.array([0.0, 15.0, 30.0, 50.0, 75.0, 110.0])
OBJ_RANGE_BINS = np.array([0.0, 15.0, 30.0, 120.0])

MARGINALS = ("range", "azimuth", "elevation", "RCS", "radial velocity",
             "points/frame", "points/object")

FEAT_NAMES = [
    "n_points", "range_mean", "range_std", "range_p10", "range_p50", "range_p90",
    "range_skew", "az_mean", "az_absmean", "az_std", "az_p90abs",
    "el_mean", "el_std", "el_spread",
    "rcs_mean", "rcs_std", "rcs_p10", "rcs_p50", "rcs_p90",
    "vr_mean", "vr_std", "vr_spread", "vr_frac_static",
    "nn1_mean", "nn1_std", "nn3_mean",
    "frac_r_lt15", "frac_r_15_30", "frac_r_gt30", "frac_az_lt20",
    "corr_range_rcs", "corr_az_vr", "corr_range_vr",
]


# ===========================================================================
# geometry helpers
# ===========================================================================
def polar(xyz, origin):
    rel = np.asarray(xyz, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    rng = np.linalg.norm(rel, axis=1)
    az = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    el = np.degrees(np.arctan2(rel[:, 2], np.hypot(rel[:, 0], rel[:, 1])))
    return rng, az, el


def points_in_box(pts, par, margin=BOX_MARGIN_M):
    """par = [x, y, z, length, width, height, rotation_z]."""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    x, y, z, length, width, height, rot = par
    d = np.asarray(pts)[:, :3] - np.array([x, y, z])
    c, s = np.cos(-rot), np.sin(-rot)
    lx = d[:, 0] * c - d[:, 1] * s
    ly = d[:, 0] * s + d[:, 1] * c
    return ((np.abs(lx) <= length / 2 + margin)
            & (np.abs(ly) <= width / 2 + margin)
            & (np.abs(d[:, 2]) <= height / 2 + margin))


# ===========================================================================
# collection
# ===========================================================================
def collect(adapter, ids, seed=SEED, verbose=True, head=None,
            material_device=None):
    """Generate one radar frame per real frame and record BOTH streams.

    Returns one dict per stream (`gen`, `real`) plus the frame's annotated
    boxes, all aligned by a per-detection `frame` index.
    """
    G = {k: [] for k in ("rng", "az", "el", "rcs", "vr", "frame")}
    R = {k: [] for k in ("rng", "az", "el", "rcs", "vr", "frame")}
    g_xyz, r_xyz = [], []
    g_clutter = []
    g_n, r_n, origins, ego_speed = [], [], [], []
    bx_frame, bx_cls, bx_par = [], [], []
    rcs_i, vr_i = 3, 4          # VoD radar: x,y,z,rcs,v_r,v_r_comp,time
    cols = adapter.config["sensors"].get("radar_columns")
    if cols:
        rcs_i, vr_i = cols.index("rcs"), cols.index("v_r")

    t0 = time.time()
    for i, fid in enumerate(ids):
        frame = adapter.load_frame(fid)
        gen = generate_for_frame(adapter, frame, seed=seed + i,
                                 architecture=head,
                                 material_device=material_device)
        T = frame.calibration.get("radar_to_lidar")
        origin = (np.asarray(T)[:3, 3] if T is not None
                  else np.asarray(adapter.radar_mount_offset, dtype=np.float64))

        p = gen.points
        rng, az, el = polar(p[:, :3], origin) if len(p) else (np.zeros(0),) * 3
        G["rng"].append(rng); G["az"].append(az); G["el"].append(el)
        G["rcs"].append(p[:, 3] if len(p) else np.zeros(0))
        G["vr"].append(p[:, 4] if len(p) else np.zeros(0))
        G["frame"].append(np.full(len(p), i, np.int32))
        g_xyz.append(p[:, :3] if len(p) else np.zeros((0, 3)))
        g_clutter.append(gen.is_clutter)
        g_n.append(len(p))

        rp = adapter.radar_xyz_in_lidar(frame)
        rrng, raz, rel_ = polar(rp, origin) if len(rp) else (np.zeros(0),) * 3
        R["rng"].append(rrng); R["az"].append(raz); R["el"].append(rel_)
        R["rcs"].append(frame.radar[:, rcs_i]); R["vr"].append(frame.radar[:, vr_i])
        R["frame"].append(np.full(len(rp), i, np.int32))
        r_xyz.append(rp)
        r_n.append(len(rp))
        origins.append(origin)
        ego_speed.append(float(np.linalg.norm(frame.ego_velocity)))

        for b in (frame.boxes or []):
            bx_frame.append(i); bx_cls.append(b.class_name)
            bx_par.append([b.x, b.y, b.z, b.length, b.width, b.height, b.rotation_z])

        if verbose and ((i + 1) % 50 == 0 or i == len(ids) - 1):
            el_s = time.time() - t0
            print(f"    frame {i + 1}/{len(ids)}  ({el_s / (i + 1):.2f}s/frame, "
                  f"eta {(len(ids) - i - 1) * el_s / (i + 1) / 60:.1f} min)", flush=True)

    def pack(D, xyz, n):
        out = {k: np.concatenate(v) if len(v) else np.zeros(0) for k, v in D.items()}
        out["xyz"] = np.concatenate(xyz) if len(xyz) else np.zeros((0, 3))
        out["n_per_frame"] = np.array(n, np.int32)
        return out

    gen_d, real_d = pack(G, g_xyz, g_n), pack(R, r_xyz, r_n)
    gen_d["is_clutter"] = np.concatenate(g_clutter) if g_clutter else np.zeros(0, bool)
    meta = {"origin": np.array(origins, np.float64),
            "ego_speed": np.array(ego_speed, np.float64),
            "box_frame": np.array(bx_frame, np.int32),
            "box_cls": np.array(bx_cls),
            "box_par": np.array(bx_par, np.float64) if bx_par else np.zeros((0, 7)),
            "n_frames": len(ids)}
    return gen_d, real_d, meta


def collect_real_fast(adapter, split, verbose=True):
    """Real radar for the real-vs-real null: marginals, xyz, and boxes.

    Reads the radar .bin, the two calib files and the label file directly.
    Going through the adapter's full frame loader would also decode the camera
    JPEG and the ~178k-point LiDAR sweep (~0.95 s/frame vs ~0.01 s here), and
    neither is needed for a real-radar marginal.

    Returns the same shape of (stream, meta) pair as `collect`, so
    `per_object` runs on it unchanged -- which is what lets the points/object
    marginal be scored against a real per-OBJECT null instead of borrowing
    the per-frame one (different scale entirely, and the ratio against it
    would be meaningless).
    """
    from data.vod_loader import (RADAR_N_COLS, VoDDataset, _parse_label_file,
                                 _read_bin_points, load_calibration)
    ds = VoDDataset(adapter.root, split=split)
    R = {k: [] for k in ("rng", "az", "el", "rcs", "vr", "frame")}
    xyz, origins, n_per_frame = [], [], []
    bx_frame, bx_cls, bx_par = [], [], []
    t0 = time.time()
    for i, fid in enumerate(ds.frame_ids):
        radar = _read_bin_points(ds.radar_root / "velodyne" / f"{fid}.bin", RADAR_N_COLS)
        calib = load_calibration(ds.lidar_root / "calib" / f"{fid}.txt",
                                 ds.radar_root / "calib" / f"{fid}.txt")
        T = calib["radar_to_lidar"]
        origin = T[:3, 3]
        h = np.concatenate([radar[:, :3], np.ones((len(radar), 1))], axis=1)
        p = (T @ h.T).T[:, :3]
        rng, az, el = polar(p, origin)
        R["rng"].append(rng); R["az"].append(az); R["el"].append(el)
        R["rcs"].append(radar[:, 3]); R["vr"].append(radar[:, 4])
        R["frame"].append(np.full(len(radar), i, np.int32))
        xyz.append(p.astype(np.float32))
        origins.append(origin)
        n_per_frame.append(len(radar))
        try:
            boxes = _parse_label_file(ds.lidar_root / "label_2" / f"{fid}.txt",
                                      calib["lidar_to_camera"])
        except (OSError, ValueError, KeyError):
            boxes = []
        for b in boxes:
            bx_frame.append(i); bx_cls.append(b.class_name)
            bx_par.append([b.x, b.y, b.z, b.length, b.width, b.height, b.rotation_z])
    out = {k: np.concatenate(v) for k, v in R.items()}
    out["xyz"] = np.concatenate(xyz)
    out["n_per_frame"] = np.array(n_per_frame, np.int32)
    out["n_frames"] = len(ds.frame_ids)
    meta = {"origin": np.array(origins, np.float64),
            "box_frame": np.array(bx_frame, np.int32),
            "box_cls": np.array(bx_cls),
            "box_par": np.array(bx_par, np.float64) if bx_par else np.zeros((0, 7)),
            "n_frames": len(ds.frame_ids)}
    if verbose:
        print(f"    real[{split}]: {out['n_frames']} frames, "
              f"{len(out['rng']):,} detections, {len(bx_par):,} boxes "
              f"({time.time() - t0:.1f}s)")
    return out, meta


def fov_of(d, fov_deg):
    return np.abs(np.asarray(d["az"])) <= fov_deg


def counts_per_frame_in_fov(d, n_frames, fov_deg):
    """Detections per frame, counting ONLY what is inside the FOV wedge.

    The raw `n_per_frame` recorded at collection time counts every real
    detection, including the ~2% that sit outside the wedge; generated output
    is FOV-masked by construction. Comparing the two directly would bias the
    count ratio and the points/frame marginal by exactly that excess -- the
    specific mistake this module's protocol exists to prevent -- so both sides
    are re-counted here after masking.
    """
    m = fov_of(d, fov_deg)
    return np.bincount(np.asarray(d["frame"])[m].astype(int),
                       minlength=n_frames).astype(float)


# ===========================================================================
# 2-D EMD (exact transportation LP over grid cells)
# ===========================================================================
def _emd_cost(nb):
    gx, gy = np.meshgrid(np.arange(nb), np.arange(nb), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    return np.linalg.norm(pts[:, None] - pts[None], axis=2)


_EMD_A = {}


def _emd_constraints(n):
    if n in _EMD_A:
        return _EMD_A[n]
    rows, cols = [], []
    for i in range(n):
        rows += [i] * n; cols += list(range(i * n, (i + 1) * n))
    for j in range(n):
        rows += [n + j] * n; cols += list(range(j, n * n, n))
    A = coo_matrix((np.ones(2 * n * n), (rows, cols)), shape=(2 * n, n * n))
    _EMD_A[n] = A
    return A


def emd_2d(P, Q, C):
    """Exact EMD between two 2-D histograms -- the standard image-EMD
    transportation LP. Ground distance is Euclidean in CELL units, so the
    value is 'mean cells of mass transport'."""
    n = P.size
    p, q = P.ravel() / P.sum(), Q.ravel() / Q.sum()
    r = linprog(C.ravel(), A_eq=_emd_constraints(n),
                b_eq=np.concatenate([p, q]), bounds=(0, None), method="highs")
    if r.status != 0:
        raise RuntimeError(f"EMD LP failed: {r.message}")
    return float(r.fun)


def hist2d(rng_v, az_v, fov_deg, nb=EMD_BINS):
    H, _, _ = np.histogram2d(
        np.clip(rng_v, 0, EMD_RANGE_MAX - 1e-6),
        np.clip(az_v, -fov_deg, fov_deg - 1e-6),
        bins=[np.linspace(0, EMD_RANGE_MAX, nb + 1),
              np.linspace(-fov_deg, fov_deg, nb + 1)])
    return H


def matched_emd(a_rng, a_az, b_rng, b_az, C, fov_deg, n=EMD_N, reps=EMD_REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        ia = rng.choice(len(a_rng), min(n, len(a_rng)), replace=False)
        ib = rng.choice(len(b_rng), min(n, len(b_rng)), replace=False)
        vals.append(emd_2d(hist2d(a_rng[ia], a_az[ia], fov_deg),
                           hist2d(b_rng[ib], b_az[ib], fov_deg), C))
    return float(np.mean(vals)), float(np.std(vals))


# ===========================================================================
# per-object structure
# ===========================================================================
def per_object(d, meta, fov_deg, classes=EVAL_CLASSES):
    """Detections per annotated object, and each object's spatial extent."""
    m = fov_of(d, fov_deg)
    xyz, fr = d["xyz"][m], d["frame"][m]
    order = np.argsort(fr, kind="stable")
    xyz, fr = xyz[order], fr[order]
    starts = np.searchsorted(fr, np.arange(meta["n_frames"] + 1))
    counts, extents, cls_out, rng_out = [], [], [], []
    for k in range(len(meta["box_par"])):
        c = meta["box_cls"][k]
        if c not in classes:
            continue
        i = meta["box_frame"][k]
        par = meta["box_par"][k]
        lo, hi = starts[i], starts[i + 1]
        pts = xyz[lo:hi]
        sel = points_in_box(pts, par)
        counts.append(int(sel.sum()))
        cls_out.append(c)
        o = meta["origin"][i]
        rng_out.append(float(np.linalg.norm(par[:3] - o)))
        if sel.sum() >= 2:
            q = pts[sel]
            extents.append(float(np.linalg.norm(q.max(axis=0) - q.min(axis=0))))
        else:
            extents.append(np.nan)
    return (np.array(counts, float), np.array(extents, float),
            np.array(cls_out), np.array(rng_out, float))


def hit_rate(d, meta, fov_deg, classes=HIT_CLASSES):
    """Fraction of annotated objects that receive at least one detection."""
    m = fov_of(d, fov_deg)
    xyz, fr = d["xyz"][m], d["frame"][m]
    order = np.argsort(fr, kind="stable")
    xyz, fr = xyz[order], fr[order]
    starts = np.searchsorted(fr, np.arange(meta["n_frames"] + 1))
    per_cls, hit, tot = {}, 0, 0
    for k in range(len(meta["box_par"])):
        c = meta["box_cls"][k]
        if c not in classes:
            continue
        i = meta["box_frame"][k]
        lo, hi = starts[i], starts[i + 1]
        got = bool(points_in_box(xyz[lo:hi], meta["box_par"][k]).any())
        per_cls.setdefault(c, [0, 0])
        per_cls[c][1] += 1
        per_cls[c][0] += int(got)
        tot += 1
        hit += int(got)
    return hit, tot, per_cls


# ===========================================================================
# classifier two-sample test
# ===========================================================================
def frame_features(rng_v, az_v, el_v, rcs_v, vr_v, xyz):
    """Frame-level summary statistics -- the classifier never sees raw points."""
    n = len(rng_v)
    if n == 0:
        return np.full(len(FEAT_NAMES), np.nan)

    def corr(a, b):
        if n < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    if n >= 4:
        dd, _ = cKDTree(xyz).query(xyz, k=4)
        nn1_m, nn1_s, nn3_m = dd[:, 1].mean(), dd[:, 1].std(), dd[:, 1:].mean()
    else:
        nn1_m = nn1_s = nn3_m = np.nan
    return np.array([
        n, rng_v.mean(), rng_v.std(), *np.percentile(rng_v, [10, 50, 90]),
        float(skew(rng_v)) if n > 2 else np.nan,
        az_v.mean(), np.abs(az_v).mean(), az_v.std(),
        float(np.percentile(np.abs(az_v), 90)),
        el_v.mean(), el_v.std(),
        float(np.percentile(el_v, 90) - np.percentile(el_v, 10)),
        rcs_v.mean(), rcs_v.std(), *np.percentile(rcs_v, [10, 50, 90]),
        vr_v.mean(), vr_v.std(),
        float(np.percentile(vr_v, 90) - np.percentile(vr_v, 10)),
        float((np.abs(vr_v) < 0.5).mean()),
        nn1_m, nn1_s, nn3_m,
        float((rng_v < 15).mean()), float(((rng_v >= 15) & (rng_v < 30)).mean()),
        float((rng_v >= 30).mean()), float((np.abs(az_v) < 20).mean()),
        corr(rng_v, rcs_v), corr(az_v, vr_v), corr(rng_v, vr_v),
    ], dtype=np.float64)


def features_from(d, n_frames, fov_deg):
    m = fov_of(d, fov_deg)
    fr = d["frame"][m]
    order = np.argsort(fr, kind="stable")
    fr = fr[order]
    cols = {k: np.asarray(d[k])[m][order] for k in ("rng", "az", "el", "rcs", "vr")}
    xyz = d["xyz"][m][order] if "xyz" in d else None
    starts = np.searchsorted(fr, np.arange(n_frames + 1))
    X = np.zeros((n_frames, len(FEAT_NAMES)))
    for i in range(n_frames):
        lo, hi = starts[i], starts[i + 1]
        X[i] = frame_features(
            cols["rng"][lo:hi].astype(np.float64), cols["az"][lo:hi].astype(np.float64),
            cols["el"][lo:hi].astype(np.float64), cols["rcs"][lo:hi].astype(np.float64),
            cols["vr"][lo:hi].astype(np.float64),
            xyz[lo:hi].astype(np.float64) if xyz is not None
            else np.zeros((hi - lo, 3)))
    return X


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vod")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap frames (smoke test); default = the WHOLE split")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(REPO_ROOT / "output" / "quantitative"))
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--head", default=None, choices=available_architectures(),
                    help="override fitted.heads.implementation for this run")
    ap.add_argument("--material-device", default="cuda",
                    help="where a segmentation model runs when the dataset "
                         "needs one to build the material block. CPU works but "
                         "is ~15x slower per frame.")
    ap.add_argument("--no-cache", action="store_true",
                    help="force regeneration instead of reusing the cached "
                         "collection for this (split, n, seed)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(args.dataset, split=args.split)
    if not adapter.has_real_radar():
        raise SystemExit(
            f"{args.dataset} has no real radar, so there is nothing to validate "
            f"generated radar AGAINST. Every metric in this script is a "
            f"comparison against real detections. Run it on a dataset with "
            f"paired real radar (vod); KITTI-CARLA output is validated "
            f"qualitatively instead -- see tools/visualize_best.py.")
    fov_deg = adapter.radar_fov_degrees
    ids = adapter.sequence_splits()[args.split]
    if args.limit:
        ids = ids[:args.limit]

    print("=" * 78)
    print(f"QUANTITATIVE VALIDATION -- {adapter.name} / {args.split}")
    print("=" * 78)
    print(f"  frames        : {len(ids)}")
    # radar_fov_degrees is a HALF-angle -- within_radar_fov() keeps
    # |azimuth| <= radar_fov_deg, so 60.0 means a +-60 deg / 120 deg sector.
    print(f"  FOV mask      : +-{fov_deg:.0f} deg (both sides of every comparison)")
    print(f"  pass ratio    : {PASS_RATIO}x the real-vs-real null")
    print(f"  seed          : {args.seed}\n")

    # Generation dominates the runtime (~1.8 s/frame); the metrics on top of
    # it are seconds. Caching the collected arrays makes re-scoring after a
    # change to a metric definition cheap instead of a full regeneration.
    # The head implementation is part of the cache identity: a collection
    # generated with one head is not reusable for another, and silently
    # reusing it would attribute one head's cloud to the other.
    _impl = args.head or adapter.fitted("heads", "implementation", default="prism")
    cache = (out_dir /
             f"_collected_{args.split}_{len(ids)}_seed{args.seed}_{_impl}.npz")
    if cache.exists() and not args.no_cache:
        print(f"  [1/3] loading cached collection {cache.name}")
        z = np.load(cache, allow_pickle=True)
        gen = {k[4:]: z[k] for k in z.files if k.startswith("gen_")}
        real = {k[5:]: z[k] for k in z.files if k.startswith("real_")}
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
        meta["n_frames"] = int(meta["n_frames"])
    else:
        print("  [1/3] generating + collecting test split...")
        gen, real, meta = collect(adapter, ids, seed=args.seed, head=_impl,
                                  material_device=args.material_device)
        np.savez_compressed(
            cache,
            **{f"gen_{k}": v for k, v in gen.items()},
            **{f"real_{k}": v for k, v in real.items()},
            **{f"meta_{k}": v for k, v in meta.items()})
        print(f"    cached -> {cache.name}")
    print("  [2/3] collecting real train/val for the real-vs-real null...")
    real_tr, meta_tr = collect_real_fast(adapter, "train")
    real_va, meta_va = collect_real_fast(adapter, "val")

    gm, rm = fov_of(gen, fov_deg), fov_of(real, fov_deg)
    trm, vam = fov_of(real_tr, fov_deg), fov_of(real_va, fov_deg)
    n_gen, n_real = int(gm.sum()), int(rm.sum())
    # Per-frame counts, FOV-masked on BOTH sides (see counts_per_frame_in_fov).
    npf_gen = counts_per_frame_in_fov(gen, meta["n_frames"], fov_deg)
    npf_real = counts_per_frame_in_fov(real, meta["n_frames"], fov_deg)
    npf_tr = counts_per_frame_in_fov(real_tr, real_tr["n_frames"], fov_deg)
    npf_va = counts_per_frame_in_fov(real_va, real_va["n_frames"], fov_deg)
    print(f"    generated: {n_gen:,} in-FOV detections over {meta['n_frames']} frames")
    print(f"    real     : {n_real:,} in-FOV detections "
          f"({100 * (1 - rm.mean()):.2f}% of real fell outside the wedge)")

    ROWS = []

    def row(part, name, model, null, note="", model_sd=np.nan):
        r, v = ratio_verdict(model, null)
        ROWS.append(dict(part=part, test=name, model=float(model),
                         model_sd=float(model_sd), null=float(null),
                         ratio=float(r), verdict=v, note=note))
        return ROWS[-1]

    # ---- Part 1: marginals -------------------------------------------------
    print("\n  [3/3] PART 1 -- marginal distributions vs the real-vs-real null")
    for key in MARGINALS:
        if key == "points/frame":
            a, b = npf_gen, npf_real
            na, nb = npf_tr, npf_va
        elif key == "points/object":
            cg, _, _, _ = per_object(gen, meta, fov_deg)
            cr, _, _, _ = per_object(real, meta, fov_deg)
            a, b = cg, cr
            # A real per-OBJECT null, from the same statistic on train/val
            # boxes. Borrowing the per-frame null here would compare a
            # scale-~1 quantity against a scale-~60 one and manufacture a
            # meaningless PASS.
            na, _, _, _ = per_object(real_tr, meta_tr, fov_deg)
            nb, _, _, _ = per_object(real_va, meta_va, fov_deg)
        else:
            col = {"range": "rng", "azimuth": "az", "elevation": "el",
                   "RCS": "rcs", "radial velocity": "vr"}[key]
            a, b = np.asarray(gen[col])[gm], np.asarray(real[col])[rm]
            na, nb = np.asarray(real_tr[col])[trm], np.asarray(real_va[col])[vam]
        md = matched_distance(a, b, seed=args.seed)
        nd = matched_distance(na, nb, seed=args.seed)
        note = ("counted over annotated boxes on both sides, incl. the null"
                if key == "points/object" else "")
        r = row("1 marginal", key, md["w1"], nd["w1"], note, md["w1_sd"])
        r.update(ks=md["ks"], w1_full=md["w1_full"], n_matched=md["n"],
                 null_sd=nd["w1_sd"], null_ks=nd["ks"])
        print(f"    {key:<20} W1 {md['w1']:8.4f} +-{md['w1_sd']:.4f}   "
              f"null {nd['w1']:8.4f}   ratio {r['ratio']:5.2f}x  {r['verdict']}")

    # ---- Part 2: joint structure ------------------------------------------
    print("\n  PART 2 -- joint (range x azimuth) and conditional structure")
    C = _emd_cost(EMD_BINS)
    e_m, e_sd = matched_emd(np.asarray(gen["rng"])[gm], np.asarray(gen["az"])[gm],
                            np.asarray(real["rng"])[rm], np.asarray(real["az"])[rm],
                            C, fov_deg, seed=args.seed)
    e_n, e_nsd = matched_emd(np.asarray(real_tr["rng"])[trm], np.asarray(real_tr["az"])[trm],
                             np.asarray(real_va["rng"])[vam], np.asarray(real_va["az"])[vam],
                             C, fov_deg, seed=args.seed)
    r = row("2 joint", "range x azimuth EMD (cells)", e_m, e_n,
            f"{EMD_BINS}x{EMD_BINS} grid, 0-{EMD_RANGE_MAX:.0f} m", e_sd)
    r.update(null_sd=e_nsd)
    print(f"    {'range x azimuth EMD':<20} {e_m:8.4f} +-{e_sd:.4f}   "
          f"null {e_n:8.4f}   ratio {r['ratio']:5.2f}x  {r['verdict']}")

    # RCS conditioned on range -- a marginal can agree while every conditional fails
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        sg = gm & (np.asarray(gen["rng"]) >= lo) & (np.asarray(gen["rng"]) < hi)
        sr = rm & (np.asarray(real["rng"]) >= lo) & (np.asarray(real["rng"]) < hi)
        st = trm & (np.asarray(real_tr["rng"]) >= lo) & (np.asarray(real_tr["rng"]) < hi)
        sv = vam & (np.asarray(real_va["rng"]) >= lo) & (np.asarray(real_va["rng"]) < hi)
        if min(sg.sum(), sr.sum(), st.sum(), sv.sum()) < 100:
            continue
        md = matched_distance(np.asarray(gen["rcs"])[sg], np.asarray(real["rcs"])[sr], seed=args.seed)
        nd = matched_distance(np.asarray(real_tr["rcs"])[st], np.asarray(real_va["rcs"])[sv], seed=args.seed)
        # No raw '|' in the name -- these strings land in markdown table cells.
        r = row("2 conditional", f"RCS given range {lo:.0f}-{hi:.0f} m",
                md["w1"], nd["w1"], "", md["w1_sd"])
        print(f"    RCS | {lo:.0f}-{hi:.0f} m{'':<9} {md['w1']:8.4f}   "
              f"null {nd['w1']:8.4f}   ratio {r['ratio']:5.2f}x  {r['verdict']}")

    # ---- Part 3: per-object structure -------------------------------------
    print("\n  PART 3 -- per-object structure (what a detector plausibly sees)")
    cg, eg, clsg, rngg = per_object(gen, meta, fov_deg)
    cr, er, clsr, rngr = per_object(real, meta, fov_deg)
    fin = np.isfinite(eg) & np.isfinite(er)
    md = matched_distance(eg[np.isfinite(eg)], er[np.isfinite(er)], seed=args.seed)
    print(f"    objects scored     : {len(cg)} ({', '.join(EVAL_CLASSES)})")
    print(f"    detections/object  : gen {cg.mean():.2f} +- {cg.std():.2f}   "
          f"real {cr.mean():.2f} +- {cr.std():.2f}")
    print(f"    extent (m)         : gen {np.nanmean(eg):.2f}   real {np.nanmean(er):.2f}"
          f"   W1 {md['w1']:.4f}")
    per_class = {}
    for c in EVAL_CLASSES:
        s = clsg == c
        if not s.any():
            continue
        per_class[c] = dict(
            n=int(s.sum()),
            gen_mean=float(cg[s].mean()), gen_std=float(cg[s].std()),
            real_mean=float(cr[s].mean()), real_std=float(cr[s].std()),
            gen_hit=float((cg[s] > 0).mean()), real_hit=float((cr[s] > 0).mean()))
        print(f"      {c:<12} n={s.sum():<5} gen {cg[s].mean():6.2f}  "
              f"real {cr[s].mean():6.2f}   hit gen {100 * (cg[s] > 0).mean():5.1f}%  "
              f"real {100 * (cr[s] > 0).mean():5.1f}%")
    by_range = {}
    for lo, hi in zip(OBJ_RANGE_BINS[:-1], OBJ_RANGE_BINS[1:]):
        s = (rngg >= lo) & (rngg < hi)
        if s.sum() < 5:
            continue
        by_range[f"{lo:.0f}-{hi:.0f} m"] = dict(
            n=int(s.sum()), gen_mean=float(cg[s].mean()), real_mean=float(cr[s].mean()),
            gen_hit=float((cg[s] > 0).mean()), real_hit=float((cr[s] > 0).mean()))

    # ---- structural criteria ----------------------------------------------
    print("\n  THE 7 PRE-REGISTERED STRUCTURAL CRITERIA")
    hg, tg, hg_cls = hit_rate(gen, meta, fov_deg)
    hr, tr_, hr_cls = hit_rate(real, meta, fov_deg)
    crit = eval_structural(
        counts_gen=cg, counts_real=cr,
        extents_gen=eg[np.isfinite(eg)], extents_real=er[np.isfinite(er)],
        rcs_gen=np.asarray(gen["rcs"])[gm], rcs_real=np.asarray(real["rcs"])[rm],
        n_gen_per_frame=npf_gen, n_real_per_frame=npf_real,
        hit_gen=hg, hit_total=tg)
    for k, v in crit.items():
        if k.startswith("_"):
            continue
        thr = DEFAULT_THRESHOLDS[k]
        ttxt = (f"in [{thr[1]}, {thr[2]}]" if thr[0] == "in"
                else (f"<= {thr[1]}" if thr[0] == "le" else f">= {thr[1]}"))
        print(f"    {k:<14}{v:10.4f}   {ttxt:<20}{crit['_verdicts'][k]}")
    print(f"    -> {crit['_n_pass']}/7 PASS; hard constraint "
          f"({'hit rate' }) {'OK' if crit['_hard_ok'] else 'VIOLATED'}")
    print(f"    real radar's own hit rate on the same objects: "
          f"{100 * hr / max(tr_, 1):.2f}%")

    # ---- Part 4: classifier two-sample test -------------------------------
    clf = None
    if not args.skip_classifier:
        print("\n  PART 4 -- classifier two-sample test (frame-level statistics only)")
        Xg = features_from(gen, meta["n_frames"], fov_deg)
        Xr = features_from(real, meta["n_frames"], fov_deg)
        half = meta["n_frames"] // 2
        fit = np.zeros(meta["n_frames"], bool); fit[:half] = True
        auc, uni = classifier_two_sample(Xg, Xr, fit, fit, seed=args.seed)
        # The null: real(train) vs real(val), same statistic, same construction.
        Xt = features_from(real_tr, real_tr["n_frames"], fov_deg)
        Xv = features_from(real_va, real_va["n_frames"], fov_deg)
        ft = np.zeros(real_tr["n_frames"], bool); ft[:real_tr["n_frames"] // 2] = True
        fv = np.zeros(real_va["n_frames"], bool); fv[:real_va["n_frames"] // 2] = True
        auc_n, uni_n = classifier_two_sample(Xt, Xv, ft, fv, seed=args.seed)
        top = np.argsort(-uni)[:8]
        clf = dict(auc=float(auc), auc_null=float(auc_n),
                   excess=float(auc - auc_n),
                   top_features=[{"name": FEAT_NAMES[j], "uni_auc": float(uni[j]),
                                  "null_uni_auc": float(uni_n[j])} for j in top])
        print(f"    generated vs real AUC : {auc:.4f}")
        print(f"    real(tr) vs real(va)  : {auc_n:.4f}   (the null)")
        print(f"    EXCESS                : {auc - auc_n:+.4f}")
        print(f"    most-separating statistics:")
        for j in top[:5]:
            print(f"      {FEAT_NAMES[j]:<20} uni-AUC {uni[j]:.3f}  (null {uni_n[j]:.3f})")

    # ---- persist ----------------------------------------------------------
    n_pass = sum(r["verdict"] == "PASS" for r in ROWS)
    summary = {
        "dataset": adapter.name, "split": args.split,
        # Which head produced this cloud, and which calibrations were resolved
        # for it -- both are head-specific, so a metrics file that did not
        # record them could not be attributed later.
        "implementation": _impl,
        "detection_calibration": f"detection_calibration_{_impl}.pkl",
        "rcs_calibration_file": f"rcs_calibration_{_impl}.pkl",
        "n_frames": meta["n_frames"], "seed": args.seed,
        "fov_deg": fov_deg, "pass_ratio": PASS_RATIO,
        "n_gen_detections": n_gen, "n_real_detections": n_real,
        "gen_clutter_fraction": float(gen["is_clutter"][gm].mean()) if n_gen else float("nan"),
        "real_outside_fov_pct": float(100 * (1 - rm.mean())),
        "n_null_train_frames": int(real_tr["n_frames"]),
        "n_null_val_frames": int(real_va["n_frames"]),
        "rows": ROWS, "n_pass": n_pass, "n_tests": len(ROWS),
        "structural": {k: (v if not isinstance(v, dict) else v)
                       for k, v in crit.items()},
        "structural_real_hit_rate_pct": float(100 * hr / max(tr_, 1)),
        "literal_hit_rate_bar": LITERAL_HIT_RATE_BAR,
        "per_object": {
            "n_objects": int(len(cg)),
            "gen_mean": float(cg.mean()), "gen_std": float(cg.std()),
            "real_mean": float(cr.mean()), "real_std": float(cr.std()),
            "gen_extent_mean": float(np.nanmean(eg)),
            "real_extent_mean": float(np.nanmean(er)),
            "extent_w1": float(md["w1"]),
            "per_class": per_class, "by_range": by_range,
            "hit_gen": [hg, tg], "hit_real": [hr, tr_],
            "hit_per_class_gen": {k: v for k, v in hg_cls.items()},
            "hit_per_class_real": {k: v for k, v in hr_cls.items()},
        },
        "classifier": clf,
        # FOV-masked on both sides -- see counts_per_frame_in_fov.
        "points_per_frame": {
            "gen_mean": float(npf_gen.mean()), "gen_std": float(npf_gen.std()),
            "real_mean": float(npf_real.mean()), "real_std": float(npf_real.std()),
            "real_mean_unmasked": float(real["n_per_frame"].mean()),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"\n  {n_pass} of {len(ROWS)} distributional tests at or below the "
          f"{PASS_RATIO}x null convention")
    print(f"  wrote {out_dir / 'metrics.json'}")
    return summary


if __name__ == "__main__":
    main()
