#!/usr/bin/env python
"""ReaR — generate a radar point cloud from a LiDAR point cloud.

Two modes:

  1. Dataset/split mode — generate for every frame of a registered dataset
     (currently KITTI-CARLA; VoD needs its own loader, see USAGE.md):

         python scripts/generate.py --dataset kitti_carla --split Town01 \\
             --out outputs/gen_kitti_carla_town01 --limit 50

  2. Single-file mode — bring your own LiDAR sweep as a flat .bin/.npy array
     of shape (N, 4): x, y, z, <4th column>. Use this to run ReaR on LiDAR
     that isn't from a supported dataset adapter at all.

         python scripts/generate.py --lidar my_scan.bin --ego-velocity 8.3 0 0 \\
             --out my_scan_radar.bin

     `--lidar-format` picks what the 4th column means and, through it, which
     fitted physics profile (voxel grid, FOV, incidence-angle source) to use:
       kitti_carla  (default) 4th column = surface incidence cosine in [0,1]
       generic                4th column ignored; incidence angle is
                               estimated from local LiDAR geometry instead
                               (use this for plain x,y,z,intensity LiDAR)

Output is always (M, 5) float32: x, y, z, RCS (dBsm), v_r (m/s), in the
LiDAR frame, written as a flat binary array (read back with
`np.fromfile(path, dtype=np.float32).reshape(-1, 5)`), one file per frame.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from rear.data.base import load_adapter, load_adapter_config  # noqa: E402
from rear.generate.generate_radar_frame import generate_for_frame, generate_radar_frame  # noqa: E402
from rear.geometry.candidates import VoxelGridConfig  # noqa: E402
from rear.materials import available_label_spaces  # noqa: E402
from rear.models.base import available_architectures  # noqa: E402


def _run_dataset(args):
    adapter = load_adapter(args.dataset, split=args.split)
    splits = adapter.sequence_splits()
    if args.split not in splits:
        raise SystemExit(f"{args.dataset} has no split {args.split!r}; "
                         f"available: {sorted(splits)}")
    ids = splits[args.split]
    if args.limit:
        ids = ids[:args.limit]

    out = Path(args.out or REPO_ROOT / "outputs" / f"gen_{args.dataset}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{adapter}\n  split {args.split}: {len(ids)} frames -> {out}")

    t0, counts = time.time(), []
    for n, i in enumerate(ids):
        fr = adapter.load_frame(i)
        g = generate_for_frame(adapter, fr, seed=args.seed + n,
                               architecture=args.head)
        pts = g.points.copy()
        if args.pcdet_frame:
            pts[:, :3] = adapter.to_pcdet_frame(pts[:, :3], fr.calibration)
        pts.astype(np.float32).tofile(out / f"{fr.frame_id.replace(':', '_')}.bin")
        counts.append(len(pts))
        if n % 100 == 0:
            print(f"  {n}/{len(ids)}  mean {np.mean(counts):.1f} pts/frame")

    print(f"\n{len(ids)} frames in {(time.time()-t0)/60:.1f} min; "
          f"mean {np.mean(counts):.1f} pts/frame "
          f"(measured real in-FOV reference: ~316/frame on VoD test)")


def _run_single_file(args):
    lidar = np.fromfile(args.lidar, dtype=np.float32).reshape(-1, 4) \
        if args.lidar.endswith(".bin") else np.load(args.lidar)
    lidar = np.asarray(lidar, dtype=np.float64)
    ego_velocity = np.array(args.ego_velocity, dtype=np.float64)

    # The physics profile (voxel size, detection FOV, ground/normal estimation)
    # is dataset-fitted, not something a single point cloud carries. Reuse the
    # KITTI-CARLA profile's grid config by default -- it already fully
    # specifies `candidates:` and needs no inherited artifact lookup. Loaded
    # lazily (config only, no adapter._open()) so no real dataset is touched.
    profile = load_adapter_config(args.profile)
    cfg = profile.grid_config()
    if args.cos_angle_source:
        cfg = VoxelGridConfig(**{**cfg.__dict__, "cos_angle_source": args.cos_angle_source})

    # MATERIAL. The shipped heads read a 10-column electromagnetic block built
    # from per-point semantics, so a single sweep needs labels to get the full
    # model. Without them every candidate receives the all-zero "no material
    # information" row -- a valid input the head has seen, but one that throws
    # away the block's whole contribution, so it has to be asked for explicitly.
    labels = None
    if args.labels:
        labels = (np.fromfile(args.labels, dtype=np.int32).astype(np.int64)
                  if args.labels.endswith(".bin") else
                  np.asarray(np.load(args.labels), dtype=np.int64))
        if len(labels) != len(lidar):
            raise SystemExit(f"--labels has {len(labels):,} entries but --lidar "
                             f"has {len(lidar):,} points; they must be aligned "
                             f"one-to-one (use -1 for 'no label').")
    elif not args.no_material:
        raise SystemExit(
            "the fitted heads read a material block built from per-point "
            "semantic labels. Pass --labels <file> --label-space <name> (see "
            "configs/materials/ for the available taxonomies), or pass "
            "--no-material to accept the degraded no-material path knowingly.")

    g = generate_radar_frame(lidar, ego_velocity, seed=args.seed, config=cfg,
                             architecture=args.head or "prism",
                             point_labels=labels,
                             label_space=args.label_space,
                             allow_missing_material=args.no_material)
    out = Path(args.out or "radar_out.bin")
    g.points.astype(np.float32).tofile(out)
    print(f"input LiDAR:  {lidar.shape[0]:>6d} points"
          + (f"  ({(labels >= 0).sum():,} labelled)" if labels is not None else ""))
    print(f"output radar: {g.points.shape[0]:>6d} points "
          f"({(~g.is_clutter).sum()} surface, {g.is_clutter.sum()} clutter)")
    print(f"written -> {out}  (float32, shape (M, 5): x,y,z,RCS,v_r)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="registered dataset name, e.g. kitti_carla")
    ap.add_argument("--split", default="Town01", help="dataset split/town (dataset mode)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pcdet-frame", action="store_true",
                    help="write points in the dataset's detector frame instead "
                         "of the generator's LiDAR frame")

    ap.add_argument("--lidar", help="path to a single LiDAR sweep (.bin float32 "
                                    "Nx4, or .npy) for single-file mode")
    ap.add_argument("--ego-velocity", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    metavar=("VX", "VY", "VZ"),
                    help="ego velocity, m/s, LiDAR-frame axes (single-file mode)")
    ap.add_argument("--profile", default="kitti_carla", choices=["kitti_carla", "vod"],
                    help="which fitted physics profile to apply (single-file mode)")
    ap.add_argument("--cos-angle-source", choices=["column3", "estimate_normals"],
                    default=None, help="override how incidence angle is obtained; "
                                       "'column3' if your 4th LiDAR column is already "
                                       "a surface cosine, else 'estimate_normals'")
    ap.add_argument("--labels", default=None,
                    help="per-point semantic ids aligned to --lidar (.npy or "
                         "int32 .bin; -1 = no label). Required in single-file "
                         "mode unless --no-material is given.")
    ap.add_argument("--label-space", default="carla",
                    choices=available_label_spaces(),
                    help="which taxonomy --labels are in (configs/materials/)")
    ap.add_argument("--no-material", action="store_true",
                    help="generate without material information. The head still "
                         "runs -- every candidate gets the all-zero block it "
                         "already sees for unobserved voxels -- but the material "
                         "branch contributes nothing. Deliberate, not a default.")

    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default=None, choices=available_architectures(),
                    help="which learned head to use; default comes from the "
                         "dataset config's fitted.heads.implementation")
    args = ap.parse_args()

    if args.lidar:
        _run_single_file(args)
    elif args.dataset:
        _run_dataset(args)
    else:
        ap.error("pass either --dataset (batch mode) or --lidar (single-file mode)")


if __name__ == "__main__":
    main()
