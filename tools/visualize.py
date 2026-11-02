#!/usr/bin/env python
"""BEV (bird's-eye-view) visualization: LiDAR objects vs. generated radar.

Renders one frame as a pair of top-down (x, y) panels sharing the same axes:

  left  -- input LiDAR point cloud, height-coloured, with object boxes
  right -- generated radar point cloud: SURFACE points (from real geometry,
           coloured by RCS) vs CLUTTER points (grey), with the SAME object
           boxes overlaid so you can see where each object's radar returns
           land relative to where the object actually is.

Usage:
    python tools/visualize.py --dataset kitti_carla --split Town01 --frame 5 \\
        --out viz_frame5.png

    # random frame, default output name
    python tools/visualize.py --split Town02 --frame 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib          # noqa: E402
matplotlib.use("Agg")      # headless-safe; script always saves to a file
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.patches import Polygon    # noqa: E402

from rear.data.base import load_adapter          # noqa: E402
from rear.generate.generate_radar_frame import generate_for_frame  # noqa: E402


def _box_corners_xy(b):
    """4 BEV corners of an oriented box, in draw order."""
    hl, hw = b.length / 2.0, b.width / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    c, s = np.cos(b.rotation_z), np.sin(b.rotation_z)
    R = np.array([[c, -s], [s, c]])
    return (local @ R.T) + np.array([b.x, b.y])


def _draw_boxes(ax, boxes, color="red"):
    for b in boxes:
        poly = Polygon(_box_corners_xy(b), closed=True, fill=False,
                       edgecolor=color, linewidth=1.3, zorder=5)
        poly.set_clip_on(True)
        ax.add_patch(poly)
        ax.text(b.x, b.y + b.width / 2.0 + 0.4, b.class_name, color=color,
                fontsize=7, ha="center", zorder=6, clip_on=True)


def _draw_fov(ax, origin, fov_deg, max_range, color="0.6"):
    """The wedge the generator actually masks to.

    `radar_fov_degrees` is a HALF-angle: rear.geometry.candidates.
    within_radar_fov() keeps candidates with |azimuth| <= radar_fov_deg, so
    the configured 60.0 means +-60 deg (a 120 deg sector), NOT a 60 deg
    sector. Drawing +-fov_deg/2 here put the dashed lines at half the true
    width and made ~23% of correctly-generated points look like they had
    escaped the FOV.
    """
    half = np.radians(fov_deg)
    for a in (-half, half):
        ax.plot([origin[0], origin[0] + max_range * np.cos(a)],
                [origin[1], origin[1] + max_range * np.sin(a)],
                color=color, lw=0.8, ls="--", zorder=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="kitti_carla")
    ap.add_argument("--split", default="Town01")
    ap.add_argument("--frame", type=int, default=0, help="frame index within the split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--range", type=float, default=60.0, help="BEV axis limit, metres")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    adapter = load_adapter(args.dataset, split=args.split)
    splits = adapter.sequence_splits()
    ids = splits[args.split]
    frame_id = ids[args.frame]
    frame = adapter.load_frame(frame_id)

    boxes = adapter.boxes_for(frame_id) if hasattr(adapter, "boxes_for") else frame.boxes
    gen = generate_for_frame(adapter, frame, seed=args.seed)
    pts, is_clutter = gen.points, gen.is_clutter

    R = args.range
    origin = adapter.radar_mount_offset[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)

    # -- left: input LiDAR, height-coloured --------------------------------
    lidar = frame.lidar
    sc1 = ax1.scatter(lidar[:, 0], lidar[:, 1], c=lidar[:, 2], cmap="viridis",
                      s=0.6, alpha=0.5, vmin=-2, vmax=3)
    _draw_boxes(ax1, boxes)
    _draw_fov(ax1, origin, adapter.radar_fov_degrees, R)
    ax1.set_title(f"input LiDAR  ({len(lidar)} pts, {len(boxes)} objects)")
    fig.colorbar(sc1, ax=ax1, shrink=0.7, label="z (m)")

    # -- right: generated radar, RCS-coloured (clutter in grey) ------------
    surf, clut = pts[~is_clutter], pts[is_clutter]
    if len(clut):
        ax2.scatter(clut[:, 0], clut[:, 1], c="0.75", s=10, marker="x",
                   label=f"clutter ({len(clut)})", zorder=2)
    if len(surf):
        sc2 = ax2.scatter(surf[:, 0], surf[:, 1], c=surf[:, 3], cmap="inferno",
                         s=16, edgecolors="k", linewidths=0.2,
                         label=f"surface ({len(surf)})", zorder=3)
        fig.colorbar(sc2, ax=ax2, shrink=0.7, label="RCS (dBsm)")
    _draw_boxes(ax2, boxes)
    _draw_fov(ax2, origin, adapter.radar_fov_degrees, R)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(f"generated radar  ({len(pts)} pts)")

    for ax in (ax1, ax2):
        ax.set_xlim(origin[0] - 5, origin[0] + R)
        ax.set_ylim(-R, R)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, forward)")
        ax.plot(*origin, marker="^", color="blue", markersize=8, zorder=6)
    ax1.set_ylabel("y (m, left)")

    fig.suptitle(f"{adapter.name} / {frame_id}  (seed={args.seed})")
    fig.tight_layout()

    out = Path(args.out or f"viz_{args.dataset}_{str(frame_id).replace(':', '_')}.png")
    fig.savefig(out, dpi=150)
    print(f"input LiDAR: {len(lidar)} pts, {len(boxes)} objects")
    print(f"generated radar: {len(surf)} surface + {len(clut)} clutter = {len(pts)} pts")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
