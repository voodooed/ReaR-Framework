#!/usr/bin/env python
"""Generate KITTI-CARLA radar + instance-derived boxes for downstream training.

    python downstream/build_kc_cache.py --head mlp   --n 1500
    python downstream/build_kc_cache.py --head prism --n 1500

Produces one .npz per head holding a ragged (points, boxes) pair per frame --
the augmentation pool the downstream experiment draws its "simulated" half
from. Everything about the radar itself comes from `generate_for_frame`, so
the ONLY difference between the two files is the learned head.

WHY KITTI-CARLA AND NOT GENERATED-FROM-VoD. The claim under test is the
sim2real one: you have simulated LiDAR and no real radar, so you synthesise
radar and use it to stretch a small real set. Generating from VoD LiDAR would
leak the real dataset's own scenes into the "simulated" half and answer a
different, easier question.

BOX HANDLING follows the source project's convention exactly
(legacy/scripts/build_detection_datasets.py):
  * the ego vehicle's own box is dropped (it sits at the origin),
  * boxes fully outside the radar FOV are dropped,
  * boxes STRADDLING the FOV edge are kept but marked IGNORE rather than
    deleted -- a half-visible object is neither a valid positive nor a safe
    negative, and deleting it would teach the detector that visible structure
    has no label.

LABEL-SPACE NOTE, carried because it shapes the result: CARLA has no bicycle
tag, so KITTI-CARLA supplies Car and Pedestrian only. Cyclist therefore gets
ZERO simulated supervision and acts as a built-in zero-supervision control --
any Cyclist movement in the downstream table is regularisation or noise, not
transfer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from rear.data._kc_boxes import boxes_from_instances  # noqa: E402
from rear.data.base import load_adapter  # noqa: E402
from rear.generate.generate_radar_frame import generate_for_frame  # noqa: E402
from rear.models.base import available_architectures  # noqa: E402

# Same conventions as the source project's builder.
CLASS_ID = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
IGNORE_ID = -1
EGO_EXCLUSION_M = 2.0


def box_fov_state(x, y, length, width, rot, fov_deg):
    """'in' / 'edge' / 'out' for a box against the radar wedge, judged on its
    four BEV corners rather than its centre -- a long box centred just inside
    the wedge can still hang most of its body outside it."""
    c, s = np.cos(rot), np.sin(rot)
    hl, hw = length / 2.0, width / 2.0
    cx = np.array([hl, hl, -hl, -hl])
    cy = np.array([hw, -hw, -hw, hw])
    px = x + cx * c - cy * s
    py = y + cx * s + cy * c
    inside = np.abs(np.degrees(np.arctan2(py, px))) <= fov_deg
    if inside.all():
        return "in"
    return "edge" if inside.any() else "out"


def kc_boxes_array(boxes, fov_deg):
    out = []
    for b in boxes:
        if np.hypot(b.x, b.y) < EGO_EXCLUSION_M:      # the ego vehicle itself
            continue
        state = box_fov_state(b.x, b.y, b.length, b.width, b.rotation_z, fov_deg)
        if state == "out":
            continue
        cid = IGNORE_ID if state == "edge" else CLASS_ID.get(b.class_name, IGNORE_ID)
        out.append([b.x, b.y, b.z, b.length, b.width, b.height, b.rotation_z, cid])
    return np.array(out, dtype=np.float32).reshape(-1, 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, choices=available_architectures())
    ap.add_argument("--n", type=int, default=1500, help="frames to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "downstream" / "cache"))
    args = ap.parse_args()

    adapter = load_adapter("kitti_carla")
    fov = adapter.radar_fov_degrees
    splits = adapter.sequence_splits()
    towns = sorted(splits)
    total = sum(len(splits[t]) for t in towns)
    # Spread the draw across both towns in proportion to their length, evenly
    # within each -- consecutive frames are ~0.1 s apart and near-duplicates,
    # so a contiguous block would be far less diverse than its count suggests.
    per = {t: int(round(args.n * len(splits[t]) / total)) for t in towns}

    print("=" * 74)
    print(f"KITTI-CARLA generation for downstream -- head={args.head}")
    print("=" * 74)
    print(f"  frames requested : {args.n}  {per}")
    print(f"  FOV              : +-{fov:.0f} deg")

    pts, box, meta = [], [], []
    t0, done = time.time(), 0
    for town in towns:
        ids = splits[town]
        for i in np.linspace(0, len(ids) - 1, per[town]).astype(int):
            fid = ids[int(i)]
            frame = adapter.load_frame(fid)
            g = generate_for_frame(adapter, frame, seed=args.seed + int(i),
                                   architecture=args.head)
            b = kc_boxes_array(adapter.boxes_for(fid), fov)
            pts.append(g.points.astype(np.float32))
            box.append(b)
            meta.append((str(fid), int(len(g.points)), int(len(b))))
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"    {done}/{args.n}  ({el/done:.2f}s/frame, "
                      f"eta {(args.n-done)*el/done/60:.1f} min)", flush=True)

    P = np.concatenate(pts) if pts else np.zeros((0, 5), np.float32)
    B = np.concatenate(box) if box else np.zeros((0, 8), np.float32)
    pptr = np.cumsum([0] + [len(p) for p in pts])
    bptr = np.cumsum([0] + [len(b) for b in box])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"kc_{args.head}.npz"
    np.savez_compressed(f, points=P, points_ptr=pptr, boxes=B, boxes_ptr=bptr,
                        frame_ids=np.array([m[0] for m in meta]))
    ev = B[B[:, 7] > 0] if len(B) else B
    print(f"\n  {len(pts)} frames, {len(P):,} points ({len(P)/max(len(pts),1):.0f}/frame)")
    print(f"  {len(B)} boxes, {len(ev)} evaluated ({len(B)-len(ev)} IGNORE)")
    for c, i in CLASS_ID.items():
        print(f"      {c:<12}{int((ev[:, 7] == i).sum()) if len(ev) else 0:>7,}")
    print(f"  wrote {f}")


if __name__ == "__main__":
    main()
