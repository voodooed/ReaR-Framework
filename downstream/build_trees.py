#!/usr/bin/env python
"""Build the pcdet trees for the data-scarcity augmentation experiment.

    python downstream/build_trees.py --total 2000 --seed 0

SIX ARMS, one fixed total training size T, so "more data" is separated from
"better data":

    A  full_real        T   real                     upper reference
    B  limited_real    T/2  real                     the scarce condition
    C  mix50_<head>    T/2  real + T/2 generated     does sim ADD to scarce real?
    D  mix40_<head>    0.4T real + 0.6T generated    does MORE sim keep helping?

C and D are built once per generated head (mlp, prism), so the same real half
is paired with each -- any difference between them is the head and nothing
else. Real frames are drawn from the SAME shuffled order for every arm, so B's
1000 frames are a strict subset of A's 2000 and C/D share B's real half
exactly. Without that, "limited real" and "limited real + sim" would differ in
which real frames they saw as well as in the augmentation.

EVALUATION IS IDENTICAL IN EVERY ARM: the full 413-frame real VoD test split,
never a subset, never anything generated. `run_experiment.check_evaluation_is_
comparable` re-asserts this before any table is produced.

COORDINATE FRAMES. KITTI labels live in RECTIFIED CAMERA coordinates and pcdet
converts them back with `boxes3d_kitti_camera_to_lidar`. Rather than hand-roll
that transform -- the historical E3 tree got exactly this wrong, writing its
generated half in the LiDAR frame against radar-frame labels, a ~2.5 m error on
half the training data that was only caught much later -- this script writes
labels with pcdet's OWN `boxes3d_lidar_to_kitti_camera` and then asserts the
round-trip recovers the input boxes. A mismatch aborts the build.

KITTI-CARLA's virtual radar sits at the LiDAR origin and its instance-derived
boxes share that frame, so no radar<->lidar transform is involved on the
generated side at all; the only conversion is lidar->camera for the label file.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REAR = Path(__file__).resolve().parents[1]
# EXTERNAL DEPENDENCY, DECLARED RATHER THAN ASSUMED. The downstream detector
# experiments need an OpenPCDet checkout, its environment, its pcdet data trees
# and the project's own scorer -- none of which belong inside a radar-generation
# package. The generator itself has no such dependency; only this directory
# does. Point REAR_DOWNSTREAM_ROOT at your checkout to move it.
PHYSRADAR = Path(os.environ.get("REAR_DOWNSTREAM_ROOT",
                                "/DATA1/Vivek/Sim2Real_RADAR/Code/physradar"))
sys.path.insert(0, str(REAR))
sys.path.insert(0, str(PHYSRADAR / "third_party" / "OpenPCDet"))
sys.path.insert(0, str(PHYSRADAR / "legacy" / "src"))

import numpy as np  # noqa: E402

from pcdet.utils import box_utils  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402

# OpenPCDet's dataset config for this radar setup. Needed to BUILD the info
# files, below -- its own `create_kitti_infos` CLI hard-codes `data/kitti` and
# ignores DATA_PATH, so it is called programmatically instead.
DATASET_CFG = (PHYSRADAR / "third_party" / "OpenPCDet" / "tools" / "cfgs"
               / "dataset_configs" / "vod_radar_dataset.yaml")
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

REAL_TREE = PHYSRADAR / "data" / "pcdet" / "E1_real"
OUT_ROOT = PHYSRADAR / "data" / "pcdet"
GEN_ID_OFFSET = 60000          # generated ids cannot collide with real ones
RADAR_N_COLS = 7               # x, y, z, rcs, v_r, v_r_comp, time
CLASS_NAME = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}


def build_infos(tree: Path, workers: int = 4):
    """Generate the four `kitti_infos_*.pkl` OpenPCDet builds its sample list
    from. WITHOUT THESE A TREE LOOKS COMPLETE AND TRAINS ON NOTHING: the
    dataloader reports "Total samples: 0" and the run dies on an empty sampler,
    a confusing failure a long way from its cause. So this is a step of tree
    construction, not a separate thing to remember.

    The four info files are built directly rather than through OpenPCDet's
    `create_kitti_infos`, for two reasons. Its CLI hard-codes `data/kitti` and
    ignores DATA_PATH. And its last step, `create_groundtruth_database`, raises
    a NameError in this checkout -- irrelevant here, because the GT database is
    only consumed by `gt_sampling` augmentation and this radar config does not
    use it (checked below rather than assumed, since silently skipping an
    augmentation the config asked for would change every arm at once).
    """
    import pickle

    from easydict import EasyDict
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    import yaml as _yaml

    cfg = EasyDict(_yaml.safe_load(DATASET_CFG.read_text()))
    cfg.DATA_PATH = str(tree)
    aug = [a.get("NAME") for a in (cfg.get("DATA_AUGMENTOR", {}) or {})
           .get("AUG_CONFIG_LIST", [])]
    if "gt_sampling" in aug:
        raise SystemExit(
            f"{DATASET_CFG.name} enables gt_sampling, which needs the GT "
            f"database this builder skips. Build it separately, or the arms "
            f"will silently train without an augmentation the config asked for.")

    dataset = KittiDataset(dataset_cfg=cfg, class_names=CLASS_NAMES,
                           root_path=tree, training=False)
    infos = {}
    for split, has_label in (("train", True), ("val", True), ("test", False)):
        dataset.set_split(split)
        infos[split] = dataset.get_infos(num_workers=workers, has_label=has_label,
                                         count_inside_pts=has_label)
        with open(tree / f"kitti_infos_{split}.pkl", "wb") as f:
            pickle.dump(infos[split], f)
    with open(tree / "kitti_infos_trainval.pkl", "wb") as f:
        pickle.dump(infos["train"] + infos["val"], f)

    if not infos["train"]:
        raise SystemExit(f"{tree.name}: info build produced 0 train samples")
    return len(infos["train"]), len(infos["val"])


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def write_bin(path: Path, pts5):
    """pts5 is (N,5) x,y,z,RCS,v_r as the generator emits.

    v_r_comp and time are written as ZERO and the VFE is configured to ignore
    them, so real and generated inputs are format-identical and the comparison
    cannot be won on channels the generator structurally cannot produce."""
    a = np.zeros((len(pts5), RADAR_N_COLS), np.float32)
    a[:, :5] = pts5
    path.parent.mkdir(parents=True, exist_ok=True)
    a.tofile(path)


def write_label(path: Path, boxes8, calib):
    """boxes8: (N,8) x,y,z,l,w,h,ry,class_id in the LiDAR frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(boxes8) == 0:
        path.write_text("")
        return np.zeros((0, 7)), boxes8
    # KITTI's info builder computes gt_boxes_lidar for the NON-DontCare
    # objects only, then filters every other annotation array by indices taken
    # from the full list. That is only consistent if all DontCare entries sit
    # at the END of the file, which is the KITTI convention. Interleaved
    # DontCare produces an out-of-bounds index deep inside the dataloader.
    order = np.argsort(boxes8[:, 7] <= 0, kind="stable")   # evaluated first
    boxes8 = boxes8[order]
    lidar7 = boxes8[:, :7].astype(np.float64)
    cam = box_utils.boxes3d_lidar_to_kitti_camera(lidar7.copy(), calib)
    lines = []
    for k, b in enumerate(boxes8):
        cid = int(b[7])
        name = CLASS_NAME.get(cid, "DontCare")
        x, y, z, l, h, w, ry = cam[k]
        # A 2-D bbox is required by the format. Radar-only detection never
        # reads it, and pcdet's own evaluator is not used here (see
        # run_experiment.py), so a fixed placeholder is honest and inert --
        # writing a fabricated projection would look like real image evidence.
        lines.append(f"{name} 0 0 0 0.00 0.00 50.00 50.00 "
                     f"{h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.4f}")
    path.write_text("\n".join(lines) + "\n")
    # The SORTED boxes are returned alongside, so the caller's round-trip check
    # compares `cam` against the boxes that actually produced it rather than
    # against the pre-sort order.
    return cam, boxes8


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=2000, help="T, frames in arm A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heads", nargs="+", default=["mlp", "prism"])
    ap.add_argument("--cache", default=str(REAR / "downstream" / "cache"))
    ap.add_argument("--workers", type=int, default=4,
                    help="parallelism for the OpenPCDet info build")
    args = ap.parse_args()

    T = args.total
    rng = np.random.default_rng(args.seed)

    # ---- real pool, one shuffled order shared by every arm ----------------
    real_train = [s.strip() for s in
                  (REAL_TREE / "ImageSets" / "train.txt").read_text().split()]
    test_ids = [s.strip() for s in
                (REAL_TREE / "ImageSets" / "val.txt").read_text().split()]
    order = rng.permutation(len(real_train))
    real_order = [real_train[i] for i in order]
    if T > len(real_order):
        raise SystemExit(f"--total {T} exceeds the {len(real_order)} real train frames")

    calib = Calibration(REAL_TREE / "training" / "calib" / f"{test_ids[0]}.txt")

    n_half, n_40, n_60 = T // 2, int(round(0.4 * T)), int(round(0.6 * T))
    print("=" * 74)
    print("DOWNSTREAM TREES -- data-scarcity augmentation")
    print("=" * 74)
    print(f"  T                : {T}")
    print(f"  real pool        : {len(real_order)} train frames (shared order, seed {args.seed})")
    print(f"  eval (all arms)  : {len(test_ids)} REAL VoD test frames")
    print(f"  A full_real      : {T} real")
    print(f"  B limited_real   : {n_half} real")
    print(f"  C mix50_<head>   : {n_half} real + {n_half} gen")
    print(f"  D mix40_<head>   : {n_40} real + {n_60} gen\n")

    def new_tree(name):
        d = OUT_ROOT / name
        if d.exists():
            shutil.rmtree(d)
        for sub in ("velodyne", "label_2", "calib", "image_2"):
            (d / "training" / sub).mkdir(parents=True, exist_ok=True)
        (d / "ImageSets").mkdir(parents=True, exist_ok=True)
        t = d / "testing"
        if t.is_symlink() or t.exists():
            t.unlink() if t.is_symlink() else shutil.rmtree(t)
        t.symlink_to(REAL_TREE / "testing")
        return d

    def add_real(d, ids):
        for fid in ids:
            for sub, ext in (("velodyne", ".bin"), ("label_2", ".txt"),
                             ("calib", ".txt"), ("image_2", ".jpg")):
                s = REAL_TREE / "training" / sub / f"{fid}{ext}"
                if s.exists():
                    link(s, d / "training" / sub / f"{fid}{ext}")

    def add_generated(d, cache, n, tag):
        z = np.load(cache)
        P, pptr = z["points"], z["points_ptr"]
        B, bptr = z["boxes"], z["boxes_ptr"]
        # Only frames carrying at least one EVALUATED box are usable: this
        # OpenPCDet checkout's info builder concatenates the per-object arrays
        # unguarded and dies on an empty label file. Frames with no in-FOV
        # object are dropped rather than given a fabricated DontCare entry.
        # The box set is identical across heads (only the radar differs), so
        # this selects the SAME frames for mlp and prism and the arms stay
        # paired.
        usable = np.array([i for i in range(len(pptr) - 1)
                           if len(B[bptr[i]:bptr[i + 1]])
                           and (B[bptr[i]:bptr[i + 1]][:, 7] > 0).any()])
        n_avail = len(usable)
        if n > n_avail:
            raise SystemExit(f"{tag}: need {n} generated frames, cache has "
                             f"{n_avail} with at least one evaluated box")
        pick = usable[np.linspace(0, n_avail - 1, n).astype(int)]
        ids, npts, nbox, worst = [], 0, 0, 0.0
        for j, i in enumerate(pick):
            gid = f"{GEN_ID_OFFSET + j:05d}"
            pts = P[pptr[i]:pptr[i + 1]]
            box = B[bptr[i]:bptr[i + 1]]
            write_bin(d / "training" / "velodyne" / f"{gid}.bin", pts)
            cam, box_sorted = write_label(
                d / "training" / "label_2" / f"{gid}.txt", box, calib)
            # Assert the label round-trips back to the boxes we meant.
            if len(box_sorted):
                back = box_utils.boxes3d_kitti_camera_to_lidar(cam.copy(), calib)
                worst = max(worst, float(np.abs(back[:, :3]
                                                - box_sorted[:, :3].astype(np.float64)).max()))
            for sub, ext in (("calib", ".txt"), ("image_2", ".jpg")):
                s = REAL_TREE / "training" / sub / f"{test_ids[0]}{ext}"
                if s.exists():
                    link(s, d / "training" / sub / f"{gid}{ext}")
            ids.append(gid); npts += len(pts); nbox += len(box)
        if worst > 1e-3:
            raise SystemExit(f"{tag}: label round-trip error {worst:.4f} m -- "
                             f"the generated half would be mis-placed relative "
                             f"to its labels. Refusing to write a tree that "
                             f"repeats the historical E3 frame bug.")
        print(f"    {tag}: {len(ids)} gen frames, {npts:,} pts "
              f"({npts/max(len(ids),1):.0f}/frame), {nbox} boxes, "
              f"round-trip max err {worst:.2e} m")
        return ids

    built = {}
    specs = [("A_full_real", T, None, 0),
             ("B_limited_real", n_half, None, 0)]
    for h in args.heads:
        specs.append((f"C_mix50_{h}", n_half, h, n_half))
        specs.append((f"D_mix40_{h}", n_40, h, n_60))

    for name, n_real, head, n_gen in specs:
        print(f"  building {name} ...")
        d = new_tree(name)
        rids = real_order[:n_real]
        add_real(d, rids)
        gids = []
        if head:
            gids = add_generated(d, Path(args.cache) / f"kc_{head}.npz", n_gen, name)
        add_real(d, test_ids)                       # eval frames live in the tree too
        (d / "ImageSets" / "train.txt").write_text("\n".join(rids + gids) + "\n")
        for f in ("val.txt", "test.txt"):
            (d / "ImageSets" / f).write_text("\n".join(test_ids) + "\n")
        built[name] = d
        print(f"    {name}: {len(rids)} real + {len(gids)} gen = "
              f"{len(rids)+len(gids)} train, {len(test_ids)} eval")
        # Part of building the tree, not a follow-up step -- see build_infos().
        n_tr, n_va = build_infos(d, workers=args.workers)
        print(f"    {name}: infos built ({n_tr} train, {n_va} val samples)")

    # ---- the subset invariant the design depends on -----------------------
    a = set((built['A_full_real'] / "ImageSets" / "train.txt").read_text().split())
    b = set((built['B_limited_real'] / "ImageSets" / "train.txt").read_text().split())
    assert b <= a, "B's real frames must be a subset of A's"
    for h in args.heads:
        c = {x for x in (built[f'C_mix50_{h}'] / "ImageSets" / "train.txt"
                         ).read_text().split() if int(x) < GEN_ID_OFFSET}
        assert c == b, f"C_mix50_{h}'s real half must equal B exactly"
    print("\n  invariants OK: B subset of A; every C shares B's real half exactly")
    print(f"  trees under {OUT_ROOT}")


if __name__ == "__main__":
    main()
