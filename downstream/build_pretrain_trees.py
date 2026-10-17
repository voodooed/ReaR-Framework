#!/usr/bin/env python
"""Build generated-ONLY trees for the pretrain-then-finetune experiment.

    python downstream/build_pretrain_trees.py

WHY THIS IS A DIFFERENT HYPOTHESIS FROM MIXING
-----------------------------------------------
The mixing arms (C, D) put real and generated frames in the same training set,
so every gradient step is contested: a simulated frame occupies a slot a real
frame could have had, and the network must fit both distributions at once.
That is the setting in which the generated radar measurably HURT.

Pretrain-then-finetune separates them in time. The generated frames only set
the initial weights; the real frames then get the entire fine-tuning budget to
themselves and are never displaced. If the generated data carries any usable
structure at all -- even structure that is wrong in detail -- this is the
setting where it can show up, because nothing is being traded away for it.

THE CONFOUND, AND THE CONTROL FOR IT
-------------------------------------
Pretraining hands the model extra optimisation steps that the from-scratch
baseline never gets. A pretrained model that beats a 20-epoch baseline may
simply have trained longer. So the comparison is against BOTH:

    B_limited_real  (20 epochs)              the standing scarce baseline
    B_long          (40 epochs, same data)   same total steps, no generated data

Pretraining only counts as working if it beats B_long, not merely B. The
source project flagged exactly this confound in its own E3 arm ("E3 also
receives 2x the gradient steps at fixed epochs"), and it is cheap to control
for, so it is controlled for here.

The generated frames used are EXACTLY the ones arm C mixes in, so the two
experiments differ only in how the data is scheduled, not in what it contains.
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

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_trees import build_infos  # noqa: E402

TREES = PHYSRADAR / "data" / "pcdet"
REAL_TREE = TREES / "E1_real"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", nargs="+", default=["mlp", "prism"])
    args = ap.parse_args()

    test_ids = (REAL_TREE / "ImageSets" / "val.txt").read_text().split()
    print("=" * 74)
    print("PRETRAIN TREES -- generated only")
    print("=" * 74)

    for head in args.heads:
        src = TREES / f"C_mix50_{head}"
        dst = TREES / f"P_gen_{head}"
        if not src.exists():
            raise SystemExit(f"{src} missing -- run build_trees.py first")
        if dst.exists():
            shutil.rmtree(dst)
        for sub in ("velodyne", "label_2", "calib", "image_2"):
            (dst / "training" / sub).mkdir(parents=True, exist_ok=True)
        (dst / "ImageSets").mkdir(parents=True, exist_ok=True)
        t = dst / "testing"
        if t.is_symlink() or t.exists():
            t.unlink() if t.is_symlink() else shutil.rmtree(t)
        t.symlink_to(REAL_TREE / "testing")

        # The GENERATED half of arm C, verbatim -- same frames, same radar,
        # same labels. Only the schedule differs between the two experiments.
        gen_ids = [i for i in (src / "ImageSets" / "train.txt").read_text().split()
                   if int(i) >= 60000]
        for fid in gen_ids + test_ids:
            for sub, ext in (("velodyne", ".bin"), ("label_2", ".txt"),
                             ("calib", ".txt"), ("image_2", ".jpg")):
                s = src / "training" / sub / f"{fid}{ext}"
                if not s.exists():                       # eval frames live in C too
                    s = REAL_TREE / "training" / sub / f"{fid}{ext}"
                if s.exists():
                    d = dst / "training" / sub / f"{fid}{ext}"
                    if d.is_symlink() or d.exists():
                        d.unlink()
                    d.symlink_to(s.resolve())

        (dst / "ImageSets" / "train.txt").write_text("\n".join(gen_ids) + "\n")
        for f in ("val.txt", "test.txt"):
            (dst / "ImageSets" / f).write_text("\n".join(test_ids) + "\n")
        print(f"  P_gen_{head}: {len(gen_ids)} generated train frames "
              f"(the generated half of C_mix50_{head}), {len(test_ids)} eval")
        # Same rule as build_trees.py: a tree without infos looks complete and
        # trains on nothing.
        n_tr, n_va = build_infos(dst)
        print(f"  P_gen_{head}: infos built ({n_tr} train, {n_va} val samples)")

    print(f"\n  trees under {TREES}")
    print("  next: pretrain -> finetune (downstream/run_pretrain.py)")


if __name__ == "__main__":
    main()
