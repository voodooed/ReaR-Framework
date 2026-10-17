#!/usr/bin/env python
"""Train and score the data-scarcity augmentation grid.

    python downstream/run_grid.py --arch pge --seeds 0 1

SIX ARMS x N SEEDS, all evaluated on the SAME 413 real VoD test frames:

    A_full_real     2000 real                     upper reference
    B_limited_real  1000 real                     the scarce condition
    C_mix50_<head>  1000 real + 1000 generated    does sim ADD to scarce real?
    D_mix40_<head>   800 real + 1200 generated    does MORE sim keep helping?

THE COMPARISON THAT ANSWERS THE QUESTION IS C - B. Both train on the same
1000 real frames; C additionally sees generated radar. A is the reference for
how much real data that augmentation is worth.

TWO RULES INHERITED FROM THE SOURCE PROJECT, BOTH LOAD-BEARING
---------------------------------------------------------------
1. SCORING IS `legacy/scripts/score_pcdet.py`, NEVER OpenPCDet's KITTI
   evaluator. That evaluator is numerically broken in this checkout -- its
   rotated-box IoU returns 0.0 for a box against itself -- and it fails
   silently, printing a well-formatted table of zeros that looks like a real
   negative result.
2. EVERY ARM IS SCORED ON THE FULL TEST SPLIT, the same frames regardless of
   what it trained on. This is asserted before any table is written.

A DIFFERENCE SMALLER THAN THE SEED SPREAD IS NOT A RESULT. The floor is
measured here, not assumed: with 2 seeds per arm it is
max_arm |metric(seed0) - metric(seed1)|, the same estimator the E-series used.
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REAR = Path(__file__).resolve().parents[1]
# EXTERNAL DEPENDENCY, DECLARED RATHER THAN ASSUMED. The downstream detector
# experiments need an OpenPCDet checkout, its environment, its pcdet data trees
# and the project's own scorer -- none of which belong inside a radar-generation
# package. The generator itself has no such dependency; only this directory
# does. Point REAR_DOWNSTREAM_ROOT at your checkout to move it.
PHYSRADAR = Path(os.environ.get("REAR_DOWNSTREAM_ROOT",
                                "/DATA1/Vivek/Sim2Real_RADAR/Code/physradar"))
PCDET = PHYSRADAR / "third_party" / "OpenPCDet"
PY = PHYSRADAR / ".venv_pcdet" / "bin" / "python"
TREES = PHYSRADAR / "data" / "pcdet"
SCORER = PHYSRADAR / "legacy" / "scripts" / "score_pcdet.py"

ARMS = ["A_full_real", "B_limited_real",
        "C_mix50_mlp", "D_mix40_mlp",
        "C_mix50_prism", "D_mix40_prism"]

# Which detector config each architecture uses. `pp` is the stock pillar
# baseline; `pge` and `seg` swap in the reimplemented radar-specific modules.
CFG = {"pp": "pointpillar_vod_radar.yaml",
       "pge": "pointpillar_vod_radar_pge.yaml",
       "seg": "pointpillar_vod_radar_seg.yaml"}


def check_eval_sets_match(arms):
    """Refuse to build a table whose arms were scored on different frames."""
    sets = {a: (TREES / a / "ImageSets" / "val.txt").read_text().split() for a in arms}
    if len({tuple(v) for v in sets.values()}) > 1:
        raise SystemExit("eval sets differ across arms; their APs are not "
                         f"comparable: { {k: len(v) for k, v in sets.items()} }")
    return len(next(iter(sets.values())))


def output_dir(arch, tag):
    """Where OpenPCDet writes this run's checkpoints and eval results."""
    return PCDET / "output" / "kitti_models" / CFG[arch].replace(".yaml", "") / tag


def train_one(arch, arm, seed, epochs, batch, workers, out_root):
    tag = f"{arch}_{arm}_s{seed}"
    # WIPE THE OUTPUT DIRECTORY FIRST. OpenPCDet RESUMES from the newest
    # checkpoint it finds under this tag, and if that checkpoint is already at
    # the target epoch it skips training entirely and goes straight to
    # evaluation -- producing a FRESH result.pkl from a STALE model. That is
    # not a hypothetical: a rerun of this grid silently re-scored the previous
    # experiment's September-1 checkpoints and wrote 28 result files that
    # looked completely normal. Runs finishing a minute apart was the only
    # visible symptom. A tag is one experiment; reusing its directory is never
    # what is wanted here.
    d = output_dir(arch, tag)
    if d.exists():
        shutil.rmtree(d)

    # RELATIVE, and the command runs with cwd=PCDET/tools. OpenPCDet derives
    # its output directory from this string; an absolute path makes it nest the
    # entire filesystem path under output/ and the results become unfindable.
    cfg = f"cfgs/kitti_models/{CFG[arch]}"
    # DATA_PATH is overridden per arm so one config serves every tree.
    cmd = [str(PY), "train.py", "--cfg_file", str(cfg),
           "--epochs", str(epochs), "--batch_size", str(batch),
           "--workers", str(workers), "--extra_tag", tag,
           "--fix_random_seed", "--seed", str(seed),
           "--set", "DATA_CONFIG.DATA_PATH", str(TREES / arm)]
    log = out_root / f"{tag}.log"
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=PCDET / "tools", stdout=f, stderr=subprocess.STDOUT)
    return tag, r.returncode, (time.time() - t0) / 60, log, t0


def score_one(arch, arm, seed, out_root, started_at=None):
    """Score this run's predictions, refusing anything older than the run.

    Training is allowed to exit non-zero -- this OpenPCDet/torch combination
    segfaults during process teardown AFTER evaluation has written its results,
    and throwing away a completed run for that would be wrong. But "ignore the
    exit code" and "score whatever file is present" together are how a stale
    prediction gets reported as a fresh one, so the file must be newer than the
    run that supposedly produced it.
    """
    tag = f"{arch}_{arm}_s{seed}"
    d = output_dir(arch, tag) / "eval"
    preds = sorted(d.rglob("result.pkl"))
    if not preds:
        return None
    newest = preds[-1]
    if started_at is not None and newest.stat().st_mtime < started_at:
        raise SystemExit(
            f"{tag}: the newest predictions ({newest}) predate this run. "
            f"Refusing to score them -- a result file older than the run that "
            f"was supposed to produce it is a stale artifact, not a result.")
    js = out_root / f"{tag}.json"
    cmd = [str(PY), str(SCORER), "--pred", str(preds[-1]),
           "--infos", str(TREES / arm / "kitti_infos_val.pkl"), "--json", str(js)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(js.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="pge", choices=list(CFG))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REAR / "downstream" / "runs"))
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n_eval = check_eval_sets_match(args.arms)
    print("=" * 74)
    print(f"DOWNSTREAM GRID -- arch={args.arch}")
    print("=" * 74)
    print(f"  arms   : {len(args.arms)} x {len(args.seeds)} seeds = "
          f"{len(args.arms)*len(args.seeds)} runs")
    print(f"  eval   : {n_eval} real VoD test frames, identical in every arm")
    print(f"  epochs : {args.epochs}\n")

    results = {}
    for arm in args.arms:
        for seed in args.seeds:
            tag = f"{args.arch}_{arm}_s{seed}"
            started_at = None
            if not args.score_only:
                print(f"  [train] {tag} ...", flush=True)
                _, rc, mins, log, started_at = train_one(
                    args.arch, arm, seed, args.epochs, args.batch,
                    args.workers, out)
                # rc=-11 is tolerated: this checkout segfaults during process
                # teardown, AFTER result.pkl is written, so the predictions are
                # intact. What makes that safe is the freshness check in
                # score_one -- without it, "ignore the exit code" would happily
                # report a stale prediction file as a result.
                if rc != 0:
                    print(f"    train exited rc={rc} after {mins:.1f} min "
                          f"(see {log}); scoring only if the predictions are "
                          f"newer than this run", flush=True)
                else:
                    print(f"    done in {mins:.1f} min", flush=True)
            r = score_one(args.arch, arm, seed, out, started_at=started_at)
            if r is None:
                print(f"    no predictions for {tag}", flush=True)
                continue
            results[tag] = r
            def _ap(k):
                v = r.get(k)
                return v["AP"] if isinstance(v, dict) else (v if v is not None else float("nan"))
            print(f"    {tag}: mAP {r.get('mAP', float('nan')):.4f}  "
                  f"Car {_ap('Car'):.4f}  Ped {_ap('Pedestrian'):.4f}  "
                  f"Cyc {_ap('Cyclist'):.4f}", flush=True)
            (out / f"grid_{args.arch}.json").write_text(json.dumps(results, indent=1))

    (out / f"grid_{args.arch}.json").write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {out / f'grid_{args.arch}.json'}")


if __name__ == "__main__":
    main()
