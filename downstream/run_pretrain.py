#!/usr/bin/env python
"""Pretrain on generated radar, fine-tune on scarce real radar.

    python downstream/run_pretrain.py --arch pge --seeds 0 1

THE HYPOTHESIS. Mixing failed because generated frames displace real ones and
contest every gradient step. Scheduling them instead of mixing them removes
that competition: the generated data only initialises the weights, and the
real data then gets the whole fine-tuning budget to itself.

ARMS

    B_limited_real   20 ep real                       standing baseline (existing)
    B_long           40 ep real                       SAME TOTAL STEPS, no generated
    P_ft_<head>      20 ep generated -> 20 ep real    pretrain then fine-tune

`B_long` is the control that makes this readable. Pretraining hands the model
40 epochs of optimisation where the standing baseline got 20, so a win over
`B_limited_real` alone would be ambiguous between "the generated data helped"
and "it just trained longer". Pretraining counts as working only if it beats
`B_long`.

Everything else is held fixed: same detector, same 1000 real frames for the
fine-tune stage (B's exact subset), same 413 real test frames for evaluation,
same scorer.
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

CFG = {"pp": "pointpillar_vod_radar.yaml",
       "pge": "pointpillar_vod_radar_pge.yaml",
       "seg": "pointpillar_vod_radar_seg.yaml"}


def out_root_for(arch, tag):
    """Where OpenPCDet puts things when given a RELATIVE cfg path."""
    return PCDET / "output" / "kitti_models" / CFG[arch].replace(".yaml", "") / tag


def train(arch, arm, tag, seed, epochs, batch, workers, log, pretrained=None):
    # WIPE FIRST -- same trap as run_grid.py. OpenPCDet resumes from the newest
    # checkpoint under this tag, and if it is already at the target epoch it
    # skips training and evaluates the OLD model into a fresh result file. Here
    # that would also silently discard `--pretrained_model`, which is the entire
    # experiment.
    d = out_root_for(arch, tag)
    if d.exists():
        shutil.rmtree(d)
    cmd = [str(PY), "train.py", "--cfg_file", f"cfgs/kitti_models/{CFG[arch]}",
           "--epochs", str(epochs), "--batch_size", str(batch),
           "--workers", str(workers), "--extra_tag", tag,
           "--fix_random_seed", "--seed", str(seed),
           "--set", "DATA_CONFIG.DATA_PATH", str(TREES / arm)]
    if pretrained:
        cmd[-3:-3] = ["--pretrained_model", str(pretrained)]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=PCDET / "tools", stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, (time.time() - t0) / 60, t0


def last_ckpt(arch, tag):
    d = out_root_for(arch, tag) / "ckpt"
    cks = sorted(d.glob("checkpoint_epoch_*.pth"),
                 key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return cks[-1] if cks else None


def score(arch, arm, tag, out_dir, started_at=None):
    """Score, refusing predictions older than the run -- see run_grid.score_one."""
    preds = sorted((out_root_for(arch, tag) / "eval").rglob("result.pkl"))
    if not preds:
        return None
    if started_at is not None and preds[-1].stat().st_mtime < started_at:
        raise SystemExit(
            f"{tag}: the newest predictions predate this run. Refusing to "
            f"score them -- that is a stale artifact, not a result.")
    js = out_dir / f"{tag}.json"
    subprocess.run([str(PY), str(SCORER), "--pred", str(preds[-1]),
                    "--infos", str(TREES / arm / "kitti_infos_val.pkl"),
                    "--json", str(js)], check=True, capture_output=True, text=True)
    return json.loads(js.read_text())


def show(tag, r):
    g = lambda k: (r[k]["AP"] if isinstance(r.get(k), dict) else r.get(k, float("nan")))
    print(f"    {tag}: mAP {r.get('mAP', float('nan')):.4f}  Car {g('Car'):.4f}  "
          f"Ped {g('Pedestrian'):.4f}  Cyc {g('Cyclist'):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="pge", choices=list(CFG))
    ap.add_argument("--heads", nargs="+", default=["mlp", "prism"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--pretrain-epochs", type=int, default=20)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REAR / "downstream" / "runs"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    total = args.pretrain_epochs + args.finetune_epochs
    print("=" * 74)
    print(f"PRETRAIN -> FINETUNE  arch={args.arch}")
    print("=" * 74)
    print(f"  pretrain : {args.pretrain_epochs} ep on generated only")
    print(f"  finetune : {args.finetune_epochs} ep on B's 1000 real frames")
    print(f"  control  : B_long = {total} ep real only (same total steps)\n")

    results = {}

    # ---- control: same total steps, no generated data --------------------
    for seed in args.seeds:
        tag = f"{args.arch}_B_long_s{seed}"
        print(f"  [control] {tag} ({total} ep real) ...", flush=True)
        rc, mins, started_at = train(args.arch, "B_limited_real", tag, seed, total,
                         args.batch, args.workers, out / f"{tag}.log")
        print(f"    exited rc={rc} in {mins:.1f} min", flush=True)
        r = score(args.arch, "B_limited_real", tag, out, started_at=started_at)
        if r:
            results[tag] = r; show(tag, r)

    # ---- pretrain -> finetune, per generated head ------------------------
    for head in args.heads:
        for seed in args.seeds:
            pt = f"{args.arch}_pre_{head}_s{seed}"
            print(f"  [pretrain] {pt} ...", flush=True)
            rc, mins, started_at = train(args.arch, f"P_gen_{head}", pt, seed,
                             args.pretrain_epochs, args.batch, args.workers,
                             out / f"{pt}.log")
            ck = last_ckpt(args.arch, pt)
            print(f"    exited rc={rc} in {mins:.1f} min; ckpt="
                  f"{ck.name if ck else 'NONE'}", flush=True)
            if ck is None:
                print("    no checkpoint -- skipping fine-tune", flush=True)
                continue

            ft = f"{args.arch}_P_ft_{head}_s{seed}"
            print(f"  [finetune] {ft} from {ck.name} ...", flush=True)
            rc, mins, started_at = train(args.arch, "B_limited_real", ft, seed,
                             args.finetune_epochs, args.batch, args.workers,
                             out / f"{ft}.log", pretrained=ck)
            print(f"    exited rc={rc} in {mins:.1f} min", flush=True)
            r = score(args.arch, "B_limited_real", ft, out, started_at=started_at)
            if r:
                results[ft] = r; show(ft, r)
            (out / f"pretrain_{args.arch}.json").write_text(json.dumps(results, indent=1))

    (out / f"pretrain_{args.arch}.json").write_text(json.dumps(results, indent=1))
    print(f"\n  wrote {out / f'pretrain_{args.arch}.json'}")


if __name__ == "__main__":
    main()
