#!/usr/bin/env python
"""Train a JOINT head -- detection and RCS from one network -- on a dataset with
real paired LiDAR+radar.

    python scripts/train_heads.py --dataset vod --head prism --seeds 0 1 2 3 4
    python scripts/train_heads.py --dataset vod --head mlp   --seeds 0 1 2 3 4

The gradient-trained stage of the fit. `scripts/fit_dataset.py`'s cheap stages
run before it (candidates, association, soft labels) and after it (Platt, RCS
calibration, clutter); see that script's docstring for the order.

WHAT IS HELD FIXED ACROSS ARCHITECTURES
---------------------------------------
Everything except the network. Both arms see the same candidate enumeration, the
same 8-feature `drop` variant in the same column order, the same 10-column
material block, the same median/IQR scaler fitted on TRAIN only, the same RCS
target standardisation, the same sequence-level split, the same optimiser, batch
size, validation cadence and early-stopping rule. An architecture comparison in
which anything else moves is not an architecture comparison.

SUB-EPOCH VALIDATION IS NOT OPTIONAL
------------------------------------
One epoch is ~1,683 optimiser steps at batch 4096, and the joint model reaches
its best validation score INSIDE the first epoch and then overfits (val AUC
0.8477 -> 0.8413, val w-R^2 0.4122 -> 0.3648 over eight epochs). Validating once
per epoch snapshots an arbitrary point on a declining curve and produced a
+/-0.0125 seed spread. Validation runs every `--eval-every` steps with patience
counted in VALIDATIONS, and this applies to every arm including the baseline.

MATERIAL IS PART OF THE INPUT, NOT AN EXPERIMENT
------------------------------------------------
The material block is built once per frame during the cache pass and appended to
the scaled physics features. It is deliberately NOT passed through the scaler:
it is already bounded (compositions and |Gamma| in [0, 1]) and standardising it
would hand a near-zero-IQR column such as `metal_fraction` an enormous
multiplier -- and keeping it out means the physics scaler is byte-identical to a
run without material.

`--material` selects the block, defaulting to the dataset config's own
`materials.variant`:

    em    the 10-column electromagnetic descriptor -- PRODUCTION
    comp  the raw material composition (n_materials columns)
    sem   raw semantic class fractions, no physics at all
    c2    CONTROL: the em block with the material->constant table DERANGED
    cov   CONTROL: `frac_labelled` alone. A candidate only gets material if its
          points project into the camera, so the block silently carries
          camera-FOV membership, which correlates with detectability. This arm
          isolates that channel so a lift from pure coverage cannot be reported
          as material.
    none  no material block

The controls are not decoration. `c2` scores within 0.0012 w-R^2 of `em` on a
0.0087 seed floor, which is how this project knows the lift is the semantic
partition rather than the permittivity values.

SIDE ARTIFACTS the next fit_dataset.py stages consume:
  outputs/fit_<dataset>/detection_scores_val_<head>.npz   -> --stage platt
  outputs/fit_<dataset>/rcs_pred_train_<head>.npz         -> --stage rcs_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from _fit_common import RAW_FEATURE_NAMES, build_labeled_frame  # noqa: E402
from rear.calibration.feature_variants import (  # noqa: E402
    VARIANT_DROP, VARIANTS, apply_variant, variant_feature_names)
from rear.calibration.scaling import (  # noqa: E402
    STANDARDISED_CLIP, apply_scaler, fit_robust_scaler)
from rear.data.base import load_adapter  # noqa: E402
from rear.materials.features import (MATERIAL_FEATURE_NAMES,  # noqa: E402
                                     features_from_composition)
from rear.materials.database import MATERIAL_DB, MATERIAL_NAMES  # noqa: E402
from rear.models.base import available_architectures, build_head, save_head  # noqa: E402
from rear.models.losses import joint_loss  # noqa: E402

MAX_EPOCHS = 50
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
EVAL_EVERY = 200          # optimiser steps between validations -- see docstring
PATIENCE = 15             # in VALIDATIONS, not epochs
DEFAULT_VARIANT = VARIANT_DROP

MATERIAL_VARIANTS = ("em", "comp", "sem", "c2", "cov", "none")

#: Per-architecture RCS objective. A property of the head's OUTPUT TYPE (point
#: estimate vs mixture), not of its internal structure -- see rear/models/losses.py.
RCS_LOSS = {"prism": "nll+mse", "mlp": "huber"}


def _fit_dir(dataset: str) -> Path:
    return REPO_ROOT / "outputs" / f"fit_{dataset}"


def _cache_dir(dataset: str) -> Path:
    return _fit_dir(dataset) / "cache"


# --------------------------------------------------------------- the cache ---
def build_split_cache(adapter, split, ids, cache_dir, extractor=None,
                      use_cache=True, max_frames=None):
    """Enumerate candidates, build V3 soft labels and extract material for every
    frame in `split`, cached to one npz.

    Enumeration (occlusion ray-casting, per-voxel normal fits) and segmentation
    dominate the cost, so both happen in ONE pass over the frames: running them
    separately would double the expensive part and risk the two passes drifting
    onto different candidate sets.
    """
    ids = ids if max_frames is None else ids[:max_frames]
    tag = f"{split}_{max_frames or 'all'}"
    cache_path = cache_dir / f"labeled_{tag}.npz"
    want_material = extractor is not None and extractor.enabled

    if use_cache and cache_path.exists():
        d = np.load(cache_path, allow_pickle=False)
        have = {k: d[k] for k in d.files}
        if want_material and "mat_comp" not in have:
            print(f"  [{split}] cache {cache_path.name} predates the material "
                  f"pipeline -- rebuilding")
        else:
            print(f"  [{split}] loading cache {cache_path.name}")
            return have

    seq_of = adapter.sequence_of() if hasattr(adapter, "sequence_of") else None
    acc = {k: [] for k in ("X", "y", "pos", "neg", "frame", "seq",
                           "winning_mask", "winning_rcs")}
    mat = {k: [] for k in ("comp", "frac", "semfrac", "conf")}
    ego_speed, skipped = [], []
    t0 = time.time()
    for i, fid in enumerate(ids):
        # A dataset can ship an unreadable frame -- VoD has exactly one in
        # 8,682, a zero-byte pose file, so its ego velocity cannot be computed.
        # Skipping it loses one frame; fabricating an ego velocity would
        # silently corrupt that frame's features AND its soft labels, and the
        # corruption would be invisible in every summary statistic. The skip is
        # recorded and reported rather than swallowed.
        try:
            frame = adapter.load_frame(fid)
        except (KeyError, OSError, ValueError) as exc:
            skipped.append((str(fid), type(exc).__name__, str(exc)))
            ego_speed.append(0.0)
            continue
        row = build_labeled_frame(adapter, frame)
        n = len(row["y"])
        ego_speed.append(float(np.linalg.norm(frame.ego_velocity)))
        if n == 0:
            continue
        acc["X"].append(row["X"]); acc["y"].append(row["y"])
        acc["pos"].append(row["pos"]); acc["neg"].append(row["neg"])
        acc["frame"].append(np.full(n, i, dtype=np.int32))
        seq = seq_of.get(str(frame.frame_id), i) if seq_of else i
        acc["seq"].append(np.full(n, seq, dtype=np.int32))

        win = row["winning_detection"]
        m = win >= 0
        acc["winning_mask"].append(m)
        if m.any() and frame.radar is not None:
            from _fit_common import radar_rcs_and_vr_comp
            rcs, _vr = radar_rcs_and_vr_comp(adapter, frame)
            acc["winning_rcs"].append(rcs[win[m]])
        else:
            acc["winning_rcs"].append(np.zeros(0))

        if want_material:
            d = extractor.composition_for_frame(
                frame, row["candidates"], row["grid_config"],
                point_semantic=row["point_semantic"])
            mat["comp"].append(d["composition"].astype(np.float32))
            mat["frac"].append(d["frac_labelled"].astype(np.float32))
            mat["semfrac"].append(d["class_fractions"].astype(np.float32))
            mat["conf"].append(d["mean_confidence"].astype(np.float32))

        if (i + 1) % 100 == 0 or i == len(ids) - 1:
            el = time.time() - t0
            print(f"  [{split}] frame {i + 1}/{len(ids)}  ({el / (i + 1):.2f}s/frame, "
                  f"eta {(len(ids) - i - 1) * el / (i + 1) / 60:.1f} min)", flush=True)

    out = {
        "X": np.concatenate(acc["X"]) if acc["X"] else np.zeros((0, len(RAW_FEATURE_NAMES)), np.float32),
        "y": np.concatenate(acc["y"]) if acc["y"] else np.zeros(0, np.float32),
        "pos": np.concatenate(acc["pos"]) if acc["pos"] else np.zeros(0, bool),
        "neg": np.concatenate(acc["neg"]) if acc["neg"] else np.zeros(0, bool),
        "frame": np.concatenate(acc["frame"]) if acc["frame"] else np.zeros(0, np.int32),
        "seq": np.concatenate(acc["seq"]) if acc["seq"] else np.zeros(0, np.int32),
        "winning_mask": np.concatenate(acc["winning_mask"]) if acc["winning_mask"] else np.zeros(0, bool),
        "winning_rcs": np.concatenate(acc["winning_rcs"]) if acc["winning_rcs"] else np.zeros(0),
        "ego_speed_by_frame": np.array(ego_speed),
        "n_frames": np.array([len(ids)]),
    }
    if want_material:
        out["mat_comp"] = np.concatenate(mat["comp"])
        out["mat_frac"] = np.concatenate(mat["frac"])
        out["mat_semfrac"] = np.concatenate(mat["semfrac"])
        out["mat_conf"] = np.concatenate(mat["conf"])
        out["mat_material_names"] = np.array(MATERIAL_NAMES)
        out["mat_label_space"] = np.array([extractor.label_space.name])
        if len(out["mat_comp"]) != len(out["y"]):
            raise RuntimeError(
                f"[{split}] material rows {len(out['mat_comp']):,} != candidate "
                f"rows {len(out['y']):,}; refusing to write a misaligned cache")

    out["n_skipped"] = np.array([len(skipped)])
    if skipped:
        print(f"  [{split}] SKIPPED {len(skipped)} unloadable frame(s): "
              + ", ".join(f"{f} ({t})" for f, t, _ in skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    extra = ""
    if want_material:
        f = out["mat_frac"]
        extra = (f" | material: {100 * (f > 0).mean():.1f}% of candidates, "
                 f"mean frac_labelled {f.mean():.3f}")
    print(f"  [{split}] cached -> {cache_path.name} ({len(out['y']):,} candidates, "
          f"{int(out['winning_mask'].sum()):,} matched){extra}")
    return out


def material_block(cache: dict, variant: str, freq_ghz: float | None) -> np.ndarray:
    """The material feature block for one split, from cached composition.

    Every variant is a function of the SAME cached 11-float composition, which
    is what makes the controls cheap enough to be mandatory rather than
    aspirational -- `c2` is a matrix multiply against a permuted table, not a
    re-extraction.
    """
    if variant == "none":
        return np.zeros((len(cache["y"]), 0), dtype=np.float64)
    if "mat_comp" not in cache:
        raise SystemExit(
            f"--material {variant} was requested but the cache carries no "
            f"material. Rebuild it with --no-cache after enabling `materials:` "
            f"in the dataset config.")
    comp = np.asarray(cache["mat_comp"], dtype=np.float64)
    frac = np.asarray(cache["mat_frac"], dtype=np.float64)
    if variant == "comp":
        return comp
    if variant == "sem":
        return np.asarray(cache["mat_semfrac"], dtype=np.float64)
    if variant == "cov":
        return frac.reshape(-1, 1)
    descriptors = (MATERIAL_DB.permuted_descriptors(0, freq_ghz)
                   if variant == "c2" else None)
    return features_from_composition(comp, frac, freq_ghz=freq_ghz,
                                     descriptors=descriptors)


def material_feature_names(variant: str, label_space=None) -> list:
    if variant == "none":
        return []
    if variant in ("em", "c2"):
        return list(MATERIAL_FEATURE_NAMES)
    if variant == "comp":
        return [f"comp_{m}" for m in MATERIAL_NAMES]
    if variant == "cov":
        return ["frac_labelled"]
    if variant == "sem":
        return ([f"sem_{n}" for n in label_space.class_names] if label_space
                else ["sem"])
    raise ValueError(variant)


# ------------------------------------------------------------------ metrics ---
def roc_auc(pos, score):
    if not pos.any() or pos.all():
        return float("nan")
    return float(roc_auc_score(pos, score))


def weighted_r2(target, prediction, weight):
    wm = np.average(target, weights=weight)
    ss_res = float(np.sum(weight * (target - prediction) ** 2))
    ss_tot = float(np.sum(weight * (target - wm) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ----------------------------------------------------------------- training ---
def train_one(architecture, data, tensors, transforms, feature_names,
              mat_names, seed, device, args, arch_kwargs):
    """Train one seed and return (model, history, selection record)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_mat = len(mat_names)
    model = build_head(architecture, n_features=len(feature_names),
                       n_material=n_mat, **arch_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rcs_kind = RCS_LOSS[architecture]
    n_train = len(tensors["train"]["X"])
    print(f"\n{architecture} seed {seed}: {model.n_parameters():,} params, "
          f"{len(feature_names)} physics + {n_mat} material inputs, "
          f"train rows {n_train:,}, rcs loss {rcs_kind}")

    # A whole split is millions of candidates and the spectral embedding widens
    # each row by an order of magnitude, so evaluation is chunked. This changes
    # no number -- the loss below is recomputed once over the concatenated
    # outputs, not averaged over chunks, which would silently weight the last
    # short chunk equally with the full ones.
    EVAL_CHUNK = 262144

    def evaluate(split):
        model.eval()
        s = tensors[split]
        with torch.no_grad():
            parts = [model(s["X"][i:i + EVAL_CHUNK])
                     for i in range(0, len(s["X"]), EVAL_CHUNK)]
            out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            loss = joint_loss(out, s["y"], s["rcs"], s["w"], s["m"], rcs_kind,
                              lam=args.lam).item()
            p = torch.sigmoid(out["det_logit"]).cpu().numpy()
            pred = out["rcs_mean"].cpu().numpy()
        d = data[split]
        keep = d["pos"] | d["neg"]
        auc = roc_auc(d["pos"][keep], p[keep]) if keep.any() else float("nan")
        m = d["winning_mask"]
        # w-R^2 is scored in RAW dBsm, not standardised units: the prediction is
        # de-standardised with the head's own constants and compared against the
        # untouched target, so the number means the same thing across arms whose
        # target statistics could in principle differ.
        wr2 = (weighted_r2(transforms["y_raw"][split],
                           pred[m] * transforms["y_sigma"] + transforms["y_mu"],
                           d["y"][m].astype(np.float64)) if m.any() else float("nan"))
        return dict(loss=loss, auc=auc, wr2=wr2)

    best = {"loss": float("inf"), "state": None, "step": -1, "val": None}
    history, stale, step = [], 0, 0
    print(f"{'eval':>5}{'epoch':>6}{'step':>8}{'train':>10}{'val loss':>10}"
          f"{'val AUC':>10}{'val w-R2':>10}{'secs':>7}")
    t0 = time.time()
    stop = False
    for epoch in range(args.max_epochs):
        perm = torch.randperm(n_train, device=device)
        model.train()
        run, run_n = 0.0, 0
        for s0 in range(0, n_train, BATCH_SIZE):
            idx = perm[s0:s0 + BATCH_SIZE]
            s = tensors["train"]
            opt.zero_grad()
            out = model(s["X"][idx])
            loss = joint_loss(out, s["y"][idx], s["rcs"][idx], s["w"][idx],
                              s["m"][idx], rcs_kind, lam=args.lam)
            loss.backward()
            opt.step()
            run += loss.item() * len(idx); run_n += len(idx)
            step += 1

            if step % args.eval_every == 0:
                v = evaluate("val")
                history.append(dict(eval=len(history) + 1, epoch=epoch, step=step,
                                    train=run / max(run_n, 1), **v))
                print(f"{len(history):>5}{epoch:>6}{step:>8}{run / max(run_n, 1):>10.5f}"
                      f"{v['loss']:>10.5f}{v['auc']:>10.4f}{v['wr2']:>10.4f}"
                      f"{time.time() - t0:>7.1f}", flush=True)
                run, run_n = 0.0, 0
                if v["loss"] < best["loss"] - 1e-6:
                    best = {"loss": v["loss"], "step": step, "val": v,
                            "state": {k: t.detach().clone()
                                      for k, t in model.state_dict().items()}}
                    stale = 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        print(f"  early stop: no val improvement for "
                              f"{args.patience} validations")
                        stop = True
                        break
                model.train()
        if stop:
            break

    if best["state"] is None:
        raise SystemExit("training ended before the first validation -- lower "
                         "--eval-every or check the split sizes")
    model.load_state_dict(best["state"])
    model.eval()
    return model, history, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--head", required=True, choices=available_architectures(),
                    help="which architecture to train")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one checkpoint per seed; the FIRST seed's model is "
                         "written as the production <head>_head.pt")
    ap.add_argument("--material", default=None, choices=MATERIAL_VARIANTS,
                    help="default: the dataset config's materials.variant")
    ap.add_argument("--feature-variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight on the RCS term of the joint objective")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="in VALIDATIONS, not epochs")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--max-frames-per-split", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--material-device", default=None,
                    help="where the segmentation model runs (default: --device)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--tag", default="",
                    help="suffix for checkpoint/result filenames, so a variant "
                         "run cannot overwrite the production head")
    ap.add_argument("--cache-only", action="store_true",
                    help="build the per-frame cache and stop")
    args = ap.parse_args()

    adapter = load_adapter(args.dataset)
    if not adapter.has_real_radar():
        raise SystemExit(
            f"{args.dataset} has no real radar: nothing to train against.\n"
            f"Point its config at a donor with `fitted_from:` instead.")
    splits = adapter.sequence_splits()
    missing = [s for s in ("train", "val", "test") if s not in splits]
    if missing:
        raise SystemExit(f"{args.dataset}.sequence_splits() is missing {missing}")

    mat_cfg = adapter.material_config()
    variant = args.material or (mat_cfg.get("variant", "em")
                                if mat_cfg.get("enabled") else "none")
    freq = float(mat_cfg.get("frequency_ghz", MATERIAL_DB.default_frequency_ghz))

    extractor = None
    if mat_cfg.get("enabled"):
        from rear.materials.pipeline import MaterialExtractor
        extractor = MaterialExtractor(adapter,
                                      device=args.material_device or args.device)

    weights_dir = (Path(args.weights_dir) if args.weights_dir else
                   REPO_ROOT / str(adapter.fitted("weights_dir",
                                                  default=f"weights/{args.dataset}")))
    weights_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = _fit_dir(args.dataset)
    fit_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset      : {adapter}")
    print(f"architecture : {args.head}")
    print(f"material     : {variant}"
          + (f" (label space {extractor.label_space.name}, source "
             f"{extractor.source}, {freq:g} GHz)" if extractor else ""))
    print(f"weights      : {weights_dir}")
    print(f"protocol     : Adam lr={LEARNING_RATE}, batch {BATCH_SIZE}, "
          f"validate every {args.eval_every} steps, patience {args.patience}\n")

    wall0 = time.time()
    data = {}
    for sp in ("train", "val", "test"):
        print(f"building [{sp}] ({len(splits[sp])} frames available)...")
        data[sp] = build_split_cache(adapter, sp, splits[sp], _cache_dir(args.dataset),
                                     extractor=extractor, use_cache=not args.no_cache,
                                     max_frames=args.max_frames_per_split)
    if args.cache_only:
        print(f"\ncache built in {(time.time() - wall0) / 60:.1f} min")
        return

    print("\n" + "=" * 78)
    print("SPLIT COMPOSITION")
    print("=" * 78)
    print(f"{'split':<8}{'frames':>9}{'candidates':>13}{'matched':>10}"
          f"{'label mean':>13}{'with material':>15}")
    for sp in ("train", "val", "test"):
        d = data[sp]
        wm = (f"{100 * (d['mat_frac'] > 0).mean():>14.1f}%"
              if "mat_frac" in d else f"{'--':>15}")
        print(f"{sp:<8}{int(d['n_frames'][0]):>9}{len(d['y']):>13,}"
              f"{int(d['winning_mask'].sum()):>10,}"
              f"{(d['y'].mean() if len(d['y']) else float('nan')):>13.4f}{wm}")

    # ---- features: variant, scaler (train only), material appended after ----
    feature_names = variant_feature_names(RAW_FEATURE_NAMES, args.feature_variant)
    for sp in data:
        data[sp]["X"], n_undef = apply_variant(
            data[sp]["X"], data[sp]["frame"], data[sp]["ego_speed_by_frame"],
            args.feature_variant)
        if n_undef:
            print(f"  [{sp}] {n_undef:,} rows below the direction threshold")

    mu, sigma = fit_robust_scaler(data["train"]["X"])
    ytr = data["train"]["winning_rcs"].astype(np.float64)
    if len(ytr) < 50:
        raise SystemExit(
            f"only {len(ytr)} matched train candidates -- too few to train an "
            f"RCS output. Check the association operating point.")
    y_mu, y_sigma = float(ytr.mean()), float(ytr.std())
    mat_names = material_feature_names(
        variant, extractor.label_space if extractor else None)

    transforms = {"mu": mu, "sigma": sigma, "clip": STANDARDISED_CLIP,
                  "y_mu": y_mu, "y_sigma": y_sigma, "y_raw": {}}
    tensors = {}
    for sp in ("train", "val", "test"):
        d = data[sp]
        Xs = apply_scaler(d["X"], mu, sigma, STANDARDISED_CLIP)
        if mat_names:
            block = material_block(d, variant, freq)
            if block.shape[1] != len(mat_names):
                raise RuntimeError(f"[{sp}] material block width {block.shape[1]} "
                                   f"!= {len(mat_names)} named columns")
            Xs = np.hstack([Xs, block])
        m = d["winning_mask"]
        # RCS targets live only on matched candidates; scatter them back onto
        # the full candidate vector so one tensor serves both outputs and the
        # loss can mask rather than index two differently-sized batches.
        rcs_full = np.zeros(len(d["y"]), dtype=np.float64)
        w_full = np.zeros(len(d["y"]), dtype=np.float64)
        y_true = d["winning_rcs"].astype(np.float64)
        rcs_full[m] = (y_true - y_mu) / y_sigma
        w_full[m] = d["y"][m].astype(np.float64)
        transforms["y_raw"][sp] = y_true
        tensors[sp] = {
            "X": torch.tensor(Xs, dtype=torch.float32, device=args.device),
            "y": torch.tensor(d["y"].astype(np.float64), dtype=torch.float32,
                              device=args.device),
            "rcs": torch.tensor(rcs_full, dtype=torch.float32, device=args.device),
            "w": torch.tensor(w_full, dtype=torch.float32, device=args.device),
            "m": torch.tensor(m, dtype=torch.bool, device=args.device),
        }

    arch_kwargs = dict(adapter.fitted("heads", args.head, default={}) or {})
    for k in ("artifact", "arm", "rcs_loss"):
        arch_kwargs.pop(k, None)
    if args.head == "mlp" and "hidden" in arch_kwargs:
        arch_kwargs["hidden"] = tuple(arch_kwargs["hidden"])

    # A readable arm name, recorded on every checkpoint so a results file can
    # always be traced back to what produced it.
    arm_name = adapter.fitted("heads", args.head, "arm",
                              default=f"{args.head}_{variant}")

    material_record = {
        "enabled": bool(mat_names), "variant": variant,
        "n_features": len(mat_names), "feature_names": mat_names,
        "label_space": extractor.label_space.name if extractor else None,
        "source": extractor.source if extractor else None,
        "frequency_ghz": freq,
        "database": MATERIAL_DB.itu_version,
    }

    # ---- one run per seed --------------------------------------------------
    results = []
    for si, seed in enumerate(args.seeds):
        model, history, best = train_one(
            args.head, data, tensors, transforms, feature_names, mat_names,
            seed, args.device, args, arch_kwargs)
        test = {}
        model.eval()
        with torch.no_grad():
            Xt = tensors["test"]["X"]
            parts = [model(Xt[i:i + 262144]) for i in range(0, len(Xt), 262144)]
            out = {k: torch.cat([q[k] for q in parts]) for k in parts[0]}
            p = torch.sigmoid(out["det_logit"]).cpu().numpy()
            pred = out["rcs_mean"].cpu().numpy()
        d = data["test"]
        keep = d["pos"] | d["neg"]
        test["auc"] = roc_auc(d["pos"][keep], p[keep])
        m = d["winning_mask"]
        test["wr2"] = weighted_r2(transforms["y_raw"]["test"],
                                  pred[m] * y_sigma + y_mu,
                                  d["y"][m].astype(np.float64))
        print(f"  seed {seed}: val AUC {best['val']['auc']:.4f} "
              f"w-R2 {best['val']['wr2']:.4f} | TEST AUC {test['auc']:.4f} "
              f"w-R2 {test['wr2']:.4f}")

        suffix = f"_s{seed}" if len(args.seeds) > 1 else ""
        name = f"{args.head}_head{args.tag}"
        ck_path = weights_dir / f"{name}{suffix}.pt"
        save_head(ck_path, model, scaler={"mu": mu, "sigma": sigma,
                                          "clip": STANDARDISED_CLIP},
                  feature_names=feature_names,
                  feature_variant=args.feature_variant,
                  y_mu=y_mu, y_sigma=y_sigma, material=material_record,
                  extra={"scaling": "median_iqr", "seed": seed,
                         "dataset": args.dataset,
                         "arm": arm_name,
                         "rcs_loss": RCS_LOSS[args.head],
                         "val": best["val"], "test": test,
                         "selected_step": best["step"]})
        print(f"  saved {ck_path}")
        if si == 0:
            # The production checkpoint is the FIRST seed, chosen before any
            # test score is seen. Picking the best seed by test AUC would be
            # selection on the test split; picking it by val would still make
            # the reported test number optimistic. Seed 0 is arbitrary and
            # therefore honest, and the multi-seed spread is reported alongside.
            prod = weights_dir / f"{name}.pt"
            if ck_path != prod:
                import shutil
                shutil.copyfile(ck_path, prod)
                print(f"  production checkpoint -> {prod}")
            _write_side_artifacts(model, tensors, data, transforms, fit_dir,
                                  args.head, y_mu, y_sigma)
        results.append(dict(seed=seed, val=best["val"], test=test,
                            n_params=model.n_parameters(),
                            selected_step=best["step"], history=history))

    if len(results) > 1:
        aucs = np.array([r["test"]["auc"] for r in results])
        wr2s = np.array([r["test"]["wr2"] for r in results])
        # The SPREAD is the number that matters, not the mean: it is the floor
        # any architecture or feature claim has to clear, and a difference
        # smaller than it is not a result.
        print(f"\n{len(results)} seeds -- TEST AUC {aucs.mean():.4f} "
              f"+/- {aucs.std():.4f} (spread {np.ptp(aucs):.4f}) | "
              f"w-R2 {wr2s.mean():.4f} +/- {wr2s.std():.4f} "
              f"(spread {np.ptp(wr2s):.4f})")

    res_path = fit_dir / f"train_{args.head}{args.tag}.json"
    res_path.write_text(json.dumps({
        "dataset": args.dataset, "architecture": args.head,
        "material": material_record, "feature_names": feature_names,
        "protocol": {"lr": LEARNING_RATE, "batch": BATCH_SIZE,
                     "eval_every": args.eval_every, "patience": args.patience,
                     "lam": args.lam, "seeds": args.seeds},
        "results": results}, indent=1, default=float))
    print(f"\nwrote {res_path}\nwall clock: {(time.time() - wall0) / 60:.1f} min")


def _write_side_artifacts(model, tensors, data, transforms, fit_dir, head,
                          y_mu, y_sigma):
    """Val detection scores and train RCS predictions -- the inputs the two
    calibration stages read. Written per architecture, because both
    calibrations are properties of a specific head's output distribution."""
    model.eval()
    def _chunked(X, key):
        with torch.no_grad():
            return torch.cat([model(X[i:i + 262144])[key]
                              for i in range(0, len(X), 262144)]).cpu().numpy()

    val_p = 1.0 / (1.0 + np.exp(-_chunked(tensors["val"]["X"], "det_logit")))
    tr_pred = _chunked(tensors["train"]["X"], "rcs_mean")
    np.savez_compressed(fit_dir / f"detection_scores_val_{head}.npz",
                        p=val_p, frame=data["val"]["frame"])
    m = data["train"]["winning_mask"]
    np.savez_compressed(fit_dir / f"rcs_pred_train_{head}.npz",
                        pred=tr_pred[m] * y_sigma + y_mu,
                        winning=data["train"]["winning_rcs"].astype(np.float64))
    print(f"  side artifacts -> {fit_dir}/detection_scores_val_{head}.npz, "
          f"rcs_pred_train_{head}.npz")


if __name__ == "__main__":
    main()
