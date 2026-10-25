#!/usr/bin/env bash
# Fit and evaluate BOTH architectures end to end, under the final
# material-information pipeline. This is the script that produced everything in
# output/, and running it from a clean checkout reproduces it.
#
#   bash scripts/run_all_experiments.sh                 # everything
#   bash scripts/run_all_experiments.sh train           # one stage
#   STAGES="eval report" bash scripts/run_all_experiments.sh
#
# ORDER IS LOAD-BEARING, not stylistic:
#
#   cache      one pass per frame that enumerates candidates, builds V3 soft
#              labels AND extracts material. Segmentation dominates the cost, so
#              it is done once and shared by every arm.
#   train      one joint head per architecture, 5 seeds each. Seed 0 becomes the
#              production checkpoint -- chosen BEFORE any test score is seen.
#   calibrate  BOTH calibrations, per architecture, in this order:
#              detection (expected count) -> RCS -> detection (realised count).
#              The last step runs the generator, which needs an RCS calibration
#              to exist, which is why it cannot come first. Neither calibration
#              is ever shared between heads: each is a function of one head's own
#              output distribution, both are monotone, and a mismatched pair
#              therefore corrupts the emitted cloud while moving no head-level
#              metric at all.
#   eval       head metrics + distributional/structural metrics, per architecture
#   qual       ranked qualitative panels, per architecture
#   downstream generate KITTI-CARLA frames per architecture, build detector
#              trees, train and score the augmentation grid
#   report     render every measured file into markdown AND the distribution
#              charts, per architecture, plus the one side-by-side index
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY=${REAR_PYTHON:-/DATA1/Vivek/Sim2Real_RADAR/Code/physradar/.venv_pcdet/bin/python}
export PYTHONPATH=${REAR_VOD_LOADER:-/DATA1/Vivek/Sim2Real_RADAR/Code/physradar/legacy/src}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

HEADS=${HEADS:-"prism mlp"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
# `downstream` expands to its four sub-stages (kc, trees, grid, pretrain); they
# can also be named individually to resume a long run part-way.
STAGES=${STAGES:-${1:-"cache train calibrate eval qual downstream report"}}
LOGS="$ROOT/logs"; mkdir -p "$LOGS"

run() {           # run <logname> <command...>
  local name=$1; shift
  echo "=== $name  ($(date +%H:%M:%S))"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED: see $LOGS/$name.log"; tail -25 "$LOGS/$name.log"; exit 1; }
}

has() { [[ " $STAGES " == *" $1 "* ]]; }

# --------------------------------------------------------------------- cache
if has cache; then
  run cache_build $PY scripts/train_heads.py --dataset vod --head prism \
      --cache-only --device cuda
fi

# --------------------------------------------------------------------- train
if has train; then
  for h in $HEADS; do
    run "train_$h" $PY scripts/train_heads.py --dataset vod --head "$h" \
        --seeds $SEEDS --device cuda
  done
fi

# ----------------------------------------------------------------- calibrate
if has calibrate; then
  for h in $HEADS; do
    run "platt_${h}_expected" $PY scripts/fit_dataset.py \
        --dataset vod --stage platt --head "$h"
    run "rcscal_$h" $PY scripts/fit_dataset.py \
        --dataset vod --stage rcs_calibration --head "$h"
    # Now that both exist, the generator can run, so the Platt map can be
    # matched against the REALISED surface count rather than the expected one.
    run "platt_${h}_realised" $PY scripts/fit_dataset.py \
        --dataset vod --stage platt --head "$h" --match-realized \
        --material-device cuda
  done
fi

# ---------------------------------------------------------------------- eval
if has eval; then
  for h in $HEADS; do
    run "eval_heads_$h" $PY tools/evaluate_heads.py --dataset vod --split test \
        --head "$h" --device cuda --material-device cuda \
        --out "output/$h/quantitative"
    run "eval_quant_$h" $PY tools/evaluate_quantitative.py --dataset vod \
        --split test --head "$h" --out "output/$h/quantitative"
  done
fi

# ---------------------------------------------------------------------- qual
if has qual; then
  for h in $HEADS; do
    run "qual_vod_$h" $PY tools/visualize_best.py --dataset vod --head "$h" \
        --n 10 --cache "output/$h/quantitative" --out "output/$h/qualitative/vod"
    run "qual_kc_$h" $PY tools/visualize_best.py --dataset kitti_carla \
        --head "$h" --n 10 --out "output/$h/qualitative/kitti_carla"
  done
fi

# ---------------------------------------------------------------- downstream
# Split into four resumable sub-stages. This is the longest part of the run by
# an order of magnitude (~8 h of detector training), and re-generating 3,000
# KITTI-CARLA frames because one grid run failed is pure waste. `downstream`
# still means all four.
if has downstream || has kc; then
  for h in $HEADS; do
    run "kc_cache_$h" $PY downstream/build_kc_cache.py --head "$h" --n 1500
  done
fi

if has downstream || has trees; then
  run trees $PY downstream/build_trees.py --heads $HEADS
fi

if has downstream || has grid; then
  for det in pge seg; do
    # 20 epochs, matching the protocol every previously recorded arm was run
    # under. Not a tuning knob: changing it would make these results
    # incomparable to the ones they replace, and the seed floor has to be
    # measured under the same budget it is applied to.
    run "grid_$det" $PY downstream/run_grid.py --arch "$det" --seeds 0 1 --epochs 20
  done
fi

if has downstream || has pretrain; then
  # The SCHEDULING alternative to mixing: pretrain on generated, then fine-tune
  # on the same scarce real set. Its control is B_long -- 40 epochs of real only
  # -- because pretraining hands the model twice the optimisation budget, so a
  # win over the 20-epoch baseline alone would be ambiguous between "the
  # generated data helped" and "it just trained longer".
  run pretrain_trees $PY downstream/build_pretrain_trees.py --heads $HEADS
  run pretrain_pge $PY downstream/run_pretrain.py --arch pge --heads $HEADS --seeds 0 1
fi

# -------------------------------------------------------------------- report
if has report; then
  for h in $HEADS; do
    mkdir -p "output/$h/downstream"
    $PY downstream/build_downstream_report.py --head "$h" --arch pge seg \
        || echo "  (no downstream runs for $h yet)"
    $PY tools/plot_distributions.py --head "$h"
    $PY tools/build_report.py --head "$h"
  done
  $PY tools/build_index.py
fi

echo "=== done ($(date +%H:%M:%S)) -- see output/README.md"
