# ReaR — Realistic Radar generation

Generate physically-grounded **3+1D automotive radar point clouds**
`[x, y, z, RCS, v_r]` from a LiDAR sweep. ReaR learns the radar detection
process — what gets detected and how strongly it reflects — from real paired
LiDAR↔radar data, and applies it to LiDAR from anywhere: a different sensor, a
simulator, a different city.

```
   LiDAR + per-point semantics                Radar point cloud
   (x, y, z, cosine/intensity)  ──ReaR──▶     (x, y, z, RCS, v_r)
   ~100k points                               ~200-350 points
```

Every candidate carries **material information derived from permittivity** — a
semantic label resolved through a material mixture into ITU-R P.2040-4
electromagnetic constants. That is a permanent, default-on stage, not an option.

No training needed to use it: `weights/vod/` ships a fitted profile for both
architectures.

---

## Install

```bash
pip install -e ".[kitti-carla,viz]"      # or: pip install -r requirements.txt
```

Torch, NumPy, SciPy, scikit-learn and PyYAML are always required.
`transformers` is needed **only** to fit on a dataset whose LiDAR carries no
per-point semantics — never to generate.

## Quickstart

```bash
# a whole KITTI-CARLA town (ships per-point semantics; material is automatic)
python scripts/generate.py --dataset kitti_carla --split Town01 --out out/town01

# your own sweep, with per-point semantic labels
python scripts/generate.py --lidar scan.npy --labels labels.npy \
    --label-space carla --ego-velocity 8.0 0 0 --out scan_radar.bin

# inspect the material/permittivity database
python tools/inspect_materials.py --label-space cityscapes

# BEV: LiDAR objects next to where the generated radar landed
python tools/visualize.py --dataset kitti_carla --split Town01 --frame 5 --out viz.png
```

Output is raw `float32`, shape `(M, 5)`:
`np.fromfile(path, dtype=np.float32).reshape(-1, 5)`.

Full formats, data layout and fitting instructions: **[USAGE.md](USAGE.md)**.

## Pipeline

| stage | what happens |
|---|---|
| **1 · Candidates** | voxelise at 0.4 m; every occupied, non-occluded voxel in the FOV is a candidate, described by 8 physics features (range, azimuth, elevation, incidence cosine, height above local ground, point density, path loss, occlusion depth) |
| **2 · Material** | each point's semantic label → a *mixture* over 11 materials → a 10-column electromagnetic block per voxel |
| **3 · Head** | one network emitting BOTH outputs from `[physics \| material]`: a detection logit and an RCS estimate |
| **4 · Calibration** | Platt map on the detection probability so per-frame counts match real ones; two-piece quantile map putting RCS onto the real marginal. Both are **per-architecture and never shared** |
| **5 · Clutter** | inhomogeneous Poisson process fitted to real non-surface detections, sampled and added |

`v_r` is **computed analytically** from ego motion, never predicted — exact for
static scatterers by construction.

## Architectures

Both are **joint heads**: one network, both outputs, one self-describing
checkpoint. Same candidates, features, material block and training protocol, so
a difference between them is the architecture and nothing else. Select with
`fitted.heads.implementation` in the dataset config, or `--head` on any CLI.

| | **PRISM** (default) | **MLP** |
|---|---|---|
| structure | five link-budget branches over disjoint feature groups mirroring the radar range equation, spectral embedding (K=4), material branch | 64-64 ReLU over the concatenated input |
| parameters | ~86k | ~5.5k |
| RCS output | 3-component mixture density — models the specular tail | point estimate, robust loss — down-weights it |
| use it when | held-out accuracy matters; the link budget is inspectable after training | cross-dataset smoothness matters, or as the control that makes PRISM's numbers readable |

Components measured and rejected are **deleted, not switched off** — the K=8
frequency bank, FiLM, the bilinear term, the residual path, range-adaptive
voxelisation and the semantic-road ground reference. Each file states what its
removal cost, with numbers.

## Material information

```
dataset → semantic label → material mixture → permittivity → model input
```

The physics is a **database in this repository**, not constants in code:

| file | contents |
|---|---|
| `rear/materials/data/itu_p2040_4.yaml` | ITU-R P.2040-4 Table 3, with each entry's frequency band of validity |
| `rear/materials/data/materials.yaml` | the 11-material palette: ITU entry or cited local constants, RMS roughness, confidence, provenance |
| `configs/materials/<space>.yaml` | one file per semantic taxonomy (`cityscapes`, `carla`): class → material mixture |

A class maps to a *mixture*, not one material: a car is metal with glass
apertures and rubber at the bottom, and at 77 GHz those differ by ~15 dB. Band
validity travels with every value, so extrapolated constants and the four
materials outside the Recommendation are flagged rather than smoothed over.

Adding a material is one YAML edit. **Adding a dataset with a new taxonomy is a
new file in `configs/materials/` plus one line in its dataset config** — no code
changes.

> **What the material block is worth, honestly.** It is the largest head-level
> improvement in this framework (w-R² 0.3957 → 0.4470, 5 seeds), but its
> pre-registered control shows **the permittivity values are not what carries
> it**: an identical block with the material→constant table deranged scores
> 0.4458, inside the 0.0173 seed floor. The information is the semantic
> partition. The EM encoding is kept because it is a fixed 10-column contract
> independent of any segmenter's vocabulary, and the only representation in
> which two incompatible label taxonomies become one coordinate system — which
> is what deployment on a new dataset needs. Reproduce with
> `scripts/train_heads.py --material {none,cov,c2,em}`.

## Layout

```
rear/
  data/            dataset adapters + the DatasetAdapter contract
  geometry/        candidate enumeration, physics features, association
  materials/       the material system (database, taxonomy, features,
                   segmentation, pipeline)
  models/          base.py (JointHead contract + registry), prism.py, mlp.py,
                   losses.py, clutter_process.py
  calibration/     feature scaling, soft labels, Platt, RCS quantile map
  generate/        generate_radar_frame() — the pipeline
  inference.py     HeadScorer — the single inference path
  validation/      distributional / null-control / structural criteria
configs/           datasets/*.yaml (what a dataset IS + what was FITTED on it)
                   materials/*.yaml (label space → material mixture)
weights/vod/       one checkpoint + two calibrations per architecture, clutter
scripts/           generate.py · fit_dataset.py · train_heads.py ·
                   run_all_experiments.sh
tools/             evaluation, visualization, distribution charts, reports
downstream/        detector-training grid on generated radar
output/<arch>/     results, never mixed across architectures
```

## Results

**[`output/README.md`](output/README.md)** is the index. Each architecture has
its own quantitative, qualitative, downstream and chart reports under
`output/<arch>/`; that index is the only page that compares them.

`output/<arch>/quantitative/distribution/` holds the distributional charts —
RCS marginal / CDF / Q–Q, RCS by range, joint and BEV densities with difference
maps, per-frame and per-object counts, object-wise class contrast, measured vs
predicted RCS, and the fitting→deployment covariate shift. Regenerate with
`python tools/plot_distributions.py --head <arch>`.

The generator reproduces a substantial share of real radar's distributional and
structural behaviour, and that does **not** by itself translate into downstream
detector transfer. Both halves are measured, and the reports say so where the
numbers are.

## Reproducing

```bash
bash scripts/run_all_experiments.sh        # fit + evaluate both architectures
pytest tests/ -v                            # regression tests
```

Stages are individually selectable (`STAGES="train eval report" bash …`) so a
long run can resume part-way.

## Extending

- **A new dataset** — a `DatasetAdapter` subclass, a
  `configs/datasets/<name>.yaml`, and a `configs/materials/<space>.yaml` if its
  taxonomy is new. Nothing under `rear/` changes; `tests/` enforces that no
  dataset-specific constant lives there.
- **A new architecture** — a `JointHead` subclass plus one line in
  `rear/models/base.py`'s registry. Training, calibration, generation,
  evaluation and the downstream builders address heads through the interface,
  never by name.

Both paths are exercised in `tests/test_extensibility.py`.

## Data and licensing

Dataset roots are configured, not hardcoded — set `REAR_KC_ROOT` (KITTI-CARLA)
and `REAR_VOD_ROOT` (View-of-Delft), or edit `configs/datasets/*.yaml`. No raw
dataset is included in this repository.

The fitted weights in `weights/vod/` were learned from View-of-Delft data —
**check VoD's license terms before redistributing them.** The material database
is derived from a public ITU Recommendation. A profile you fit on your own data
carries whatever license your data does.
