"""Package sanity: the shipped weights + configs actually generate radar.

Uses a SYNTHETIC LiDAR point cloud, not a real dataset download, so this runs
anywhere `rear/` + `weights/` are present -- exactly the condition a fresh clone
of this package should satisfy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rear.data.base import load_adapter_config  # noqa: E402
from rear.generate.generate_radar_frame import generate_radar_frame  # noqa: E402
from rear.models.base import available_architectures, load_head  # noqa: E402

# Weights are namespaced by which dataset they were fitted on (see
# rear/data/base.py's fitted_from chain) -- VoD is the only dataset shipped
# fitted, hence weights/vod/.
WEIGHTS = REPO_ROOT / "weights" / "vod"
ARCHITECTURES = ["mlp", "prism"]

# One joint checkpoint per architecture, plus BOTH calibrations per
# architecture. The calibrations are not shared: each is a function of one
# head's own output distribution, and both are monotone, so a mismatched pair
# corrupts the generated cloud without moving any head-level metric.
REQUIRED_WEIGHTS = ["clutter_process.pkl"] + [
    f"{a}{suffix}" for a in ARCHITECTURES
    for suffix in ("_head.pt",)] + [
    f"{stem}_{a}.pkl" for a in ARCHITECTURES
    for stem in ("detection_calibration", "rcs_calibration")]


@pytest.mark.parametrize("name", REQUIRED_WEIGHTS)
def test_required_weight_present(name):
    assert (WEIGHTS / name).exists(), f"missing weights/vod/{name}"


def _synthetic_lidar(n=20000, seed=0):
    """A ground plane + a wall segment, roughly in front of the sensor, with a
    plausible incidence-cosine 4th column (KITTI-CARLA convention)."""
    rng = np.random.default_rng(seed)
    ground = np.column_stack([
        rng.uniform(1, 60, n), rng.uniform(-20, 20, n), rng.normal(-1.6, 0.05, n),
        rng.uniform(0.6, 1.0, n),
    ])
    wall = np.column_stack([
        np.full(500, 15.0) + rng.normal(0, 0.1, 500),
        rng.uniform(-2, 2, 500), rng.uniform(-1.5, 1.0, 500),
        rng.uniform(0.5, 0.95, 500),
    ])
    return np.vstack([ground, wall]).astype(np.float64)


def _synthetic_labels(lidar, label_space="carla"):
    """Per-point CARLA tags for the synthetic scene: road under the sensor,
    building for the wall. Enough to exercise the material path end to end."""
    lab = np.full(len(lidar), 7, dtype=np.int64)          # Road
    lab[lidar[:, 2] > -1.4] = 1                            # Building
    return lab


def test_config_inheritance_resolves_vod_donor_values():
    """kitti_carla.yaml has no rcs_calibration block of its own; it must
    resolve through `fitted_from: vod` to configs/datasets/vod.yaml."""
    k = load_adapter_config("kitti_carla")
    assert k.fitted("rcs_calibration", "enabled") is True
    assert k.fitted("heads", "implementation") == "prism"


def test_material_config_is_enabled_on_both_datasets():
    """Material is a permanent component, so both shipped datasets declare it,
    and each names the label space its own semantics live in."""
    v = load_adapter_config("vod").material_config()
    k = load_adapter_config("kitti_carla").material_config()
    assert v["enabled"] and k["enabled"]
    assert v["label_space"] == "cityscapes"
    assert k["label_space"] == "carla"
    # The two sources are the asymmetry the sim2real problem is made of.
    assert v["source"] == "camera_segmentation"
    assert k["source"] == "lidar_semantic"


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_generate_radar_frame_end_to_end(architecture):
    lidar = _synthetic_lidar()
    ego_velocity = np.array([8.0, 0.0, 0.0])
    cfg = load_adapter_config("kitti_carla").grid_config()

    g = generate_radar_frame(lidar, ego_velocity, seed=0, config=cfg,
                             architecture=architecture,
                             point_labels=_synthetic_labels(lidar),
                             label_space="carla")

    assert g.points.shape[1] == 5           # x, y, z, RCS, v_r
    assert g.points.shape[0] == g.is_clutter.shape[0]
    assert g.points.shape[0] > 0            # a wall + ground should yield SOME returns
    assert np.isfinite(g.points).all()

    # Reproducibility: same seed -> byte-identical output.
    g2 = generate_radar_frame(lidar, ego_velocity, seed=0, config=cfg,
                              architecture=architecture,
                              point_labels=_synthetic_labels(lidar),
                              label_space="carla")
    assert np.array_equal(g.points, g2.points)
    assert np.array_equal(g.is_clutter, g2.is_clutter)


def test_missing_material_is_an_error_not_a_silent_degradation():
    """A head trained with a material branch must refuse to run on a frame
    with no labels unless the caller opts in. Silently substituting the
    all-zero block would look like working generation while throwing away the
    block's entire contribution."""
    lidar = _synthetic_lidar()
    cfg = load_adapter_config("kitti_carla").grid_config()
    with pytest.raises(ValueError, match="material"):
        generate_radar_frame(lidar, np.array([8.0, 0.0, 0.0]), seed=0,
                             config=cfg, architecture="prism")
    # ...and the opt-out works, so the degraded path stays reachable.
    g = generate_radar_frame(lidar, np.array([8.0, 0.0, 0.0]), seed=0,
                             config=cfg, architecture="prism",
                             allow_missing_material=True)
    assert g.points.shape[1] == 5


def test_material_changes_the_generated_cloud():
    """If the material block did not reach the head, every arm of this project
    that measured it would be measuring nothing."""
    lidar = _synthetic_lidar()
    ego = np.array([8.0, 0.0, 0.0])
    cfg = load_adapter_config("kitti_carla").grid_config()
    with_mat = generate_radar_frame(lidar, ego, seed=0, config=cfg,
                                    architecture="prism",
                                    point_labels=_synthetic_labels(lidar),
                                    label_space="carla")
    without = generate_radar_frame(lidar, ego, seed=0, config=cfg,
                                   architecture="prism",
                                   allow_missing_material=True)
    assert not np.array_equal(with_mat.points, without.points)


def test_architectures_are_selectable_and_differ():
    """Either architecture can be requested, and they are genuinely different
    models -- otherwise the switch would be decorative."""
    lidar = _synthetic_lidar()
    ego = np.array([8.0, 0.0, 0.0])
    cfg = load_adapter_config("kitti_carla").grid_config()
    labels = _synthetic_labels(lidar)
    out = {a: generate_radar_frame(lidar, ego, seed=0, config=cfg,
                                   architecture=a, point_labels=labels,
                                   label_space="carla")
           for a in ARCHITECTURES}
    for a in ARCHITECTURES:
        assert len(out[a].points) > 0
    assert not np.array_equal(out["mlp"].points, out["prism"].points)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_checkpoint_is_self_describing(architecture):
    """Everything needed to reproduce a head's own preprocessing travels inside
    its file. Inference never recomputes any of it, because train/serve skew
    here is silent -- a mis-scaled feature yields a plausible-looking cloud."""
    import torch  # noqa: F401

    model, ck = load_head(WEIGHTS / f"{architecture}_head.pt")
    assert ck["architecture"] == architecture
    for key in ("mu", "sigma", "clip", "feature_names", "feature_variant",
                "y_mu", "y_sigma", "material", "n_features", "n_material"):
        assert key in ck, f"{architecture} checkpoint is missing {key!r}"
    assert len(ck["mu"]) == ck["n_features"] == len(ck["feature_names"])
    assert ck["material"]["n_features"] == ck["n_material"]
    assert model.n_inputs == ck["n_features"] + ck["n_material"]


def test_prism_mixture_has_not_collapsed():
    """With the mixture read-out's weight AND bias zero-initialised, all K
    components are numerically identical, receive identical gradients forever,
    and the mixture degenerates to a single Gaussian with NO error raised. An
    entire ablation ladder was once run with that bug present, so the shipped
    model is checked for it rather than assumed free of it."""
    import torch

    model, ck = load_head(WEIGHTS / "prism_head.pt")
    x = torch.randn(4096, model.n_inputs)
    with torch.no_grad():
        out = model(x)
    pi = out["rcs_pi"]
    assert pi.shape[1] > 1
    # Uniform weights for EVERY candidate is the signature of the degeneracy.
    assert not np.allclose(pi.mean(0).numpy(), 1.0 / pi.shape[1], atol=1e-3)
    assert torch.allclose(out["rcs_mean"], (pi * out["rcs_mu"]).sum(-1))
    assert model.rcs_variance(x).min() > 0


def test_registry_lists_exactly_the_shipped_architectures():
    assert sorted(available_architectures()) == sorted(ARCHITECTURES)
