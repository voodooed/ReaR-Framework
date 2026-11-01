"""The two extension points, exercised rather than asserted.

The framework claims a new dataset needs no code change and a new architecture
needs one registry line. Both claims are easy to write in a README and easy to
break silently — a stray import, a hardcoded vocabulary, a head addressed by
name somewhere in the pipeline — so they are tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rear.materials import MATERIAL_DB, load_label_space  # noqa: E402
from rear.materials.features import features_from_composition  # noqa: E402
from rear.materials.taxonomy import LABEL_SPACE_DIR, LabelSpace  # noqa: E402
from rear.models.base import (JointHead, available_architectures,  # noqa: E402
                              build_head, load_head, save_head)

import yaml  # noqa: E402


# ---------------------------------------------------------------- datasets ---
def test_a_new_taxonomy_is_a_file_and_needs_no_code_change(tmp_path):
    """A dataset with a vocabulary nobody anticipated — different ids, different
    names, a class the shipped spaces do not have — must map into the SAME
    material space and produce the SAME fixed-width block."""
    doc = {
        "name": "invented_sensor_space",
        "description": "a taxonomy this repository has never seen",
        "n_classes": 4,
        "classes": {
            0: {"name": "tarmac", "mixture": {"asphalt": 1.0}},
            1: {"name": "shipping_container", "mixture": {"metal": 1.0}},
            2: {"name": "hedge", "mixture": {"vegetation": 0.7, "soil": 0.3}},
            3: {"name": "void", "mixture": {}},
        },
    }
    p = tmp_path / "invented.yaml"
    p.write_text(yaml.safe_dump(doc))

    ls = LabelSpace(yaml.safe_load(p.read_text()), path=p)
    assert ls.n_materials == len(MATERIAL_DB)

    labels = np.array([0, 1, 2, 3, -1, 99])
    mix = ls.point_mixtures(labels)
    assert mix.shape == (6, len(MATERIAL_DB))
    assert mix[1, MATERIAL_DB.index["metal"]] == 1.0
    assert mix[3].sum() == 0.0        # void returns nothing
    assert mix[4].sum() == 0.0        # unobserved
    assert mix[5].sum() == 0.0        # out of vocabulary

    # And it lands in the same 10-column block the shipped heads consume.
    block = features_from_composition(ls.mixture_matrix(), np.ones(ls.n_classes))
    assert block.shape[1] == 10


def test_an_unseen_class_lands_at_a_sensible_position(tmp_path):
    """The property that makes cross-dataset transfer expressible: a class the
    fitting dataset never had is not an unseen one-hot column, it is a point in
    a space the head already understands. `GuardRail` is the real instance —
    CARLA has it, Cityscapes does not."""
    carla, city = load_label_space("carla"), load_label_space("cityscapes")
    guard = carla.mixture_matrix()[17:18]           # GuardRail -> metal 1.0
    pole = city.mixture_matrix()[5:6]                # pole -> metal 0.9, plastic 0.1
    gb = features_from_composition(guard, np.ones(1))
    pb = features_from_composition(pole, np.ones(1))
    assert gb.shape == pb.shape                      # same fixed contract
    # A metal guard rail must read as MORE reflective than a mostly-metal pole,
    # not as an arbitrary unseen coordinate.
    i = 0                                            # gamma_pow_mean
    assert gb[0, i] > pb[0, i]


def test_shipped_label_spaces_all_load_from_disk():
    """Nothing about the shipped taxonomies is compiled in."""
    files = sorted(LABEL_SPACE_DIR.glob("*.yaml"))
    assert files, "configs/materials/ has no label spaces"
    for f in files:
        ls = load_label_space(f.stem)
        assert ls.n_classes > 0 and ls.n_materials == len(MATERIAL_DB)


# ----------------------------------------------------------- architectures ---
class _ToyHead(JointHead):
    """A third architecture, defined entirely outside `rear/`."""

    architecture = "toy"

    def __init__(self, n_features=8, n_material=0, width=7):
        super().__init__(n_features, n_material)
        self.arch_config = dict(width=width)
        self.body = torch.nn.Linear(self.n_inputs, width)
        self.out = torch.nn.Linear(width, 2)

    def forward(self, x):
        self.check_input(x)
        h = torch.relu(self.body(x))
        o = self.out(h)
        return {"det_logit": o[:, 0], "rcs_mean": o[:, 1]}


def test_a_new_architecture_is_one_registry_entry(monkeypatch, tmp_path):
    """Register a head the framework has never seen, then drive it through the
    same construct → save → load path the shipped ones use. If anything in that
    path addressed a head by NAME rather than through the interface, this fails.
    """
    import rear.models.base as base

    real_registry = base._registry

    def patched():
        r = dict(real_registry())
        r["toy"] = _ToyHead
        return r

    monkeypatch.setattr(base, "_registry", patched)

    assert "toy" in base.available_architectures()
    model = base.build_head("toy", n_features=8, n_material=10)
    x = torch.randn(16, 18)
    out = model(x)
    assert out["det_logit"].shape == (16,) and out["rcs_mean"].shape == (16,)
    # The interface's provided behaviour comes for free.
    assert model.predict_detection_proba(x).min() >= 0.0
    assert model.rcs_variance(x) is None          # point estimate: honest None

    ck_path = tmp_path / "toy_head.pt"
    base.save_head(ck_path, model, scaler={"mu": np.zeros(8), "sigma": np.ones(8),
                                           "clip": 10.0},
                   feature_names=[f"f{i}" for i in range(8)],
                   feature_variant="drop", y_mu=0.0, y_sigma=1.0,
                   material={"enabled": True, "n_features": 10})
    restored, ck = base.load_head(ck_path)
    assert ck["architecture"] == "toy"
    assert restored.n_inputs == 18
    with torch.no_grad():
        assert torch.allclose(restored(x)["det_logit"], out["det_logit"])


def test_input_width_mismatch_is_caught_not_broadcast():
    """A head trained with material must refuse an 8-column matrix. Torch would
    otherwise raise somewhere deep in a Linear, or worse, broadcast."""
    m = build_head("prism", n_features=8, n_material=10)
    with pytest.raises(ValueError, match="material"):
        m(torch.randn(4, 8))


def test_unknown_architecture_names_the_alternatives():
    with pytest.raises(ValueError, match="available"):
        build_head("does_not_exist")


def test_registry_and_checkpoints_agree_on_the_shipped_heads():
    for arch in available_architectures():
        m = build_head(arch, n_features=8, n_material=10)
        assert m.architecture == arch
        assert m.n_inputs == 18
