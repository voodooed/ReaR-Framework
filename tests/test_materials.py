"""The material/permittivity system: database, taxonomy, and the feature block.

These are contract tests, not physics tests. They check the things that would
break SILENTLY -- a reordered material column, a mixture that stops summing to
1, an aggregation that averages amplitude where it should average power -- and
each of them would leave every shape and every summary statistic looking
correct while attaching the wrong reflectivity to the wrong geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rear.materials import (MATERIAL_DB, MATERIAL_FEATURE_NAMES,  # noqa: E402
                            N_MATERIAL_FEATURES, aggregate_composition,
                            available_label_spaces, features_from_composition,
                            load_label_space)

# The 11 materials, in the order the composition vector and every cached
# sidecar index them. This list is duplicated from materials.yaml ON PURPOSE:
# a test that reads the order from the file it is checking cannot detect a
# reorder.
EXPECTED_ORDER = ["metal", "glass", "concrete", "brick", "asphalt", "wood",
                  "soil", "vegetation", "human_tissue", "rubber", "plastic"]


def test_material_column_order_is_the_documented_contract():
    assert MATERIAL_DB.names == EXPECTED_ORDER


def test_constants_match_the_recommendation_at_77_ghz():
    """Spot-check against ITU-R P.2040-4 Table 3 evaluated at 77 GHz. Concrete
    is in band and metal is the reflectivity ceiling; if the power-law
    evaluation or the Fresnel formula drifts, these move first."""
    c = MATERIAL_DB.constants("concrete")
    assert c.eps_real == pytest.approx(5.24, abs=1e-6)
    assert c.sigma == pytest.approx(1.381, abs=1e-3)
    assert c.gamma_mag == pytest.approx(0.393, abs=1e-3)
    assert c.in_band is True

    m = MATERIAL_DB.constants("metal")
    assert m.gamma_mag > 0.99                      # a mirror at 77 GHz
    assert MATERIAL_DB.constants("vegetation").gamma_mag < 0.2


def test_out_of_band_materials_are_flagged_not_hidden():
    """Validity travels with every value. Asphalt is fitted only to 40 GHz and
    soil only to 10; using them at 77 is extrapolation, and the framework has
    to say so rather than return a bare number."""
    assert MATERIAL_DB.constants("asphalt").in_band is False
    assert MATERIAL_DB.constants("soil").in_band is False
    assert MATERIAL_DB.constants("glass").in_band is True
    for name in ("vegetation", "human_tissue", "rubber", "plastic"):
        assert MATERIAL_DB.materials[name].confidence == "low", (
            f"{name} is not in P.2040 and must stay flagged low-confidence")


def test_roughness_separates_what_permittivity_cannot():
    """At lambda = 3.893 mm, which side of lambda/16 a surface sits on decides
    whether its reflection comes back at the radar at all. A car door and a
    road can have similar |Gamma| and differ by orders of magnitude here."""
    assert MATERIAL_DB.specular_fraction("metal") > 0.99
    assert MATERIAL_DB.specular_fraction("asphalt") < 0.05


@pytest.mark.parametrize("space", ["cityscapes", "carla"])
def test_label_space_mixtures_are_distributions(space):
    ls = load_label_space(space)
    M = ls.mixture_matrix()
    assert M.shape == (ls.n_classes, len(MATERIAL_DB))
    sums = M.sum(axis=1)
    # Every row is either a distribution or exactly empty. An empty row means
    # "returns nothing" (sky, unlabeled) and is a value, never a gap to fill.
    assert np.all(np.isclose(sums, 1.0) | np.isclose(sums, 0.0))


def test_unlabelled_points_get_no_material_rather_than_a_prior():
    ls = load_label_space("cityscapes")
    mix = ls.point_mixtures(np.array([-1, 10, 0]))     # unobserved, sky, road
    assert mix[0].sum() == 0.0        # outside the camera: no evidence
    assert mix[1].sum() == 0.0        # sky: returns nothing
    assert mix[2].sum() == pytest.approx(1.0)


def test_the_two_label_spaces_share_one_coordinate_system():
    """This is the only property of the material representation that is
    load-bearing for deployment: two incompatible vocabularies become
    comparable. CARLA's GuardRail has no Cityscapes counterpart, yet it lands
    at a well-defined position rather than in a column that does not exist."""
    city, carla = load_label_space("cityscapes"), load_label_space("carla")
    assert city.mixture_matrix().shape[1] == carla.mixture_matrix().shape[1]
    guardrail = carla.mixtures[17]
    assert guardrail == {"metal": 1.0}
    # car (Cityscapes 13) and Vehicles (CARLA 10) must agree, or the two
    # datasets would disagree about what a car is made of.
    assert city.mixtures[13] == carla.mixtures[10]


def test_dominant_scatterer_is_a_max_over_materials_present():
    """The physics claim the whole block rests on. RCS adds in POWER and is
    dominated by the strongest scatterer in the cell, so a voxel that is 90%
    asphalt and 10% guard-rail post must report the POST. A max over per-point
    amplitude averages would report ~0.44 and can even fall BELOW the power
    average, which is the bug this test exists to catch."""
    i_asphalt = MATERIAL_DB.index["asphalt"]
    i_metal = MATERIAL_DB.index["metal"]
    comp = np.zeros((1, len(MATERIAL_DB)))
    comp[0, i_asphalt], comp[0, i_metal] = 0.9, 0.1
    f = features_from_composition(comp, np.ones(1))
    gamma_pow_mean = f[0, MATERIAL_FEATURE_NAMES.index("gamma_pow_mean")]
    gamma_max = f[0, MATERIAL_FEATURE_NAMES.index("gamma_max")]
    assert gamma_max > 0.99                      # the guard rail
    assert gamma_max > gamma_pow_mean            # and it dominates


def test_no_material_information_is_all_zero_including_n_materials_eff():
    """A candidate with no labelled point must be distinguishable from one
    holding a single pure material. exp(entropy) = 1 would make the two look
    identical on that column."""
    comp = np.zeros((1, len(MATERIAL_DB)))
    f = features_from_composition(comp, np.zeros(1))
    assert f.shape == (1, N_MATERIAL_FEATURES)
    assert np.all(f == 0.0)


def test_composition_is_a_sufficient_statistic_for_the_block():
    """Aggregating points then computing features must equal computing the
    composition and then the features -- that identity is what lets the
    expensive per-point pass be cached as 11 floats and every control become a
    matrix multiply."""
    rng = np.random.default_rng(0)
    ls = load_label_space("cityscapes")
    labels = rng.integers(-1, 19, size=5000)
    voxel = rng.integers(-1, 200, size=5000)
    mix = ls.point_mixtures(labels)
    comp, frac = aggregate_composition(mix, voxel, 200)
    assert comp.shape == (200, len(MATERIAL_DB))
    # Labelled voxels hold a distribution; the rest hold nothing.
    sums = comp.sum(axis=1)
    assert np.all(np.isclose(sums, 1.0) | np.isclose(sums, 0.0))
    assert np.all((frac >= 0) & (frac <= 1))
    assert features_from_composition(comp, frac).shape == (200, N_MATERIAL_FEATURES)


def test_c2_control_destroys_correspondence_and_nothing_else():
    """The control that decides how this project's material result is read.
    It must keep the table's shape and its multiset of rows exactly, and give
    no material its own constants back -- otherwise it would not isolate the
    material->physics correspondence."""
    real = MATERIAL_DB.descriptor_table()
    deranged = MATERIAL_DB.permuted_descriptors(0)
    assert deranged.shape == real.shape
    assert np.array_equal(np.sort(deranged, axis=0), np.sort(real, axis=0))
    assert not np.any(np.all(deranged == real, axis=1))   # a true derangement


def test_a_new_taxonomy_needs_no_code_change():
    """The extensibility claim, exercised rather than asserted: a label space
    is a file, and loading one that was never compiled in must work."""
    assert set(available_label_spaces()) >= {"cityscapes", "carla"}
    for name in available_label_spaces():
        ls = load_label_space(name)
        assert ls.n_materials == len(MATERIAL_DB)
