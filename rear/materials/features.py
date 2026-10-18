"""Per-candidate material features from per-point material mixtures.

A candidate is a voxel holding some number of LiDAR points. Each point carries a
material mixture (from a semantic label, via taxonomy.py). This module reduces
those to the fixed-width block that is concatenated onto the 8 physics features.

TWO STAGES, AND THE SPLIT IS LOAD-BEARING
-----------------------------------------
    aggregate_composition()      points -> per-candidate MATERIAL COMPOSITION
    features_from_composition()  composition -> the 10-column feature block

**Composition is a sufficient statistic for the whole block**, and that is not
an approximation. The power-average terms are linear in the mixture -- the mean
over a voxel's points of `mix @ gamma^2` is exactly `(mean mix) @ gamma^2` --
and the remaining terms (entropy, dominant scatterer, coverage) depend only on
the composition vector.

That buys three things. The expensive per-point pass runs once and caches 11
floats per candidate. Every ablation over the constants -- including the C2
control, which permutes which material gets which constants -- becomes a matrix
multiply on cached data instead of a re-extraction. And the physics (the
database) is cleanly separated from the geometry (voxel assignment), so an error
in one cannot masquerade as the other.

THE AGGREGATION IS THE PART THAT MATTERS
----------------------------------------
The obvious reduction -- average the material vectors in the voxel -- is the
wrong physics, and getting this right is most of what distinguishes a material
feature from a re-encoded class label.

RCS adds in POWER, not amplitude, and it is dominated by the strongest scatterer
in the resolution cell rather than the typical one. A voxel that is 90% asphalt
and 10% guard-rail post returns like a guard-rail post, not like a slightly
shiny road. So the block carries BOTH

    gamma_pow_mean   sqrt(comp @ |Gamma|^2)               the power average
    gamma_max        max |Gamma| among materials PRESENT  dominant scatterer

and lets the head learn which regime each candidate is in.

HETEROGENEITY IS A FEATURE, NOT NOISE
-------------------------------------
A voxel spanning two materials contains a material BOUNDARY, and boundaries are
strong, angle-insensitive scatterers -- this is why a corner reflector works and
a smooth flat plate is invisible off-boresight. `material_entropy` measures that
mixing directly, and it is a quantity a per-box class label cannot express at
all: inside a box, class is constant by construction.

VALIDITY TRAVELS WITH THE FEATURES
----------------------------------
`frac_low_confidence` reports how much of a voxel's material mass came from
constants flagged low-confidence in the database (vegetation, human tissue,
rubber, plastic, soil), so "does the result survive dropping every uncertain
constant?" is a filter over an existing column.
"""

from __future__ import annotations

import numpy as np

from rear.materials.database import (DESCRIPTOR_NAMES, MATERIAL_DB,
                                     MaterialDatabase)

# Columns emitted, in order. A module constant because -- exactly like the 8
# physics features -- this is a contract that checkpoints, scalers and swap
# tests all depend on, and it must be readable in one place.
MATERIAL_FEATURE_NAMES = [
    "gamma_pow_mean",       # sqrt(comp @ |Gamma|^2)  power-average reflectivity
    "gamma_max",            # max |Gamma| among present materials
    "specular_pow_mean",    # sqrt(comp @ |Gamma|^2 rho_s)  what returns AT you
    "eps_real_mean",        # composition-averaged eps_r'
    "log10_sigma_mean",     # composition-averaged log10 conductivity
    "metal_fraction",       # the single strongest driver, kept explicit
    "material_entropy",     # Shannon entropy of the voxel's material mixture
    "n_materials_eff",      # exp(entropy): effective number of materials
    "frac_labelled",        # fraction of voxel points carrying any material
    "frac_low_confidence",  # material mass from low-confidence constants
]
N_MATERIAL_FEATURES = len(MATERIAL_FEATURE_NAMES)

# Composition mass below this is numerical dust, not a real minority component:
# the smallest deliberate fraction anywhere in the shipped taxonomy is 0.05.
PRESENCE_EPS = 1e-6


def aggregate_composition(point_mix: np.ndarray, voxel_index: np.ndarray,
                          n_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    """points -> (composition (M, n_materials), frac_labelled (M,)).

    `voxel_index[i]` is the candidate each point belongs to, or -1 for a point
    in no candidate voxel. `composition` is the mean material mixture over the
    voxel's LABELLED points and sums to 1 where any exist, 0 otherwise.

    Normalising by the labelled count rather than the total is what makes the
    composition a sufficient statistic: an unlabelled point contributes no
    reflectivity evidence, and averaging it in as zeros would silently darken
    every partially-labelled voxel. How much of the voxel was labelled at all is
    reported separately, in `frac_labelled`, where a model can use it.
    """
    vi = np.asarray(voxel_index, dtype=np.int64)
    keep = vi >= 0
    vi, mix = vi[keep], np.asarray(point_mix, dtype=np.float64)[keep]
    n_mat = point_mix.shape[1]

    comp = np.zeros((n_candidates, n_mat), dtype=np.float64)
    frac = np.zeros(n_candidates, dtype=np.float64)
    if len(vi) == 0:
        return comp, frac

    mass = mix.sum(axis=1)                    # 1.0 for a labelled point, else 0
    n_pts = np.bincount(vi, minlength=n_candidates).astype(np.float64)
    n_lab = np.bincount(vi, weights=mass, minlength=n_candidates)
    for j in range(n_mat):
        comp[:, j] = np.bincount(vi, weights=mix[:, j], minlength=n_candidates)
    comp /= np.maximum(n_lab, 1e-12)[:, None]
    frac = n_lab / np.maximum(n_pts, 1e-12)
    return comp, frac


def features_from_composition(comp: np.ndarray, frac_labelled: np.ndarray,
                              db: MaterialDatabase = MATERIAL_DB,
                              freq_ghz: float | None = None,
                              descriptors: np.ndarray | None = None
                              ) -> np.ndarray:
    """(M, N_MATERIAL_FEATURES) from a cached composition.

    `descriptors` overrides the physical table -- this is how the C2 control
    permutes which material carries which constants while leaving the
    representation's structure, dimensionality and marginals untouched
    (`MaterialDatabase.permuted_descriptors`).
    """
    # D is (n_material, n_descriptor); its COLUMNS are DESCRIPTOR_NAMES, not
    # material names. Indexed by name so adding a descriptor cannot silently
    # shift the physics onto the wrong column.
    D = db.descriptor_table(freq_ghz) if descriptors is None else np.asarray(descriptors)
    eps = D[:, DESCRIPTOR_NAMES.index("eps_real")]
    lsig = D[:, DESCRIPTOR_NAMES.index("log10_sigma")]
    gamma = D[:, DESCRIPTOR_NAMES.index("gamma_mag")]
    spec = D[:, DESCRIPTOR_NAMES.index("specular_fraction")]

    comp = np.asarray(comp, dtype=np.float64)
    if comp.shape[1] != len(D):
        raise ValueError(
            f"composition has {comp.shape[1]} material columns but the "
            f"descriptor table has {len(D)}. The cache and the database "
            f"disagree -- rebuild the cache rather than truncating.")
    out = np.zeros((len(comp), N_MATERIAL_FEATURES), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(comp > 0, comp * np.log(np.maximum(comp, 1e-300)), 0.0),
                      axis=1)
    ent = np.maximum(ent, 0.0)          # kill -0.0 on the pure-material case

    # Dominant scatterer: the largest |Gamma| among materials actually PRESENT,
    # not a max over per-point amplitude averages. The 90% asphalt + 10%
    # guard-rail voxel must report the guard rail (0.999); a max over points
    # would report the composition-averaged 0.44, which can even fall BELOW
    # gamma_pow_mean since the power average weights metal by |Gamma|^2.
    gmax = np.max(np.where(comp > PRESENCE_EPS, gamma[None, :], 0.0), axis=1)

    out[:, 0] = np.sqrt(comp @ (gamma ** 2))
    out[:, 1] = gmax
    out[:, 2] = np.sqrt(comp @ (gamma ** 2 * spec))
    out[:, 3] = comp @ eps
    out[:, 4] = comp @ lsig
    out[:, 5] = comp[:, db.index["metal"]]
    out[:, 6] = ent
    out[:, 7] = np.exp(ent)
    out[:, 8] = np.asarray(frac_labelled, dtype=np.float64)
    out[:, 9] = comp @ db.low_confidence_mask

    # A candidate with no labelled point carries no material evidence at all.
    # Everything zeroes, including n_materials_eff, which exp(0) = 1 would
    # otherwise make look like "one pure material".
    empty = comp.sum(axis=1) <= PRESENCE_EPS
    out[empty] = 0.0
    return out


def aggregate_to_candidates(point_mix: np.ndarray, voxel_index: np.ndarray,
                            n_candidates: int,
                            db: MaterialDatabase = MATERIAL_DB,
                            freq_ghz: float | None = None) -> np.ndarray:
    """Convenience: points straight to the feature block, no cache in between."""
    comp, frac = aggregate_composition(point_mix, voxel_index, n_candidates)
    return features_from_composition(comp, frac, db=db, freq_ghz=freq_ghz)
