"""The material/permittivity database and the one interface for reading it.

    from rear.materials import MATERIAL_DB
    MATERIAL_DB.names                       -> the 11 materials, in column order
    MATERIAL_DB.constants("concrete")       -> MaterialConstants at 77 GHz
    MATERIAL_DB.descriptor_table()          -> (11, 5) EM descriptor matrix

Two files back it, both inside this repository so the framework is
self-contained and both readable as data:

    data/itu_p2040_4.yaml   Recommendation ITU-R P.2040-4 Table 3, verbatim
    data/materials.yaml     the scene palette: which ITU entry (or which
                            locally declared constants) each material uses,
                            plus roughness, confidence and provenance

Nothing else in `rear/` contains an electromagnetic constant. Adding or
changing a material is an edit to `materials.yaml`; adding a new ITU entry is
an edit to `itu_p2040_4.yaml`. Neither requires touching code.

COLUMN ORDER IS A CONTRACT. `materials.yaml` declares `order:` explicitly and
this module checks it against the mapping, because cached compositions and
trained checkpoints index materials by position: a silent reorder would attach
every voxel's material mass to the wrong constants while leaving every shape
and every summary statistic looking correct.

FREQUENCY. One value, 77.0 GHz, from `materials.yaml`. Measured rather than
assumed: sweeping 76->81 GHz moves concrete's |Gamma| by 0.005%, so a
frequency-dependent representation would be complexity with no content. The
frequency is nonetheless a parameter of every call, so a database built for a
different radar band needs no code change either.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from rear.materials.itu import (MaterialConstants, evaluate, from_eps_sigma,
                                rayleigh_smoothness)

DATA_DIR = Path(__file__).resolve().parent / "data"

# The per-material electromagnetic descriptor. `gamma_mag` and
# `specular_fraction` carry the physics the pipeline actually needs -- how much
# comes back, and whether it comes back at you or scatters away. The other
# three are kept because they are free and let an ablation ask which axis
# matters.
DESCRIPTOR_NAMES = ["eps_real", "log10_sigma", "gamma_mag",
                    "specular_fraction", "log10_loss_tangent"]


@dataclass(frozen=True)
class Material:
    """One palette entry: how to resolve its constants, plus its provenance."""

    name: str
    itu_name: str | None          # ITU database entry, or None if declared locally
    h_rms_m: float                # RMS surface roughness
    confidence: str               # high | medium | low
    source: str
    eps_real_override: float | None = None
    sigma_override: float | None = None


class MaterialDatabase:
    """Loaded once, queried everywhere. Immutable after construction."""

    def __init__(self, materials_path: Path | None = None,
                 itu_path: Path | None = None):
        self.materials_path = Path(materials_path or DATA_DIR / "materials.yaml")
        self.itu_path = Path(itu_path or DATA_DIR / "itu_p2040_4.yaml")

        itu_doc = yaml.safe_load(self.itu_path.read_text())
        self.itu_version = itu_doc.get("version", "")
        self.itu_source = itu_doc.get("source", "")
        # {name: {(f_lo, f_hi): (a, b, c, d)}}
        self._itu: dict[str, dict[tuple[float, float], tuple]] = {
            name: {(float(r[0]), float(r[1])): tuple(float(v) for v in r[2:])
                   for r in rows}
            for name, rows in itu_doc["entries"].items()}

        doc = yaml.safe_load(self.materials_path.read_text())
        self.default_frequency_ghz = float(doc["default_frequency_ghz"])
        entries = doc["materials"]
        order = list(doc["order"])
        if set(order) != set(entries):
            raise ValueError(
                f"{self.materials_path.name}: `order` and `materials` disagree "
                f"-- only in order: {sorted(set(order) - set(entries))}, only in "
                f"materials: {sorted(set(entries) - set(order))}. The order list "
                f"is the column contract and must name every material exactly once.")
        self.names: list[str] = order

        self.materials: dict[str, Material] = {}
        for name in order:
            e = entries[name]
            itu_name = e.get("itu")
            if itu_name is not None and itu_name not in self._itu:
                raise KeyError(
                    f"material {name!r} references ITU entry {itu_name!r}, which "
                    f"is not in {self.itu_path.name}. Materials outside the "
                    f"Recommendation must declare eps_real/sigma locally with an "
                    f"explicit source rather than being added to the ITU table.")
            if itu_name is None and ("eps_real" not in e or "sigma" not in e):
                raise ValueError(
                    f"material {name!r} declares no `itu:` entry and no local "
                    f"eps_real/sigma -- it has no constants at all.")
            self.materials[name] = Material(
                name=name, itu_name=itu_name, h_rms_m=float(e["h_rms_m"]),
                confidence=str(e["confidence"]), source=str(e.get("source", "")),
                eps_real_override=(None if itu_name else float(e["eps_real"])),
                sigma_override=(None if itu_name else float(e["sigma"])))

        self.index = {n: i for i, n in enumerate(self.names)}
        self.low_confidence_mask = np.array(
            [self.materials[n].confidence == "low" for n in self.names], dtype=np.float64)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.materials

    def constants(self, name: str, freq_ghz: float | None = None) -> MaterialConstants:
        """EM constants of one material, with band validity attached."""
        f = float(freq_ghz if freq_ghz is not None else self.default_frequency_ghz)
        m = self.materials[name]
        if m.itu_name is not None:
            return evaluate(m.itu_name, f, self._itu[m.itu_name], source=m.source)
        # Locally declared: built through the same constructor so downstream code
        # cannot tell the difference. in_band is False by construction -- these
        # materials have no Recommendation band to be inside.
        return from_eps_sigma(m.name, f, m.eps_real_override, m.sigma_override,
                              in_band=False, band=(f, f), source=m.source)

    def specular_fraction(self, name: str, freq_ghz: float | None = None) -> float:
        f = float(freq_ghz if freq_ghz is not None else self.default_frequency_ghz)
        return rayleigh_smoothness(self.materials[name].h_rms_m, f)

    def descriptor_table(self, freq_ghz: float | None = None) -> np.ndarray:
        """(n_material, 5) descriptor matrix, rows in `self.names` order.

        Columns are DESCRIPTOR_NAMES:
          0 eps_real          real relative permittivity
          1 log10_sigma       conductivity in log10 S/m -- the raw value spans
                              1e-2 to 1e7 and would swamp every other column
          2 gamma_mag         |Gamma_0|, normal-incidence Fresnel magnitude
          3 specular_fraction Ament factor: how much reflected power returns
                              specularly rather than scattering away
          4 log10_loss_tangent the absorption axis
        """
        f = float(freq_ghz if freq_ghz is not None else self.default_frequency_ghz)
        return _descriptor_table(self, f)

    def summary_rows(self, freq_ghz: float | None = None) -> list[dict]:
        """One inspectable row per material -- what `tools/inspect_materials.py`
        prints and what a reviewer checks against the Recommendation."""
        f = float(freq_ghz if freq_ghz is not None else self.default_frequency_ghz)
        out = []
        for n in self.names:
            c = self.constants(n, f)
            out.append(dict(material=n, eps_real=c.eps_real, sigma=c.sigma,
                            gamma_mag=c.gamma_mag, reflectivity_db=c.reflectivity_db,
                            h_rms_mm=self.materials[n].h_rms_m * 1e3,
                            specular_fraction=self.specular_fraction(n, f),
                            confidence=self.materials[n].confidence,
                            in_band=c.in_band, band=c.band, source=c.source))
        return out

    def permuted_descriptors(self, seed: int = 0,
                             freq_ghz: float | None = None) -> np.ndarray:
        """The C2 control's table: every material keeps a REAL material's
        constants, but not its own.

        This is the sharpest control available on the material feature.
        Dimensionality, marginals, the per-voxel mixing structure and the
        range/aspect composition of the scene all survive untouched; the only
        thing destroyed is the CORRESPONDENCE between a material and its
        electromagnetic properties. An arm that survives C2 is not using
        physics, it is using the semantic partition -- and on this project's
        measurement, that is exactly what happens (w-R2 0.4458 for C2 against
        0.4470 for the real table, on a 0.0087 seed floor). A derangement is
        used, so no material keeps its own row by accident.
        """
        D = self.descriptor_table(freq_ghz)
        rng = np.random.default_rng(seed)
        n = len(D)
        while True:
            p = rng.permutation(n)
            if not np.any(p == np.arange(n)):     # derangement
                return D[p]

    def __repr__(self):
        return (f"MaterialDatabase({len(self)} materials @ "
                f"{self.default_frequency_ghz:g} GHz, {self.itu_version})")


@lru_cache(maxsize=8)
def _descriptor_table(db: MaterialDatabase, freq_ghz: float) -> np.ndarray:
    rows = []
    for name in db.names:
        c = db.constants(name, freq_ghz)
        rows.append([c.eps_real,
                     np.log10(max(c.sigma, 1e-6)),
                     c.gamma_mag,
                     db.specular_fraction(name, freq_ghz),
                     np.log10(max(c.loss_tangent, 1e-6))])
    out = np.asarray(rows, dtype=np.float64)
    out.flags.writeable = False        # cached: hand out a read-only view
    return out


@lru_cache(maxsize=4)
def load_database(materials_path: str | None = None,
                  itu_path: str | None = None) -> MaterialDatabase:
    """The database, loaded once per path pair."""
    return MaterialDatabase(materials_path, itu_path)


#: The shipped database. Import this rather than constructing your own.
MATERIAL_DB = load_database()
MATERIAL_NAMES = MATERIAL_DB.names
DEFAULT_RADAR_FREQ_GHZ = MATERIAL_DB.default_frequency_ghz
