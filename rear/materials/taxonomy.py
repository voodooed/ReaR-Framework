"""Semantic label space -> material mixture.

A `LabelSpace` is one file under `configs/materials/` (cityscapes.yaml,
carla.yaml, ...) loaded into a dense (n_class, n_material) fraction matrix. It
is the ONLY place a dataset's own vocabulary appears: `rear/` never mentions a
class called "road" or "GuardRail", and a dataset config names its space in one
line.

    Dataset -> segmentation -> semantic label -> LabelSpace -> composition
            -> MaterialDatabase -> permittivity -> the 10-column model input

WHAT THIS ADDS OVER A CLASS LABEL, HONESTLY
-------------------------------------------
A deterministic map `class -> material vector` adds ZERO information over the
one-hot class label, for any model that can fit an arbitrary function of a
categorical input, and this project measured exactly that: the pre-registered
control with the material->constant table DERANGED scores w-R2 0.4458 against
the real table's 0.4470, on a 0.0087 seed floor. **The information is the
semantic partition; the permittivity values are a re-encoding of the category.**
That is a headline result, not a footnote, and nothing here claims otherwise.

Three things nonetheless make the material representation the right engineering
choice, and only these three are claimed:

  1. COVERAGE. Box class is defined only inside annotated boxes -- 5.1% of the
     swappable population. Material is defined on the other 94.9%: road, facade,
     vegetation, poles, guard rails. That population had no reflectivity
     information at all, and it is where the detector's RCS contrast has to come
     from.
  2. SUB-OBJECT GRANULARITY. A box says "Car". A per-pixel segmentation says
     windscreen (glass) here, door (metal) there, wheel arch (rubber) below. A
     box class cannot vary within an object; a mixture can, and the resulting
     material ENTROPY measures material boundaries -- strong, angle-insensitive
     scatterers a box class cannot express at all.
  3. A SHARED COORDINATE SYSTEM ACROSS DATASETS. Cityscapes' 19 classes and
     CARLA's 23 tags cannot be aligned as one-hots. Projected into the same
     material space they can, and a KITTI-CARLA tag never seen on VoD lands at a
     sensible position rather than in an unseen column. This is the only part of
     the representation that is load-bearing for deployment, and it is why the
     EM block -- fixed width, independent of any segmenter's vocabulary -- is the
     production input rather than raw class fractions.

AN EMPTY MIXTURE IS A VALUE, NOT A GAP
--------------------------------------
Classes that map to nothing (sky, unlabeled) get an all-zero row deliberately,
never a uniform prior: "this returns nothing to the radar" is the physically
correct statement for sky and the honest one for an unlabelled point. Inventing
reflectivity where the sensor reports none is the one failure mode that would
be invisible in every summary statistic.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from rear.materials.database import MATERIAL_DB, MaterialDatabase

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_SPACE_DIR = REPO_ROOT / "configs" / "materials"


class LabelSpace:
    """One dataset-facing semantic vocabulary, mapped into material space."""

    def __init__(self, doc: dict, db: MaterialDatabase = MATERIAL_DB,
                 path: Path | None = None):
        self.name = doc["name"]
        self.description = doc.get("description", "")
        self.path = path
        self.db = db

        classes = doc["classes"]
        self.n_classes = int(doc.get("n_classes", max(classes) + 1))
        self.ids = sorted(int(k) for k in classes)
        self.class_names: list[str] = [""] * self.n_classes
        self.mixtures: dict[int, dict[str, float]] = {}

        for cid in self.ids:
            entry = classes[cid]
            if cid >= self.n_classes:
                raise ValueError(
                    f"{self.name}: class id {cid} is outside the declared "
                    f"n_classes={self.n_classes}")
            mix = dict(entry.get("mixture") or {})
            unknown = set(mix) - set(db.names)
            if unknown:
                raise KeyError(
                    f"{self.name}: class {cid} ({entry.get('name')}) references "
                    f"material(s) {sorted(unknown)} that are not in the material "
                    f"database ({db.materials_path.name}). Add them there first.")
            total = sum(mix.values())
            if mix and abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"{self.name}: class {cid} ({entry.get('name')}) mixture sums "
                    f"to {total:.6f}, not 1.0. Fractions are of the radar-visible "
                    f"exterior surface and must be a distribution.")
            self.class_names[cid] = str(entry.get("name", cid))
            self.mixtures[cid] = mix

    # ------------------------------------------------------------------
    @property
    def n_materials(self) -> int:
        return len(self.db)

    def mixture_matrix(self) -> np.ndarray:
        """(n_classes, n_materials) dense fraction matrix, row i = class id i.

        Classes absent from the file, and classes that map to no material, are
        all-zero rows -- see the module docstring on why that is a value.
        """
        M = np.zeros((self.n_classes, self.n_materials), dtype=np.float64)
        for cid, mix in self.mixtures.items():
            for mat, frac in mix.items():
                M[cid, self.db.index[mat]] = frac
        return M

    def point_mixtures(self, labels: np.ndarray) -> np.ndarray:
        """(N, n_materials) material mixture per point, from integer labels.

        Labels outside the space (including the -1 this pipeline uses for
        "no label available") produce an all-zero row.
        """
        lut = self.mixture_matrix()
        lab = np.asarray(labels, dtype=np.int64)
        out = np.zeros((len(lab), self.n_materials), dtype=np.float64)
        ok = (lab >= 0) & (lab < len(lut))
        out[ok] = lut[lab[ok]]
        return out

    def __repr__(self):
        return (f"LabelSpace({self.name!r}, {self.n_classes} classes -> "
                f"{self.n_materials} materials)")


@lru_cache(maxsize=8)
def load_label_space(name_or_path: str) -> LabelSpace:
    """Load `configs/materials/<name>.yaml` (or an explicit path)."""
    p = Path(name_or_path)
    if not p.exists():
        p = LABEL_SPACE_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        available = sorted(q.stem for q in LABEL_SPACE_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"no material label space {name_or_path!r} ({p}). Available: "
            f"{available}. A new dataset taxonomy is a new file here plus one "
            f"line in the dataset config -- no code change.")
    return LabelSpace(yaml.safe_load(p.read_text()), path=p)


def available_label_spaces() -> list[str]:
    return sorted(p.stem for p in LABEL_SPACE_DIR.glob("*.yaml"))
