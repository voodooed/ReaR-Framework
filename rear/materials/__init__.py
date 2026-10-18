"""Material information: semantic class -> material mixture -> permittivity.

A PERMANENT, DEFAULT-ON COMPONENT of the framework, not an optional experiment.
Every arm of the final pipeline -- MLP and PRISM alike -- reads the 10-column
electromagnetic block this package produces alongside the 8 physics features.

    database.py      the material/permittivity DATABASE (data/*.yaml) and the
                     one interface for reading it
    itu.py           the electromagnetic model: Fresnel, Ament, band validity.
                     Pure physics, no constants of its own
    taxonomy.py      dataset label space -> material mixture, loaded from
                     configs/materials/<space>.yaml
    features.py      per-candidate composition -> the 10-column feature block
    segmentation.py  RGB segmentation + camera->LiDAR label transfer, needed
                     only where a dataset ships no per-point semantics
    pipeline.py      frame + candidates -> composition. The entry point.

WHAT THE MATERIAL BLOCK IS WORTH, MEASURED
------------------------------------------
Against the same head without it: **w-R2 0.3957 -> 0.4470, AUC 0.8530 -> 0.8650**
(5 seeds; seed floor 0.0087 w-R2, 0.0089 AUC), at 1.08x input sensitivity on the
deployment dataset.

**But the permittivity values are not what carries that.** The pre-registered
control with the material->constant table DERANGED scores 0.4458 -- statistically
indistinguishable from the real table's 0.4470. The information is the SEMANTIC
PARTITION; the constants re-encode a category, exactly as theory predicts. The
EM representation is still the right engineering choice, on four grounds that do
not require it to carry information: best w-R2 and best cross-dataset
robustness of the material arms, a shared coordinate system that makes two
incompatible label vocabularies alignable at all, a fixed 10-column input
contract independent of the segmenter, and graceful placement of unseen classes.
See taxonomy.py. Do not claim the physics improves accuracy.
"""

from rear.materials.database import (DESCRIPTOR_NAMES, DEFAULT_RADAR_FREQ_GHZ,
                                     MATERIAL_DB, MATERIAL_NAMES,
                                     MaterialDatabase, load_database)
from rear.materials.features import (MATERIAL_FEATURE_NAMES,
                                     N_MATERIAL_FEATURES,
                                     aggregate_composition,
                                     aggregate_to_candidates,
                                     features_from_composition)
from rear.materials.itu import (MaterialConstants, rayleigh_smoothness,
                                wavelength_m)
from rear.materials.taxonomy import (LabelSpace, available_label_spaces,
                                     load_label_space)

__all__ = [
    "MATERIAL_DB", "MATERIAL_NAMES", "MaterialDatabase", "load_database",
    "DESCRIPTOR_NAMES", "DEFAULT_RADAR_FREQ_GHZ",
    "MaterialConstants", "wavelength_m", "rayleigh_smoothness",
    "LabelSpace", "load_label_space", "available_label_spaces",
    "MATERIAL_FEATURE_NAMES", "N_MATERIAL_FEATURES",
    "aggregate_composition", "features_from_composition",
    "aggregate_to_candidates",
]
