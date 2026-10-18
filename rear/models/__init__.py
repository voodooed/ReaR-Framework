"""Learning architectures, behind one interface.

    build_head("prism", n_features=8, n_material=10)   construct by name
    load_head(path)                                    restore from checkpoint

`base.py` defines the `JointHead` contract every architecture implements and the
registry that selects between them; `mlp.py` and `prism.py` are the two shipped
heads; `losses.py` holds the objectives, deliberately outside the model files so
that swapping the architecture does not swap the loss.

Adding an architecture is a `JointHead` subclass plus one registry entry. No
training, generation, calibration or evaluation code needs to change: they all
address a head through the interface, never by name.
"""

from rear.models.base import (JointHead, available_architectures, build_head,
                              load_head, save_head)
from rear.models.mlp import MLPHead
from rear.models.prism import PRISMNet

__all__ = ["JointHead", "build_head", "load_head", "save_head",
           "available_architectures", "MLPHead", "PRISMNet"]
