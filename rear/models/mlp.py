"""MLPHead — the unstructured baseline, and the reference every claim is against.

Two hidden ReLU layers over the standardised input, one linear read-out per
task. That is the whole model: 64-64, ~5k parameters with the material block
attached. It is deliberately small, because in this pipeline the physics is
supposed to live in the FEATURES rather than be learned from scratch — and that
is measured, not assumed. Widening it to 256-256-256 (134k parameters) scores
**0.0128 WORSE on detection AUC and 0.0061 worse on w-R^2** than the 4.8k
incumbent. Extra unstructured capacity is not what this problem is short of.

WHY IT IS STILL HERE
--------------------
It is the control that makes PRISM's numbers readable. PRISM wins by +0.016 AUC
and +0.017 w-R^2 over this head under an identical protocol — same features,
same scaler, same split, same optimiser, same sub-epoch validation, same
material block. Without an arm that changes only the architecture, that
difference could be any of a dozen other things.

It is also the conservative deployment choice on one axis that matters for
sim2real: measured input sensitivity on the deployment dataset is LOWER for this
head than for any structured alternative (Jacobian 1.43 against PRISM-2's 2.57),
and its generated field is smoother. When cross-dataset smoothness outranks
held-out accuracy, this is the head to ship.

MATERIAL ENTERS AS EXTRA COLUMNS
--------------------------------
For an unstructured MLP that is the only sensible thing it can mean: there is no
decomposition to preserve, so the material block is simply concatenated onto the
physics columns and the first layer widens. (PRISM handles it differently, and
for a reason — see `prism.py`.) The input contract is identical either way:
`[physics | material]`, material appended after scaling.

THE RCS OUTPUT IS A POINT ESTIMATE
----------------------------------
No mixture, no predictive variance. That is the honest baseline: the heavy tail
of the RCS target is handled by the Huber loss, i.e. by DOWN-WEIGHTING it rather
than modelling it. The generator therefore has only a conditional mean to sample
around for this head, which is exactly the capability difference the mixture in
PRISM buys.
"""

from __future__ import annotations

import torch
from torch import nn

from rear.models.base import JointHead

DEFAULT_HIDDEN = (64, 64)
#: Point-estimate head: the robust loss is the whole tail strategy.
DEFAULT_RCS_LOSS = "huber"


class MLPHead(JointHead):
    """Shared trunk, two linear read-outs.

    One trunk rather than two independent networks: detection and RCS are
    functions of the same scatterer, the joint form is what PRISM does, and
    comparing a joint structured head against two disjoint MLPs would confound
    architecture with parameter sharing.
    """

    architecture = "mlp"

    def __init__(self, n_features: int = 8, n_material: int = 0,
                 hidden: tuple = DEFAULT_HIDDEN, dropout: float = 0.0):
        super().__init__(n_features, n_material)
        self.arch_config = dict(hidden=tuple(hidden), dropout=dropout)

        layers: list = []
        in_dim = self.n_inputs
        for width in hidden:
            layers += [nn.Linear(in_dim, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = width
        self.body = nn.Sequential(*layers)
        self.det = nn.Linear(in_dim, 1)
        self.rcs = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> dict:
        self.check_input(x)
        h = self.body(x)
        return {"det_logit": self.det(h).squeeze(-1),
                "rcs_mean": self.rcs(h).squeeze(-1)}
