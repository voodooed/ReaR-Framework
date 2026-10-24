"""The head interface every architecture implements, and the registry.

ONE INTERFACE, TWO OUTPUTS
--------------------------
A head is a single network that emits BOTH learned quantities from one forward
pass:

    forward(x) -> {"det_logit": (N,), "rcs_mean": (N,), ...}

Detection and RCS were once two independent networks and two checkpoints, which
made every downstream stage carry a pair of everything -- two scalers, two
calibrations, two loading paths, two chances to pair the wrong files. They are
one object now. `MLPHead` and `PRISMNet` differ only in what happens between the
input and those two outputs, so training, calibration, generation, evaluation and
the downstream builders all address them identically:

    build_head("prism", n_features=8, n_material=10)      construct
    save_head(path, model, ...)                           persist, self-describing
    model, ckpt = load_head(path)                         restore, no guessing

ADDING AN ARCHITECTURE is a subclass plus one line in `HEAD_REGISTRY`. Nothing
else in the framework needs to learn its name: the checkpoint records which
architecture built it, and `load_head` rebuilds exactly that.

THE CHECKPOINT IS SELF-DESCRIBING, AND THAT IS A HARD RULE
----------------------------------------------------------
Everything needed to reproduce the transform a head was trained under travels
inside its file: the architecture name and its config, the feature scaler
(median/IQR/clip), the feature variant and column names, the material block's
configuration, and the RCS target standardisation. Inference never recomputes
any of it. Train/serve skew in this pipeline is silent -- a mis-scaled feature
produces a plausible-looking radar cloud, not an error -- so the constants are
carried rather than re-derived.

MATERIAL IS PART OF THE CONTRACT
--------------------------------
`n_material` and the material configuration are recorded on the checkpoint, and
the input matrix is always `[physics | material]` in that order. A head trained
with material cannot silently be fed an 8-column matrix, and a checkpoint from a
different material variant cannot be scored as if it were this one.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
from torch import nn


class JointHead(nn.Module):
    """Base class for a head emitting detection and RCS from one network.

    Subclasses set `self.arch_config` to the kwargs that rebuild them, and
    implement `forward` returning at least `det_logit` and `rcs_mean`.
    """

    #: registry key, set by the subclass
    architecture: str = ""

    def __init__(self, n_features: int, n_material: int = 0):
        super().__init__()
        self.n_features = int(n_features)
        self.n_material = int(n_material)
        self.arch_config: dict = {}

    # ---- required -------------------------------------------------------
    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """`x` is (N, n_features + n_material), physics columns first.

        Returns a dict with at least:
          det_logit (N,)  detection logit, pre-sigmoid
          rcs_mean  (N,)  point estimate of RCS in STANDARDISED units
        A head with a predictive distribution adds its parameters alongside.
        """

    # ---- provided -------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        return self.n_features + self.n_material

    @torch.no_grad()
    def predict_detection_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Detection probability in [0, 1]. Uncalibrated -- the Platt map is a
        separate, head-specific artifact applied by the generator."""
        return torch.sigmoid(self.forward(x)["det_logit"])

    @torch.no_grad()
    def predict_rcs(self, x: torch.Tensor) -> torch.Tensor:
        """Point estimate of RCS, standardised. De-standardise with the
        checkpoint's y_mu/y_sigma."""
        return self.forward(x)["rcs_mean"]

    @torch.no_grad()
    def rcs_variance(self, x: torch.Tensor) -> torch.Tensor | None:
        """Predictive variance in standardised units, or None for a head that
        emits a point estimate only."""
        return None

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def check_input(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.n_inputs:
            raise ValueError(
                f"{type(self).__name__} expects {self.n_inputs} input columns "
                f"({self.n_features} physics + {self.n_material} material), got "
                f"{x.shape[-1]}. The material block is appended AFTER the "
                f"physics columns and after scaling -- see rear/materials/.")


# --------------------------------------------------------------- registry ---
def _registry() -> dict:
    from rear.models.mlp import MLPHead
    from rear.models.prism import PRISMNet
    return {"mlp": MLPHead, "prism": PRISMNet}


def available_architectures() -> list[str]:
    return sorted(_registry())


def build_head(architecture: str, n_features: int = 8, n_material: int = 0,
               **cfg) -> JointHead:
    """Construct a head by name. The one place architectures are selected."""
    reg = _registry()
    if architecture not in reg:
        raise ValueError(
            f"unknown architecture {architecture!r}; available: {sorted(reg)}. "
            f"A new one is a JointHead subclass plus an entry in "
            f"rear/models/base.py's registry.")
    return reg[architecture](n_features=n_features, n_material=n_material, **cfg)


# ------------------------------------------------------------ persistence ---
def save_head(path: Path | str, model: JointHead, *, scaler: dict,
              feature_names, feature_variant: str, y_mu: float, y_sigma: float,
              material: dict | None = None, extra: dict | None = None) -> dict:
    """Write a self-describing checkpoint. See the module docstring on why.

    `scaler` is {"mu", "sigma", "clip"} for the PHYSICS columns only -- the
    material block is deliberately not standardised (it is already bounded, and
    standardising would hand a near-zero-IQR column an enormous multiplier), so
    a run with material keeps a byte-identical physics scaler to one without.
    """
    ck = {
        "architecture": model.architecture,
        "arch_config": dict(model.arch_config),
        "state_dict": model.state_dict(),
        "n_features": model.n_features,
        "n_material": model.n_material,
        "mu": scaler["mu"], "sigma": scaler["sigma"],
        "clip": scaler.get("clip"),
        "feature_names": list(feature_names),
        "feature_variant": feature_variant,
        "y_mu": float(y_mu), "y_sigma": float(y_sigma),
        "material": dict(material or {"enabled": False, "n_features": 0}),
    }
    ck.update(extra or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    return ck


def load_head(path: Path | str, device: str = "cpu") -> tuple[JointHead, dict]:
    """Rebuild the exact head a checkpoint describes, plus the checkpoint."""
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    if "architecture" not in ck:
        raise ValueError(
            f"{path} carries no `architecture` field. It predates the unified "
            f"head interface and cannot be loaded; retrain with "
            f"scripts/train_heads.py.")
    model = build_head(ck["architecture"], n_features=ck["n_features"],
                       n_material=ck.get("n_material", 0),
                       **{k: v for k, v in ck["arch_config"].items()
                          if k not in ("n_features", "n_material")})
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
