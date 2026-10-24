"""The training objectives, shared by every architecture.

Kept out of the model files so that swapping the head does not swap the loss:
an architecture comparison in which the two arms optimise different objectives
measures two changes at once. Both shipped heads train on the same objective,

    L = soft_BCE(det_logit; soft_label) + lambda * L_RCS,      lambda = 1

and differ only in `L_RCS`, which is itself a property of the head's OUTPUT type
(a point estimate takes MSE/Huber, a mixture takes NLL + MSE) rather than of its
internal structure.

WHY THE HYBRID NLL+MSE FOR A MIXTURE HEAD
-----------------------------------------
NLL alone optimises the whole predictive DENSITY, but the reported score
(weighted R^2) only ever sees the conditional MEAN, and the two are not the same
target -- NLL will happily trade mean accuracy for tail shape. Adding a
weighted-MSE term on the mixture mean makes the estimator that is actually
scored an explicit objective while the NLL term keeps the density calibrated.
The mixture mean is a differentiable function of (pi, mu), so both terms train
the same parameters and no separate head is needed. The two terms are on
comparable scales (both O(1) in standardised units); the weight is the untuned
default 1.0 and is reported as such.

WEIGHTING
---------
Every RCS loss is weighted per example by the V3 association soft label, so a
match the association is unsure about pulls the regression less hard, and is
normalised by TOTAL WEIGHT rather than example count -- the reported value is a
weighted mean and stays comparable across batches and splits whose confidence
mass differs.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Huber transition point, in units of the STANDARDISED target. 1.0 sigma is
# ~13.7 dBsm on the measured train distribution: quadratic within a typical
# spread, linear beyond it.
HUBER_DELTA = 1.0

# Weight on the conditional-mean term of the hybrid objective. Untuned default.
HYBRID_MSE_WEIGHT = 1.0


def soft_bce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """BCE against the continuous V3 soft label. Targets are NOT rounded.

    BCE with a probabilistic target is the cross-entropy H(target, prediction),
    minimised exactly when the prediction equals the target, so this trains the
    network to reproduce the association's confidence value rather than to pick
    a side. Rounding first would discard the calibration the soft labels exist
    for.
    """
    return nn.functional.binary_cross_entropy_with_logits(logits, soft_targets)


def weighted_mse_loss(prediction: torch.Tensor, target: torch.Tensor,
                      weight: torch.Tensor) -> torch.Tensor:
    """The loss whose minimiser is exactly the estimator w-R^2 scores."""
    return (((prediction - target) ** 2) * weight).sum() / weight.sum().clamp_min(1e-12)


def weighted_huber_loss(prediction: torch.Tensor, target: torch.Tensor,
                        weight: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Huber, weighted by association confidence.

    The RCS target has excess kurtosis +1.60 and 3.3% of values beyond 5x the
    median absolute deviation. Those tails are real physics -- specular flashes
    off flat metal and corner reflectors -- not noise, but under squared error a
    handful of +58 dBsm returns would dominate the gradient. Huber keeps them in
    the data without letting them set the solution. A mixture head MODELS them
    instead; this is the point-estimate alternative.
    """
    per_example = nn.functional.huber_loss(prediction, target, reduction="none",
                                           delta=delta)
    return (per_example * weight).sum() / weight.sum().clamp_min(1e-12)


def mixture_nll(logit_pi: torch.Tensor, mu: torch.Tensor,
                log_sigma: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
    """Weighted negative log-likelihood of a 1-D Gaussian mixture."""
    log_pi = torch.log_softmax(logit_pi, dim=-1)
    z = (target.unsqueeze(-1) - mu) / log_sigma.exp()
    log_comp = -0.5 * z ** 2 - log_sigma - 0.5 * math.log(2 * math.pi)
    nll = -torch.logsumexp(log_pi + log_comp, dim=-1)
    return (nll * weight).sum() / weight.sum().clamp_min(1e-12)


def rcs_loss(kind: str, out: dict, target: torch.Tensor,
             weight: torch.Tensor) -> torch.Tensor:
    """The RCS objective, selected by name.

        nll+mse  the production hybrid for a mixture head (see module docstring)
        nll      density only
        mse      point estimate, squared error
        huber    point estimate, robust
    """
    if kind == "nll+mse":
        return (mixture_nll(out["rcs_logit_pi"], out["rcs_mu"],
                            out["rcs_log_sigma"], target, weight)
                + HYBRID_MSE_WEIGHT * weighted_mse_loss(out["rcs_mean"], target, weight))
    if kind == "nll":
        return mixture_nll(out["rcs_logit_pi"], out["rcs_mu"],
                           out["rcs_log_sigma"], target, weight)
    if kind == "mse":
        return weighted_mse_loss(out["rcs_mean"], target, weight)
    if kind == "huber":
        return weighted_huber_loss(out["rcs_mean"], target, weight)
    raise ValueError(f"unknown rcs loss {kind!r}; expected one of "
                     f"'nll+mse', 'nll', 'mse', 'huber'")


def joint_loss(out: dict, y_det: torch.Tensor, y_rcs: torch.Tensor,
               w_rcs: torch.Tensor, matched: torch.Tensor,
               rcs_kind: str, lam: float = 1.0) -> torch.Tensor:
    """L = soft_BCE(detection) + lambda * L_RCS(matched candidates only).

    Detection trains on EVERY candidate; RCS only on those the association
    matched to a real detection, because only those have a target at all.
    Unmatched candidates are excluded rather than given a filler value.
    """
    loss = soft_bce_loss(out["det_logit"], y_det)
    if bool(matched.any()):
        sub = {k: (v[matched] if torch.is_tensor(v) else v) for k, v in out.items()}
        loss = loss + lam * rcs_loss(rcs_kind, sub, y_rcs[matched], w_rcs[matched])
    return loss
