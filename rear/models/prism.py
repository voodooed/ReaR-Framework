"""PRISM-2 — Physics-factorised Spectral Mixture head, with a material branch.

The production head. One network emits BOTH learned outputs — detection and RCS
— from five branches over disjoint feature groups that mirror the terms of the
radar range equation.

WHY A STRUCTURED NETWORK AND NOT A BIGGER MLP
---------------------------------------------
The 8 physics features are not an unstructured vector. They are the arguments of
the radar range equation, which is ADDITIVE IN dB,

    SNR_dB = P_t + G_tx(az, el) + G_rx(az, el) + sigma_dB
             - 40 log10(r) - L_occ + const,

and the network's inductive bias should say so. Measured, this matters and mere
capacity does not: an unstructured MLP at 27x the parameter count scores BELOW
the 4.8k baseline, while PRISM-2 at 86k beats it by +0.016 AUC / +0.017 w-R^2.

THE FOUR SURVIVING IDEAS
------------------------
1  SPECTRAL EMBEDDING. Each standardised feature is lifted to
   `[z, sin(Bz), cos(Bz)]` on a FIXED dyadic bank, `B = (pi/4) * 2^i`, i = 0..3.
   A ReLU MLP on raw scalars has a documented low-frequency bias, and azimuth
   (beam lobes), range/path_loss (four decades of dynamic range) and
   occlusion_depth (a sharp integer knee) carry genuinely high-frequency
   structure. Frequencies are buffers, not parameters: a basis is not something
   to fit, and freezing it keeps seed-to-seed spread a measure of optimisation
   noise alone.

2  LINK-BUDGET FACTORISATION. Four branches over DISJOINT, EXHAUSTIVE groups —
   propagation {range, path_loss}, beam {azimuth, elevation}, surface
   {cos_angle, point_density}, occlusion {occlusion_depth, height_above_ground}.
   Each emits an embedding AND a scalar per task, which makes the model a neural
   GAM in the physics groups: the learned range law and beam pattern are
   separately plottable objects, not hidden activations.

3  THE MATERIAL BRANCH — a FIFTH group over the 10-column electromagnetic block,
   carrying the `sigma_dB` term the geometry features never had. See below.

4  MIXTURE-DENSITY RCS. K = 3 Gaussians over standardised dBsm. The target has
   excess kurtosis +1.60 and 3.3% of values beyond 5x the MAD — specular flashes
   off flat metal and corner reflectors, which are real physics. A Huber loss
   DOWN-WEIGHTS them; a mixture MODELS them. The reported point estimate is the
   mixture mean (Bayes-optimal under squared error, hence the right estimator
   for w-R^2), and the generator gains a conditional distribution to sample from
   plus a per-candidate predictive variance at no extra cost.

K = 4 FREQUENCIES, AND WHY THE NUMBER IS THE ARCHITECTURE
---------------------------------------------------------
The previous version of this head used K = 8 plus FiLM cross-conditioning and a
low-rank bilinear term. It scored best on held-out fitting data and produced a
visibly wrong field on the deployment dataset. Held-out data is INTERPOLATION;
the generator's job is EXTRAPOLATION. Measured input-Jacobian ratios
(deployment / fitting): the plain MLP 0.78, that head **1.26** — its sensitivity
RISES out of distribution where the MLP's falls.

The mechanism is the frequency bank. At K = 8 the top channel is
`B = (pi/4) * 128 = 100.53` rad per standardised unit — 32 oscillations across
the interquartile range — so a covariate shift of 0.59 standardised units
(measured median; elevation 1.84) rotates that channel through 9-29 full
periods. The embedding of a shifted input is not "nearby", it is DECORRELATED:
high-frequency Fourier features behave like a hash outside their training
support, and FiLM and the bilinear term then multiply by it.

Cutting K from 8 to 4 alone removes 62% of that failure while IMPROVING w-R^2.
Removing FiLM raises AUC 0.8466 -> 0.8530. The bilinear term contributes
nothing. The residual MLP path was a liability — deleting it was the single
largest RCS gain in the ladder. **None of them are switches here.** They are
gone, and reintroducing one is a deliberate act against a measured result rather
than flipping a default. K = 2 is the conservative alternative if deployment
smoothness outranks fitting accuracy (`n_frequencies=2`).

THE MATERIAL BLOCK IS A BRANCH, NOT EXTRA INPUT COLUMNS
-------------------------------------------------------
Concatenating it onto the input would have dissolved it into whichever physics
branch happened to read those columns and destroyed the decomposition. As its
own branch it becomes one more separately plottable term — the learned
reflectivity response — and the four physics groups keep seeing only their own 8
columns. `_branch_pass` splits `[physics | material]`.

**It is deliberately NOT Fourier-lifted.** The block is bounded (compositions
and |Gamma| in [0, 1]) and smooth, and a spectral lift would reintroduce exactly
the extrapolation failure K = 4 exists to remove — on the one input whose
train/deploy distributions are KNOWN to differ (built from predicted semantics
while fitting, from simulator ground truth at deployment). The asymmetry with
the physics features is intentional.

SYMMETRY BREAKING IS MANDATORY
------------------------------
With the mixture read-out's weight AND bias both zero-initialised, all K
components are numerically identical, receive identical gradients forever, and
the mixture silently degenerates to a single Gaussian **with no error raised**.
An entire ablation ladder was run with that bug present. The components are
therefore given distinct initial locations and scales; in the shipped model they
are live, with mean weights [0.122, 0.610, 0.268].
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rear.models.base import JointHead

N_PHYSICS_FEATURES = 8

# Column indices in the 8-feature "drop" variant. Order is fixed by
# rear/geometry/features.py and is a contract.
F_RANGE, F_AZIMUTH, F_ELEVATION, F_COS_ANGLE = 0, 1, 2, 3
F_HEIGHT, F_DENSITY, F_PATH_LOSS, F_OCCLUSION = 4, 5, 6, 7

#: The four link-budget groups. Disjoint and exhaustive over the 8 features.
LINK_BUDGET_GROUPS = {
    "propagation": (F_RANGE, F_PATH_LOSS),      # -40 log10(r)
    "beam": (F_AZIMUTH, F_ELEVATION),           # G_tx + G_rx
    "surface": (F_COS_ANGLE, F_DENSITY),        # scatterer response
    "occlusion": (F_OCCLUSION, F_HEIGHT),       # -L_occ
}
MATERIAL_BRANCH = "material"                    # the sigma_dB argument

# K = 4, not 8. The single most consequential number in this file; see the
# module docstring.
DEFAULT_N_FOURIER = 4
DEFAULT_BRANCH_WIDTH = 64
DEFAULT_TRUNK_WIDTH = 128
DEFAULT_N_MIXTURE = 3
#: Production RCS objective for a mixture head. See rear/models/losses.py.
DEFAULT_RCS_LOSS = "nll+mse"

# Predictive sigma is clamped in log space. The lower bound stops a component
# collapsing onto a single point (the classic mixture-NLL divergence); the upper
# bound stops a component going so broad it explains everything at once.
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -4.0, 3.0


class FourierFeatures(nn.Module):
    """Lift each scalar to `[z, sin(Bz), cos(Bz)]` on a fixed dyadic bank.

    The input is already median/IQR standardised and clipped to +/-10, so
    `base = pi/4` puts the lowest frequency at roughly one period across the
    interquartile range.
    """

    def __init__(self, n_features: int = N_PHYSICS_FEATURES,
                 n_frequencies: int = DEFAULT_N_FOURIER,
                 base: float = math.pi / 4):
        super().__init__()
        self.n_features = n_features
        self.n_frequencies = n_frequencies
        freqs = base * (2.0 ** torch.arange(n_frequencies, dtype=torch.float32))
        self.register_buffer("freqs", freqs)

    @property
    def out_dim_per_feature(self) -> int:
        return 1 + 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, F) -> (N, F, 1 + 2K)."""
        z = x.unsqueeze(-1)
        arg = z * self.freqs
        return torch.cat([z, torch.sin(arg), torch.cos(arg)], dim=-1)


class LinkBudgetBranch(nn.Module):
    """One additive term of the radar equation.

    Emits a `width`-dim embedding for the trunk AND a scalar per task. The
    scalar read-out is what makes the model a neural GAM in its groups: the
    prediction is the sum of separately plottable one-group functions plus the
    trunk's correction.
    """

    def __init__(self, in_dim: int, width: int = DEFAULT_BRANCH_WIDTH,
                 n_tasks: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.GELU(),
                  nn.Linear(width, width), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.readout = nn.Linear(width, n_tasks)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return h, self.readout(h)


class PRISMNet(JointHead):
    """PRISM-2. Detection logit + K-component RCS mixture from one network."""

    architecture = "prism"

    def __init__(self, n_features: int = N_PHYSICS_FEATURES, n_material: int = 0,
                 n_frequencies: int = DEFAULT_N_FOURIER,
                 branch_width: int = DEFAULT_BRANCH_WIDTH,
                 trunk_width: int = DEFAULT_TRUNK_WIDTH,
                 n_mixture: int = DEFAULT_N_MIXTURE,
                 dropout: float = 0.0):
        super().__init__(n_features, n_material)
        if n_features != N_PHYSICS_FEATURES:
            raise ValueError(
                f"the link-budget factorisation is defined over the "
                f"{N_PHYSICS_FEATURES}-feature 'drop' variant and its fixed "
                f"column order; got n_features={n_features}. A different physics "
                f"feature set needs its own group definition, not a reshape.")
        self.arch_config = dict(n_frequencies=n_frequencies,
                                branch_width=branch_width,
                                trunk_width=trunk_width,
                                n_mixture=n_mixture, dropout=dropout)
        self.n_mixture = n_mixture

        # ---- spectral embedding, physics columns only ----------------------
        self.embed = FourierFeatures(n_features, n_frequencies)
        per_feat = self.embed.out_dim_per_feature

        # ---- the four physics branches -------------------------------------
        self.groups = dict(LINK_BUDGET_GROUPS)
        self.branch_names = list(self.groups)
        for name, cols in self.groups.items():
            self.register_buffer(f"cols_{name}",
                                 torch.tensor(cols, dtype=torch.long),
                                 persistent=False)
        self.branches = nn.ModuleDict({
            name: LinkBudgetBranch(len(cols) * per_feat, branch_width,
                                   n_tasks=2, dropout=dropout)
            for name, cols in self.groups.items()})

        # ---- the fifth branch: material (NOT Fourier-lifted) ---------------
        self.material_branch = (
            LinkBudgetBranch(n_material, branch_width, n_tasks=2, dropout=dropout)
            if n_material > 0 else None)

        n_branches = len(self.branch_names) + (1 if n_material else 0)
        fused_dim = branch_width * n_branches

        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_width), nn.GELU(),
            nn.Linear(trunk_width, trunk_width), nn.GELU())
        if dropout > 0:
            self.trunk = nn.Sequential(*self.trunk, nn.Dropout(dropout))

        # Task read-outs off the trunk, zero-initialised, so the model begins as
        # the pure additive GAM and the trunk's non-additive correction is
        # something the data has to pay for.
        self.det_head = nn.Linear(trunk_width, 1)
        nn.init.zeros_(self.det_head.weight)
        nn.init.zeros_(self.det_head.bias)

        # Mixture parameters: (logit pi, mu, log sigma) per component.
        # SYMMETRY BREAKING — see the module docstring. Locations are symmetric
        # about 0, so at uniform mixing weights the mixture MEAN equals the
        # additive shift; the 1e-3 weight init perturbs slightly off uniform, so
        # the model starts NEAR (not exactly at) the pure additive predictor.
        self.rcs_head = nn.Linear(trunk_width, 3 * n_mixture)
        nn.init.normal_(self.rcs_head.weight, std=1e-3)
        with torch.no_grad():
            b = torch.zeros(3 * n_mixture)
            if n_mixture > 1:
                b[n_mixture:2 * n_mixture] = torch.linspace(-1.0, 1.0, n_mixture)
                b[2 * n_mixture:] = torch.linspace(-0.7, 0.7, n_mixture)
            self.rcs_head.bias.copy_(b)

        self.det_bias = nn.Parameter(torch.zeros(1))
        self.rcs_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def _branch_pass(self, x: torch.Tensor):
        """Run the branches; return (fused embedding, additive scalars, per-branch).

        The input is `[physics | material]` and is split here, so the spectral
        embedding and the four link-budget groups only ever see the physics
        columns they are defined over.
        """
        xm = None
        if self.material_branch is not None:
            x, xm = x[:, :-self.n_material], x[:, -self.n_material:]
        e = self.embed(x)                                   # (N, F, 1 + 2K)
        h, s = {}, 0.0
        for name in self.branch_names:
            cols = getattr(self, f"cols_{name}")
            h[name], si = self.branches[name](e.index_select(1, cols).flatten(1))
            s = s + si
        order = list(self.branch_names)
        if xm is not None:
            h[MATERIAL_BRANCH], sm = self.material_branch(xm)
            s = s + sm
            order.append(MATERIAL_BRANCH)
        return torch.cat([h[n] for n in order], dim=-1), s, h

    def forward(self, x: torch.Tensor) -> dict:
        """`det_logit` (N,), `rcs_logit_pi`/`rcs_mu`/`rcs_log_sigma` (N, K),
        `rcs_pi` (N, K) and `rcs_mean` (N,) — all in standardised units."""
        self.check_input(x)
        fused, additive, _ = self._branch_pass(x)
        t = self.trunk(fused)

        det = additive[:, 0] + self.det_head(t).squeeze(-1) + self.det_bias
        rcs_shift = additive[:, 1] + self.rcs_bias

        logit_pi, mu, log_sigma = self.rcs_head(t).chunk(3, dim=-1)
        # The additive/GAM scalar shifts the whole mixture location, so the GAM
        # decomposition applies to the predicted RCS as a whole rather than to
        # one arbitrary component.
        mu = mu + rcs_shift.unsqueeze(-1)
        log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
        pi = torch.softmax(logit_pi, dim=-1)

        return {"det_logit": det, "rcs_logit_pi": logit_pi, "rcs_mu": mu,
                "rcs_log_sigma": log_sigma, "rcs_pi": pi,
                "rcs_mean": (pi * mu).sum(-1)}

    # ------------------------------------------------- inspection helpers ---
    @torch.no_grad()
    def branch_contributions(self, x: torch.Tensor) -> dict:
        """Per-group additive scalars (N, 2), for plotting the learned range
        law, beam pattern and reflectivity response as separate objects.

        This is what the factorisation buys that an unstructured head cannot
        offer at all: the link budget is inspectable after training.
        """
        self.check_input(x)
        xm = None
        if self.material_branch is not None:
            x, xm = x[:, :-self.n_material], x[:, -self.n_material:]
        e = self.embed(x)
        out = {}
        for name in self.branch_names:
            cols = getattr(self, f"cols_{name}")
            _, si = self.branches[name](e.index_select(1, cols).flatten(1))
            out[name] = si
        if xm is not None:
            _, sm = self.material_branch(xm)
            out[MATERIAL_BRANCH] = sm
        return out

    @torch.no_grad()
    def rcs_variance(self, x: torch.Tensor) -> torch.Tensor:
        """Predictive variance of the RCS mixture, in standardised units."""
        o = self.forward(x)
        pi, mu, sd = o["rcs_pi"], o["rcs_mu"], o["rcs_log_sigma"].exp()
        m = o["rcs_mean"].unsqueeze(-1)
        return (pi * (sd ** 2 + (mu - m) ** 2)).sum(-1)

    @torch.no_grad()
    def sample_rcs(self, x: torch.Tensor, generator=None) -> torch.Tensor:
        """Draw one RCS sample per candidate from the predicted mixture.

        The capability a point-estimate head does not have. Not used by the
        default generation path — which samples the real RCS MARGINAL through
        the fitted quantile calibration instead, a measured trade documented in
        rear/calibration/rcs_calibration.py — but exposed because the mixture is
        a genuine conditional distribution and a caller may want it.
        """
        o = self.forward(x)
        k = torch.multinomial(o["rcs_pi"], 1, generator=generator).squeeze(-1)
        idx = torch.arange(len(k), device=k.device)
        mu = o["rcs_mu"][idx, k]
        sd = o["rcs_log_sigma"][idx, k].exp()
        return mu + sd * torch.randn(mu.shape, generator=generator, device=mu.device)
