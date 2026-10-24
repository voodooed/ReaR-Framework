"""The electromagnetic model. Pure physics, no data.

Every constant this module needs is supplied by the caller (ultimately from
`rear/materials/data/itu_p2040_4.yaml`); nothing electromagnetic is hardcoded
here. That separation is deliberate: the database is inspectable and editable
as data, and the four functions below are the only place a Fresnel coefficient
is ever computed.

THE MODEL
---------
Recommendation ITU-R P.2040-4 parameterises the two constants as power laws in
frequency (GHz):

    eps_r'(f) = a * f^b            real relative permittivity
    sigma(f)  = c * f^d            conductivity, S/m

and everything else follows. The imaginary part comes from sigma,

    eps_r''(f) = sigma / (2 pi f eps_0)

and the normal-incidence Fresnel reflection coefficient of a smooth half-space
is the physically meaningful reflectivity scalar:

    Gamma_0 = (1 - sqrt(eps_c)) / (1 + sqrt(eps_c)),   eps_c = eps_r' - j eps_r''

WHY A STANDARD AND NOT A HAND-PICKED CONSTANT
---------------------------------------------
Every number is traceable to an international standard with a stated frequency
of validity, so the material representation is auditable rather than invented.
That matters more than accuracy: an untraceable constant cannot be checked, and
this project has already been burned once by a number nobody could re-derive.

VALIDITY IS TRACKED, NOT ASSUMED
--------------------------------
P.2040's bands do not all cover 77 GHz:

    concrete, glass, wood, metal, plasterboard   1-100 GHz    -> IN BAND
    asphalt_concrete                             1-40  GHz    -> EXTRAPOLATED
    brick                          gap between 40 and 110 GHz -> EXTRAPOLATED
    very/medium_dry_ground, wet_ground           1-10  GHz    -> EXTRAPOLATED HARD

`evaluate()` returns the extrapolation status with every value and never
silently hides it. `MaterialConstants.in_band` is False for anything evaluated
outside its stated range, and the feature builder propagates a low-confidence
mass column so an ablation can be run with extrapolated materials excluded.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

EPS_0 = 8.8541878128e-12          # F/m, vacuum permittivity (CODATA 2018)
C_LIGHT = 2.99792458e8            # m/s


@dataclass(frozen=True)
class MaterialConstants:
    """One material evaluated at one frequency, with its provenance attached."""

    name: str
    freq_ghz: float
    eps_real: float           # eps_r'
    sigma: float              # S/m
    eps_imag: float           # eps_r'' = sigma / (2 pi f eps_0)
    gamma_mag: float          # |Gamma_0|, normal-incidence Fresnel magnitude
    reflectivity_db: float    # 20 log10 |Gamma_0|
    loss_tangent: float       # eps_r'' / eps_r'
    in_band: bool             # was freq_ghz inside the stated validity band?
    band: tuple[float, float]  # the band whose coefficients were used
    source: str = ""

    @property
    def provenance(self) -> str:
        lo, hi = self.band
        status = "in band" if self.in_band else "EXTRAPOLATED"
        return f"{self.source or 'ITU-R P.2040-4 Table 3'}, {lo:g}-{hi:g} GHz ({status})"


def select_band(bands, freq_ghz: float):
    """The band to use, and whether the frequency is actually inside it.

    Prefers a band that contains the frequency. Failing that -- which is the
    interesting case, and the one that must never be silent -- it falls back to
    the band whose edge is nearest in LOG frequency, and reports in_band=False.

    Log distance, not linear: these are power laws in f, so a fixed RATIO of
    frequency is the natural step, and the choice changes answers. Brick is the
    case that shows why it needs stating rather than assuming: P.2040 fits brick
    over 1-40 GHz and again over 110-330 GHz, and 77 GHz falls in the GAP
    between the two measurement campaigns -- 1.93x above the lower band's edge,
    1.43x below the upper band's. Both rules pick 110-330 GHz here because it
    really is nearer, but the honest description is that brick at automotive
    radar frequency is unmeasured by this Recommendation, and the choice between
    its two fits changes sigma by an order of magnitude (0.048 vs 0.552 S/m).

    `bands` maps (f_lo, f_hi) -> coefficients.
    """
    for band in bands:
        if band[0] <= freq_ghz <= band[1]:
            return band, True
    nearest = min(bands, key=lambda b: min(abs(math.log(freq_ghz / b[0])),
                                           abs(math.log(freq_ghz / b[1]))))
    return nearest, False


def from_eps_sigma(name: str, freq_ghz: float, eps_real: float, sigma: float,
                   in_band: bool, band: tuple[float, float],
                   source: str = "") -> MaterialConstants:
    """Build the derived quantities from (eps_r', sigma) at one frequency.

    The single place the Fresnel coefficient is computed, so an ITU-table
    material and a locally declared one are indistinguishable downstream and
    validity travels with both.
    """
    eps_imag = sigma / (2.0 * math.pi * freq_ghz * 1e9 * EPS_0)
    sq = cmath.sqrt(complex(eps_real, -eps_imag))
    gamma = abs((1.0 - sq) / (1.0 + sq))
    return MaterialConstants(
        name=name, freq_ghz=freq_ghz, eps_real=eps_real, sigma=sigma,
        eps_imag=eps_imag, gamma_mag=gamma,
        reflectivity_db=20.0 * math.log10(max(gamma, 1e-12)),
        loss_tangent=eps_imag / eps_real if eps_real > 0 else float("inf"),
        in_band=in_band, band=band, source=source)


def evaluate(name: str, freq_ghz: float, bands: dict,
             source: str = "") -> MaterialConstants:
    """Evaluate one power-law entry at `freq_ghz`, carrying validity with it.

    `bands` is {(f_lo, f_hi): (a, b, c, d)} as read from the ITU database file.
    """
    band, in_band = select_band(bands, freq_ghz)
    a, b, c, d = bands[band]
    return from_eps_sigma(name, freq_ghz,
                          eps_real=a * freq_ghz ** b,
                          sigma=c * freq_ghz ** d,
                          in_band=in_band, band=band, source=source)


def wavelength_m(freq_ghz: float) -> float:
    """Free-space wavelength. At 77 GHz this is 3.893 mm -- the number that
    makes surface roughness a first-order effect for automotive radar and a
    second-order one for LiDAR."""
    return C_LIGHT / (freq_ghz * 1e9)


def rayleigh_smoothness(h_rms_m: float, freq_ghz: float,
                        incidence_rad: float = 0.0) -> float:
    """Specular power fraction of a rough surface (Ament / Boithias factor).

        rho_s = exp(-2 (2 pi h_rms cos(theta) / lambda)^2)

    Returns the fraction of incident power that reflects SPECULARLY; 1 - rho_s
    scatters diffusely. This is the term the permittivity table cannot supply
    and it is not a detail: at 77 GHz, lambda = 3.89 mm, so a road surface with
    h_rms ~ 1 mm is almost entirely diffuse while a car door at h_rms ~ 0.01 mm
    is a mirror. Two materials with identical eps_r can differ by tens of dB in
    what returns to the radar because of this term alone.
    """
    k = 2.0 * math.pi / wavelength_m(freq_ghz)
    return math.exp(-2.0 * (k * h_rms_m * math.cos(incidence_rad)) ** 2)
