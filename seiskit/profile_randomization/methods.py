"""Spatial-variability methods for 1D ground response (Hallal-style arms).

Three approaches share a common interface:

* :class:`ToroMethod` — Toro (1995) correlated :math:`V_s` randomization
* :class:`PasseriMethod` — Passeri travel-time (:math:`t^*`) randomization
* :class:`DmultMethod` — Dmin multiplier from :math:`V_s` contrast
  (Tao & Rathje 2019 method; Dawadi et al. 2026 contrast correlation)

Default configurations for the first two keep layering and bedrock depth fixed
(Vs-only / tts-only), matching the Response_Variability Hallal arms. Full NHPP
and interface randomization remain available via :class:`ProfileRandomizationConfig`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from .common import build_base_case_profile, _nominal_soil_samples
from .models import ProfileRandomizationConfig, RandomizedProfile
from .passeri import generate_passeri_profile
from .toro import generate_toro_profile

# ---------------------------------------------------------------------------
# Dmult formula (Tao & Rathje 2019; Dawadi et al. 2026)
# ---------------------------------------------------------------------------

# Least-squares line through the four downhole-array sites of Dawadi et al.
# (2026) Fig. 9, (contrast, multiplier) = (2.93, 10), (4.83, 8), (7.84, 3),
# (9.47, 2), clipped to the calibrated range [2, 10]. Those multipliers were
# calibrated on Darendeli (2001) Dmin at f = 3 Hz, not on Taborda–Bielak ξ_Q.
_DMULT_SLOPE = -1.3
_DMULT_INTERCEPT = 13.90
_DMULT_LO = 2.0
_DMULT_HI = 10.0


def dmult_from_vs_contrast(vs1: float, vs2: float) -> float:
    """Damping multiplier from soil–rock :math:`V_s` contrast.

    .. math::

        D_\\mathrm{mult}
            = \\mathrm{clip}\\bigl(-1.3\\,V_{s2}/V_{s1} + 13.90,\\ 2,\\ 10\\bigr)

    ``vs2`` is the bedrock :math:`V_s` and ``vs1`` the time-averaged soil
    :math:`V_s` above it (Dawadi et al. 2026 definition of velocity contrast).

    Applied as :math:`\\xi = D_\\mathrm{mult}\\,D_\\mathrm{min}` where
    :math:`D_\\mathrm{min}` is Darendeli (2001) small-strain damping at 3 Hz
    (soil and rock, each at layer mid-depth), under OpenSees ``global_avg``
    on the whole column, with one base :math:`V_s` profile and no
    randomization.

    References
    ----------
    Tao, Y., and Rathje, E. (2019). "Insights into Modeling Small-Strain Site
    Response Derived from Downhole Array Data." *Journal of Geotechnical and
    Geoenvironmental Engineering*, 145(7).
    https://doi.org/10.1061/(ASCE)GT.1943-5606.0002048

    Dawadi, N., Mohammadi, K., Hallal, M. M., and Cox, B. R. (2026).
    "Insights on Numerical Damping Formulations Gained from Calibrating
    Two-Dimensional Ground Response Analyses at Downhole Array Sites."
    *Earthquake Spectra*. https://doi.org/10.1002/esp4.70087
    (preprint: arXiv:2511.04074)
    """
    vs1_f = float(vs1)
    vs2_f = float(vs2)
    if vs1_f <= 0.0:
        raise ValueError(f"vs1 must be positive, got {vs1_f}")
    contrast = vs2_f / vs1_f
    return float(np.clip(_DMULT_SLOPE * contrast + _DMULT_INTERCEPT, _DMULT_LO, _DMULT_HI))


def vs_contrast(vs1: float, vs2: float) -> float:
    """Return ``Vs2 / Vs1``."""
    vs1_f = float(vs1)
    if vs1_f <= 0.0:
        raise ValueError(f"vs1 must be positive, got {vs1_f}")
    return float(vs2) / vs1_f


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class SpatialVariabilityMethod(ABC):
    """Common interface for 1D spatial-variability / damping-proxy methods."""

    name: ClassVar[str]
    label: ClassVar[str]

    @abstractmethod
    def generate_profile(
        self,
        config: ProfileRandomizationConfig,
        rng: np.random.Generator,
    ) -> RandomizedProfile:
        """Return a (possibly randomized) full-column :math:`V_s` profile."""

    def damping_multiplier(self, vs1: float, vs2: float) -> float:
        """Scale applied to Taborda–Bielak :math:`\\xi_Q` / lab Dmin. Default 1.0."""
        return 1.0

    def uses_elemental_damping(self) -> bool:
        """If True, OpenSees should use ``elemental_varying`` damping."""
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete methods
# ---------------------------------------------------------------------------


class ToroMethod(SpatialVariabilityMethod):
    """Toro (1995) correlated :math:`\\ln V_s` randomization.

    With ``randomize_layer_thickness=False`` and ``randomize_bedrock_depth=False``
    this is the Vs-only Hallal arm (one soil layer, fixed :math:`H`). Set those
    flags True for full NHPP + interface randomization.

    For Response_Variability, ``sigma_ln_vs`` is set equal to the design CoV.
    """

    name = "toro"
    label = "Toro Vs randomization"

    def generate_profile(
        self,
        config: ProfileRandomizationConfig,
        rng: np.random.Generator,
    ) -> RandomizedProfile:
        return generate_toro_profile(config, rng)


class PasseriMethod(SpatialVariabilityMethod):
    """Passeri cumulative travel-time (:math:`t^*`) randomization.

    With fixed :math:`H` / no NHPP this is the tts-only Hallal arm.
    For Response_Variability, ``sigma_ln_tts`` is set equal to the design CoV.
    """

    name = "passeri"
    label = "Passeri tts randomization"

    def generate_profile(
        self,
        config: ProfileRandomizationConfig,
        rng: np.random.Generator,
    ) -> RandomizedProfile:
        return generate_passeri_profile(config, rng)


class DmultMethod(SpatialVariabilityMethod):
    """Dmin multiplier from :math:`V_s` contrast (Tao & Rathje 2019).

    Returns the deterministic base profile and a contrast-based
    :func:`dmult_from_vs_contrast` multiplier. No seed dependence — one 1D
    simulation per column.

    See :func:`dmult_from_vs_contrast` for the formula and citation.
    """

    name = "dmult"
    label = "Dmult damping modification"

    def generate_profile(
        self,
        config: ProfileRandomizationConfig,
        rng: np.random.Generator,
    ) -> RandomizedProfile:
        del rng  # deterministic base profile
        vs_depth = build_base_case_profile(config)
        n_soil = _nominal_soil_samples(config)
        return RandomizedProfile(
            vs_depth=vs_depth,
            n_soil_samples=n_soil,
            interface_depth=float(config.thickness),
        )

    def damping_multiplier(self, vs1: float, vs2: float) -> float:
        return dmult_from_vs_contrast(vs1, vs2)

    def uses_elemental_damping(self) -> bool:
        """Dmult scales whole-profile ``global_avg`` Darendeli Dmin (soil + rock)."""
        return False


# ---------------------------------------------------------------------------
# Registry (library names + Response_Variability aliases)
# ---------------------------------------------------------------------------

_METHOD_REGISTRY: dict[str, SpatialVariabilityMethod] = {
    "toro": ToroMethod(),
    "passeri": PasseriMethod(),
    "dmult": DmultMethod(),
    # Response_Variability method ids
    "hallal_vs": ToroMethod(),
    "hallal_tts": PasseriMethod(),
    "hallal_dmin": DmultMethod(),
}


def get_method(name: str) -> SpatialVariabilityMethod:
    """Return the method instance for ``name`` (library or Hallal alias)."""
    key = str(name).strip().lower()
    try:
        return _METHOD_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_METHOD_REGISTRY))
        raise ValueError(f"Unknown variability method {name!r}. Known: {known}") from exc


def hallal_profile_config(
    *,
    vs1: float,
    H: float,
    cov: float,
    vs2: float,
    dz: float,
    bedrock_thickness: float = 10.0,
) -> ProfileRandomizationConfig:
    """Standard Hallal-arm config: fixed :math:`H`, no NHPP, ``σ_ln =`` CoV."""
    return ProfileRandomizationConfig(
        vs_mean=float(vs1),
        thickness=float(H),
        dz=float(dz),
        cov=float(cov),
        vs_bedrock=float(vs2),
        bedrock_thickness=float(bedrock_thickness),
        sigma_ln_vs=float(cov),
        sigma_ln_tts=float(cov),
        use_full_model=True,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )
