"""Spatial-variability methods for 1D ground response (Hallal-style arms).

Three approaches share a common interface:

* :class:`ToroMethod` — Toro (1995) correlated :math:`V_s` randomization
* :class:`PasseriMethod` — Passeri travel-time (:math:`t^*`) randomization
* :class:`DmultMethod` — damping modification from :math:`V_s` contrast
  (Hallal et al. 2022 Approach 5)

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
# Dmult formula (Hallal et al. 2022 Approach 5)
# ---------------------------------------------------------------------------

# Linear fit of damping multiplier vs impedance contrast, clipped to [2, 10].
# Used to inflate Campbell/Taborda ξ_Q (or lab Dmin) so a single 1D profile
# approximates scattering from spatial Vs variability.
_DMULT_SLOPE = -1.3
_DMULT_INTERCEPT = 13.90
_DMULT_LO = 2.0
_DMULT_HI = 10.0


def dmult_from_vs_contrast(vs1: float, vs2: float) -> float:
    """Damping multiplier from soil–rock :math:`V_s` contrast.

    .. math::

        D_\\mathrm{mult}
            = \\mathrm{clip}\\bigl(-1.3\\,V_{s2}/V_{s1} + 13.90,\\ 2,\\ 10\\bigr)

    Applied as :math:`\\xi = D_\\mathrm{mult}\\,\\xi_{Q}` under OpenSees
    ``global_avg`` damping on the **whole** column (soil harmonic-mean and
    rock :math:`\\xi_Q` both scaled). Approach 5: one base :math:`V_s`
    profile, no randomization.

    References
    ----------
    Hallal, M. M., Cox, B. R., and Vantassel, J. P. (2022).
    "Comparison of State-of-the-Art Approaches Used to Account for Spatial
    Variability in 1D Ground Response Analyses."
    *Journal of Geotechnical and Geoenvironmental Engineering*, 148(5).
    https://doi.org/10.1061/(ASCE)GT.1943-5606.0002774

    Google Scholar citation:
    https://scholar.google.com/citations?view_op=view_citation&hl=en&user=vI1wl_IAAAAJ&citation_for_view=vI1wl_IAAAAJ:5nxA0vEk-isC
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
        """Scale applied to Campbell :math:`\\xi_Q` / lab Dmin. Default 1.0."""
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
    """Damping modification from :math:`V_s` contrast (Hallal et al. 2022).

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
        """Dmult scales whole-profile ``global_avg`` ξ (soil + rock)."""
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
