"""Fabio Crameri perceptually-uniform, colorblind-friendly colormaps.

All continuous colour bars in the project must use one of these scientific
colormaps.  The ``cmcrameri`` package is an optional dependency; when it is
not installed a sensible matplotlib fallback is used instead.

Reference: https://www.fabiocrameri.ch/colourmaps/
"""

from __future__ import annotations

import matplotlib.pyplot as plt

try:
    import cmcrameri.cm as cmc  # type: ignore[import-untyped]

    _CMC_AVAILABLE = True
except ImportError:  # pragma: no cover
    cmc = None
    _CMC_AVAILABLE = False

# Recommended Crameri colormaps for common seismic uses
_DEFAULT_CRAMERI = "batlow"
_SEQUENTIAL_CRAMERI = "batlow"
_DIVERGING_CRAMERI = "vik"

# Matplotlib fallbacks when cmcrameri is not installed
_FALLBACK: dict[str, str] = {
    "batlow": "viridis",
    "vik": "RdBu_r",
    "roma": "coolwarm",
    "berlin": "RdBu_r",
    "lapaz": "cividis",
    "lajolla": "YlOrBr",
    "devon": "cividis",
    "oslo": "Greys",
    "davos": "inferno",
    "hawaii": "plasma",
    "bamako": "magma",
    "acton": "Purples",
}


def get_crameri_cmap(name: str | None = None, *, reverse: bool = False):
    """Return a Crameri colourmap, falling back to matplotlib if unavailable.

    Parameters
    ----------
    name:
        Crameri colourmap name (e.g. ``"batlow"``, ``"vik"``).
        ``None`` returns the project default (``"batlow"``).
    reverse:
        If ``True``, return the reversed variant (``"_r"``).

    Returns
    -------
    matplotlib.colors.Colormap
    """
    if name is None:
        name = _DEFAULT_CRAMERI

    if _CMC_AVAILABLE and cmc is not None:
        cmap_name = f"{name}_r" if reverse else name
        cmap = getattr(cmc, cmap_name, None)
        if cmap is not None:
            return cmap

    # Fallback
    mpl_name = _FALLBACK.get(name, "viridis")
    if reverse:
        mpl_name += "_r"
    return plt.colormaps[mpl_name]
