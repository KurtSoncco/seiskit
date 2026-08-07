"""Appendix 2 / IM–peak upstream material for the full paper.

This folder documents the pipeline that produces ``join_master.h5`` and hosts
Nature-style appendix figure scripts. Campaign arrays and the ``window_max``
peak extractor live on Box under ``complete/peak_analysis/``; TF and IM math
live in ``seiskit/``.

Pipeline
--------
1. Transfer functions — ``seiskit.ttf.TTF`` (+ 1D layered TF in ``seiskit.theory``)
2. Intensity measures — ``seiskit.intensity_measures`` (PGA, PSA/Sa, Arias \(I_a\))
3. Peak picking — Box ``peak_analysis`` ``window_max`` / midpoint window policy
   (convergence and reliability plots under ``plots/stability/``)
4. 1D normalization — \(\chi^N = \chi / \chi^{\mathrm{1D}}\) → ratios in
   ``join_master.h5`` (columns ``f_ratio``, ``abs_TF_ratio``, ``PGA_ratio``,
   ``PSA_ratio``, ``Ia_ratio``)

PSA period convention in this campaign: spectral ordinate at the 1D fundamental
(equivalently \(T_0 = 4H/V_{s1}\) for a uniform layer).

Outputs
-------
Appendix figures → ``complete/full_paper/figures/appendix_im/``
"""

from __future__ import annotations

from pathlib import Path

# Re-export seiskit IM helpers for appendix notebooks / scripts.
try:
    from seiskit.intensity_measures import arias_intensity, compute_sa, pga
except ImportError:  # pragma: no cover
    arias_intensity = compute_sa = pga = None  # type: ignore

APPENDIX_DIR = Path(__file__).resolve().parent
