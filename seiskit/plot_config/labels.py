"""Nomenclature, LaTeX variable mapping, and text helpers.

Every label that appears on an axis, legend entry, or title should be
routed through :func:`format_label` (for LaTeX variable substitution).
:func:`to_title_case` remains available for callers that want it
explicitly; ``apply_style(auto_format=True)`` does not apply it.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# LaTeX variable map — regex patterns (longest first to avoid partial matches)
# Each tuple: (compiled_regex, replacement_string)
# ---------------------------------------------------------------------------
_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Longest compound terms first to prevent partial matches.
    # Modeled amplitude uses natural log; display as \ln.
    # Keep absolute-value bars on TF (|TF| → \left| TF \right|).
    (re.compile(r"\\ln\(\s*\|TF\|_0\^N\s*\)"), r"\ln(\left| TF \right|_0^N)"),
    (re.compile(r"\|TF\|_0\^N"), r"\left| TF \right|_0^N"),
    (re.compile(r"\|TF\|"), r"\left| TF \right|"),
    (
        re.compile(r"\bln\s*\(\s*abs[_ ]?TF[_ ]?(?:ratio)?\s*\)", re.IGNORECASE),
        r"$\ln(\left| TF \right|_0^N)$",
    ),
    (
        re.compile(r"\blog\s*\(\s*abs[_ ]?TF[_ ]?(?:ratio)?\s*\)", re.IGNORECASE),
        r"$\ln(\left| TF \right|_0^N)$",
    ),
    (
        re.compile(r"\blog[_ \s]+abs[_ \s]+TF(?:[_ \s]+ratio)?\b", re.IGNORECASE),
        r"$\ln(\left| TF \right|_0^N)$",
    ),
    (re.compile(r"\blog[_ ]?abs\b", re.IGNORECASE), r"$\ln(\left| TF \right|_0^N)$"),
    (re.compile(r"\babs[_ \s]?TF[_ \s]?ratio\b", re.IGNORECASE), r"$\left| TF \right|_0^N$"),
    (re.compile(r"\babs[_ \s]?TF\b", re.IGNORECASE), r"$\left| TF \right|_0^N$"),
    # 1D-normalized IM ratios (χ family)
    (re.compile(r"\bPGA[_ ]?ratio\b", re.IGNORECASE), r"$\mathrm{PGA}^N$"),
    (re.compile(r"\bPSA[_ ]?ratio\b", re.IGNORECASE), r"$\mathrm{SA}^N$"),
    (re.compile(r"\bI[_ ]?a[_ ]?ratio\b", re.IGNORECASE), r"$I_a^N$"),
    # Handle variants where $f$ is already partially LaTeX-ified
    (re.compile(r"\$f\$\s*ratio", re.IGNORECASE), r"$f_0^N$"),
    (re.compile(r"\bf[_ ]?ratio\b", re.IGNORECASE), r"$f_0^N$"),
    (re.compile(r"\ba[_ ]?HV\b", re.IGNORECASE), r"$a_{hv}$"),
    (re.compile(r"\bHeight\b"), r"$H$ (m)"),
    (re.compile(r"\br[_ ]?H\b"), r"$r_{h}$ (m)"),
    (re.compile(r"\bCoV\b", re.IGNORECASE), r"$CoV$"),
    (re.compile(r"\bCV\b"), r"$CoV$"),
    (re.compile(r"\bVs[_ ]?1\b"), r"$V_{s1}$ (m/s)"),
    (re.compile(r"\bChannel\b", re.IGNORECASE), "Recorder"),
]

# Simple key map kept for direct lookups (e.g. df column names)
LABEL_MAP: dict[str, str] = {
    "log_abs": r"$\ln(\left| TF \right|_0^N)$",
    "log(abs_TF)": r"$\ln(\left| TF \right|_0^N)$",
    "ln(abs_TF)": r"$\ln(\left| TF \right|_0^N)$",
    "abs_TF_ratio": r"$\left| TF \right|_0^N$",
    "abs_TF": r"$\left| TF \right|_0^N$",
    "f_ratio": r"$f_0^N$",
    "PGA_ratio": r"$\mathrm{PGA}^N$",
    "PSA_ratio": r"$\mathrm{SA}^N$",
    "Ia_ratio": r"$I_a^N$",
    "ln_f_ratio": r"$\ln(f_0^N)$",
    "ln_abs_TF_ratio": r"$\ln(\left| TF \right|_0^N)$",
    "ln_PGA_ratio": r"$\ln(\mathrm{PGA}^N)$",
    "ln_PSA_ratio": r"$\ln(\mathrm{SA}^N)$",
    "ln_Ia_ratio": r"$\ln(I_a^N)$",
    "a_HV": r"$a_{hv}$",
    "aHV": r"$a_{hv}$",
    "Vs1": r"$V_{s1}$ (m/s)",
    "Height": r"$H$ (m)",
    "r_H": r"$r_{h}$ (m)",
    "rH": r"$r_{h}$ (m)",
    "CoV": r"$\mathrm{CoV}$",
    "CV": r"$\mathrm{CoV}$",
}

_LN_WRAP = re.compile(r"^(?:ln|log)\s*\(\s*(.+?)\s*\)$", re.IGNORECASE)


def as_ln_label(label: str) -> str:
    """Wrap a mathtext ``$...$`` label as ``$\\ln(...)$`` (no double wrap)."""
    s = label.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        inner = s[1:-1]
    else:
        inner = s
    stripped = inner.strip()
    if stripped.startswith(r"\ln(") and stripped.endswith(")"):
        return f"${stripped}$"
    return rf"$\ln({stripped})$"


def rename_channel(text: str) -> str:
    """Replace every occurrence of *Channel* (case-insensitive) with *Recorder*."""
    return re.sub(
        r"\bChannel\b",
        lambda m: "Recorder" if m.group(0)[0].isupper() else "recorder",
        text,
        flags=re.IGNORECASE,
    )


def format_label(text: str) -> str:
    """Substitute known variable names with their LaTeX equivalents.

    Handles underscore, space, and mixed-case variants (e.g. ``"abs TF ratio"``,
    ``"abs_TF_ratio"``, ``"f_ratio"``, ``"PGA_ratio"``, ``"Ia_ratio"``).
    ``ln(...)`` / ``log(...)`` wrappers become ``$\\ln(...)$`` around the
    formatted inner label.
    """
    if not isinstance(text, str) or not text:
        return text
    wrapped = _LN_WRAP.match(text.strip())
    if wrapped:
        return as_ln_label(format_label(wrapped.group(1)))
    result = text
    for pattern, repl in _REPLACEMENTS:
        # Use lambda to avoid re.sub interpreting LaTeX backslashes as backrefs
        result = pattern.sub(lambda m, r=repl: r, result)
    return result


_UPPERCASE_WORDS = {"OLS", "GBM", "PI", "SHAP", "RMSE", "MAE", "QQ", "R2", "LGBM"}


def to_title_case(text: str) -> str:
    """Convert *text* to Title Case while preserving LaTeX, units, and acronyms.

    Substrings wrapped in ``$...$`` or ``(...)`` are left untouched
    (the latter preserves SI units like ``(m/s)``).  Known acronyms
    (OLS, GBM, SHAP, etc.) are restored to uppercase after title-casing.
    """
    parts = re.split(r"(\$[^$]+\$|\([^)]*\))", text)
    result: list[str] = []
    for part in parts:
        if (part.startswith("$") and part.endswith("$")) or (
            part.startswith("(") and part.endswith(")")
        ):
            result.append(part)
        else:
            orig_words = part.split(" ")
            titled_words = part.title().split(" ")
            restored: list[str] = []
            for orig, tw in zip(orig_words, titled_words):
                if tw.upper() in _UPPERCASE_WORDS:
                    restored.append(tw.upper())
                elif "=" in orig:
                    restored.append(orig)
                else:
                    restored.append(tw)
            result.append(" ".join(restored))
    return "".join(result)
