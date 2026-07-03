"""Nomenclature, LaTeX variable mapping, and text helpers.

Every label that appears on an axis, legend entry, or title should be
routed through :func:`format_label` (for LaTeX variable substitution)
and :func:`to_title_case` (for axis labels).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# LaTeX variable map (raw identifier → LaTeX string)
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    "f_ratio": r"$f_0^N$",
    "abs_TF": r"${\left|TF\right|}_0^N$",
    "log(abs_TF)": r"${log\left|TF\right|}_0^N$",
    "a_HV": r"$a_{hv}$",
    "Height": r"$H$",
    "r_H": r"$r_{h}$",
    "rH": r"$r_{h}$",
    "CoV": r"$\text{CoV}$",
    "CV": r"$\text{CoV}$",
}

# Regex compiled once: matches any key surrounded by word boundaries
_LABEL_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(LABEL_MAP, key=len, reverse=True))
)

# ---------------------------------------------------------------------------
# Channel → Recorder rename
# ---------------------------------------------------------------------------
_CHANNEL_RE = re.compile(r"\bChannel\b", re.IGNORECASE)


def rename_channel(text: str) -> str:
    """Replace every occurrence of *Channel* (case-insensitive) with *Recorder*."""
    def _repl(m: re.Match) -> str:
        original = m.group(0)
        if original[0].isupper():
            return "Recorder"
        return "recorder"

    return _CHANNEL_RE.sub(_repl, text)


def format_label(text: str) -> str:
    """Substitute known variable names with their LaTeX equivalents.

    Also applies the Channel → Recorder rename.
    """
    text = rename_channel(text)
    return _LABEL_RE.sub(lambda m: LABEL_MAP[m.group(0)], text)


def to_title_case(text: str) -> str:
    """Convert *text* to Title Case while preserving LaTeX fragments.

    Any substring already wrapped in ``$...$`` is left untouched.
    """
    parts = re.split(r"(\$[^$]+\$)", text)
    result: list[str] = []
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            result.append(part)
        else:
            result.append(part.title())
    return "".join(result)
