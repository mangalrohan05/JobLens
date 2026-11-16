"""Experience range parsing from job posting text, titles, and seniority indicators."""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Range: "3-5 years", "3 to 5 years", "3 - 5 yrs"
RE_RANGE = re.compile(
    r"\b(\d{1,2}(?:\.\d)?)\s*(?:-|–|to)\s*(\d{1,2}(?:\.\d)?)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+relevant|\s+professional|\s+work)?\s*(?:experience)?\b",
    re.IGNORECASE,
)

# Minimum: "5+ years", "at least 5 years", "minimum 5 years", "5 years minimum"
RE_MIN = re.compile(
    r"(?:(?:at\s+least|minimum|min\.?)\s+(\d{1,2}(?:\.\d)?)\s*(?:years?|yrs?)|(\d{1,2}(?:\.\d)?)\s*\+\s*(?:years?|yrs?))(?:\s+of)?(?:\s+relevant|\s+professional|\s+work)?\s*(?:experience)?\b",
    re.IGNORECASE,
)

# Single number: "5 years of experience", "requires 3 years experience"
RE_EXACT = re.compile(
    r"\b(\d{1,2}(?:\.\d)?)\s*(?:years?|yrs?)\s+of\s+(?:relevant|hands-on|industry|professional|work)?\s*experience\b",
    re.IGNORECASE,
)

# Maximum / entry: "up to 3 years", "maximum 2 years"
RE_MAX = re.compile(
    r"\b(?:up\s+to|maximum|max\.?)\s+(\d{1,2}(?:\.\d)?)\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

# No experience: "no experience required", "0 years experience"
RE_NO_EXP = re.compile(
    r"\b(?:no\s+experience\s+required|0\s*(?:years?|yrs?)\s*(?:of)?\s*experience|zero\s+experience)\b",
    re.IGNORECASE,
)

# Title and formatted level keywords for fallback
LEVEL_INDICATORS = [
    (re.compile(r"\b(?:intern|internship|trainee|apprentice)\b", re.IGNORECASE), (0.0, 1.0)),
    (re.compile(r"\b(?:entry\s*level|new\s+grad|junior|jr\.?|associate)\b", re.IGNORECASE), (0.0, 2.0)),
    (re.compile(r"\b(?:mid[\s\-]level|intermediate)\b", re.IGNORECASE), (2.0, 5.0)),
    (re.compile(r"\b(?:mid[\s\-]senior\s*level)\b", re.IGNORECASE), (4.0, 8.0)),
    (re.compile(r"\b(?:principal|director|fellow|vp|head\s+of|executive)\b", re.IGNORECASE), (8.0, None)),
    (re.compile(r"\b(?:staff|lead|architect)\b", re.IGNORECASE), (6.0, None)),
    (re.compile(r"\b(?:senior|sr\.?)\b", re.IGNORECASE), (5.0, None)),
]


def parse_experience_with_source(
    description: str | None,
    title: str | None = None,
    experience_level_raw: str | None = None,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse minimum and maximum years of experience and record derivation source.

    Returns:
        Tuple of (min_years, max_years, derivation_source),
        where derivation_source is 'description', 'title', or None.
    """
    if description and isinstance(description, str):
        # 1. Check explicit "no experience"
        if RE_NO_EXP.search(description):
            return (0.0, 0.0, "description")

        # 2. Check explicit range "3-5 years"
        range_match = RE_RANGE.search(description)
        if range_match:
            try:
                min_val = float(range_match.group(1))
                max_val = float(range_match.group(2))
                if min_val <= 30 and max_val <= 30 and min_val <= max_val:
                    return (min_val, max_val, "description")
            except (ValueError, TypeError):
                pass

        # 3. Check minimum "5+ years", "at least 3 years"
        min_match = RE_MIN.search(description)
        if min_match:
            try:
                val = float(min_match.group(1) or min_match.group(2))
                if 0 <= val <= 30:
                    return (val, None, "description")
            except (ValueError, TypeError):
                pass

        # 4. Check "up to X years"
        max_match = RE_MAX.search(description)
        if max_match:
            try:
                val = float(max_match.group(1))
                if 0 <= val <= 30:
                    return (0.0, val, "description")
            except (ValueError, TypeError):
                pass

        # 5. Check exact "X years of experience"
        exact_match = RE_EXACT.search(description)
        if exact_match:
            try:
                val = float(exact_match.group(1))
                if 0 <= val <= 30:
                    return (val, None, "description")
            except (ValueError, TypeError):
                pass

    # Fallback to title or explicit formatted level
    combined_fallback = f"{title or ''} {experience_level_raw or ''}"
    if combined_fallback.strip():
        for pattern, (t_min, t_max) in LEVEL_INDICATORS:
            if pattern.search(combined_fallback):
                return (t_min, t_max, "title")

    return (None, None, None)


def parse_experience_range(
    description: str | None, title: str | None = None
) -> Tuple[Optional[float], Optional[float]]:
    """Convenience wrapper returning (min_years, max_years) for backwards compatibility."""
    min_exp, max_exp, _ = parse_experience_with_source(description, title=title)
    return (min_exp, max_exp)
