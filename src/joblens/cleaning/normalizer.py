"""Text normalization, HTML stripping, title, location, work_mode, and role family standardization."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

# Precompiled regex patterns for performance
RE_HTML_TAGS = re.compile(r"<[^>]+>", re.IGNORECASE)
RE_LINE_BREAK_TAGS = re.compile(r"<\s*(?:br|p|div|li|tr|h[1-6])(?:\s+[^>]*)?>", re.IGNORECASE)
RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff\u00a0]")
RE_EXCESS_WHITESPACE = re.compile(r"[ \t]+")
RE_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Title noise patterns to strip
RE_TITLE_NOISE = re.compile(
    r"\s*(?:\[[^\]]*\]|\((?:remote|hybrid|onsite|on-site|full[\s\-]time|part[\s\-]time|contract|permanent|temporary|w\/m\/d|m\/w\/d|f\/m\/d|h\/f|urgent|immediate|\d+%\s*remote)\)|\|.*$)",
    re.IGNORECASE,
)

TITLE_ABBREVIATIONS = [
    (re.compile(r"\bSr(?:\.|\b)", re.IGNORECASE), "Senior"),
    (re.compile(r"\bJr(?:\.|\b)", re.IGNORECASE), "Junior"),
    (re.compile(r"\bSWE\b", re.IGNORECASE), "Software Engineer"),
    (re.compile(r"\bSDE\s*I{1,3}\b", re.IGNORECASE), lambda m: f"Software Development Engineer {m.group(0).split()[-1].upper()}"),
    (re.compile(r"\bSDE\b", re.IGNORECASE), "Software Development Engineer"),
    (re.compile(r"\bDev\b", re.IGNORECASE), "Developer"),
    (re.compile(r"\bEng(?:\.|\b)", re.IGNORECASE), "Engineer"),
    (re.compile(r"\bMgr(?:\.|\b)", re.IGNORECASE), "Manager"),
    (re.compile(r"\bFullstack\b", re.IGNORECASE), "Full Stack"),
    (re.compile(r"\bBackend\b", re.IGNORECASE), "Backend"),
    (re.compile(r"\bFrontend\b", re.IGNORECASE), "Frontend"),
]

SPECIAL_CASING = {
    "devops": "DevOps",
    "ios": "iOS",
    "ai": "AI",
    "ml": "ML",
    "nlp": "NLP",
    "llm": "LLM",
    "aws": "AWS",
    "gcp": "GCP",
    "qa": "QA",
    "sre": "SRE",
    "ui": "UI",
    "ux": "UX",
    "ui/ux": "UI/UX",
    "bi": "BI",
    "etl": "ETL",
    "ci/cd": "CI/CD",
}

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks",
    "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny",
    "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc", "california", "new york", "texas", "washington", "massachusetts", "florida",
    "illinois", "pennsylvania", "ohio", "georgia", "north carolina", "michigan", "new jersey",
    "virginia", "colorado", "arizona", "tennessee", "indiana", "missouri", "maryland", "wisconsin",
    "minnesota", "oregon", "south carolina", "alabama", "louisiana", "kentucky", "oregon", "oklahoma",
    "connecticut", "utah", "iowa", "nevada", "arkansas", "mississippi", "kansas", "new mexico",
    "nebraska", "idaho", "west virginia", "hawaii", "new hampshire", "maine", "montana", "rhode island",
    "delaware", "south dakota", "north dakota", "alaska", "vermont", "wyoming",
}

US_METRO_AREAS = [
    "metropolitan area", "metroplex", "bay area", "greater chicago", "greater houston",
    "greater boston", "greater seattle", "greater philadelphia", "greater minneapolis",
]

# Explicit foreign country mapping
EXPLICIT_COUNTRIES = {
    "united kingdom": "GB",
    "great britain": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "uk": "GB",
    "canada": "CA",
    "germany": "DE",
    "deutschland": "DE",
    "india": "IN",
    "france": "FR",
    "australia": "AU",
    "brazil": "BR",
    "brasil": "BR",
    "mexico": "MX",
    "ireland": "IE",
    "netherlands": "NL",
    "singapore": "SG",
    "spain": "ES",
    "italy": "IT",
    "japan": "JP",
    "china": "CN",
}

# Role Family Classification Patterns
ROLE_FAMILIES = [
    (
        "Software Engineering",
        re.compile(
            r"\b(?:software|swe|sde|developer|programmer|engineer|backend|frontend|full\s*stack|mobile|ios|android|embedded|systems|qa|test|quality\s*assurance)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Data & AI",
        re.compile(
            r"\b(?:data\s*(?:scientist|engineer|analyst|analytics)|machine\s*learning|ml|ai|deep\s*learning|nlp|computer\s*vision|big\s*data|bi\s*developer|business\s*intelligence|statistician)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Cloud & DevOps",
        re.compile(
            r"\b(?:devops|cloud|sre|site\s*reliability|infrastructure|platform|sysadmin|systems\s*administrator|network\s*engineer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Cybersecurity",
        re.compile(
            r"\b(?:security|cybersecurity|infosec|soc|appsec|penetration|threat|vulnerability|cryptograph)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Product & Management",
        re.compile(
            r"\b(?:product\s*manager|project\s*manager|program\s*manager|engineering\s*manager|director|vp|head\s*of|lead|scrum\s*master|agile\s*coach)\b",
            re.IGNORECASE,
        ),
    ),
]


def strip_html(text: str | None) -> str:
    """Strip HTML tags and convert HTML entities into clean text."""
    if not text or not isinstance(text, str):
        return ""

    clean = RE_LINE_BREAK_TAGS.sub("\n", text)
    clean = RE_HTML_TAGS.sub(" ", clean)
    clean = html.unescape(clean)
    return clean


def normalize_whitespace(text: str | None) -> str:
    """Normalize unicode characters, smart quotes, zero-width chars, and whitespace."""
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = RE_ZERO_WIDTH.sub(" ", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ").replace("•", "\n- ")
    text = RE_EXCESS_WHITESPACE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = RE_EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_text(text: str | None) -> str:
    """Apply both HTML stripping and whitespace/unicode normalization."""
    return normalize_whitespace(strip_html(text))


def standardize_title(title: str | None) -> str:
    """Normalize and standardize job titles."""
    if not title or not isinstance(title, str):
        return "Untitled Role"

    title = clean_text(title)
    title = RE_TITLE_NOISE.sub("", title).strip()

    for pattern, replacement in TITLE_ABBREVIATIONS:
        if callable(replacement):
            title = pattern.sub(replacement, title)
        else:
            title = pattern.sub(replacement, title)

    words = title.split()
    standardized_words = []
    for word in words:
        clean_w = word.strip(".,;:()").lower()
        if clean_w in SPECIAL_CASING:
            prefix = word[:len(word) - len(word.lstrip(".,;:()"))]
            suffix = word[len(word.rstrip(".,;:()")):]
            standardized_words.append(f"{prefix}{SPECIAL_CASING[clean_w]}{suffix}")
        elif clean_w in {"i", "ii", "iii", "iv", "v", "vi"}:
            standardized_words.append(clean_w.upper())
        elif clean_w in {"and", "or", "of", "in", "for", "to", "with", "/"}:
            standardized_words.append(word.lower())
        else:
            standardized_words.append(word.capitalize())

    result = " ".join(standardized_words)
    if result:
        result = result[0].upper() + result[1:]
    return result or "Untitled Role"


def standardize_location(
    raw_location: str | None,
    remote_allowed: float | int | bool | None = None,
    work_type_raw: str | None = None,
) -> Dict[str, any]:
    """Parse and standardize job location into real ISO country code and work mode.

    Guarantees that country is a real ISO country code (never 'Remote').
    """
    loc_clean = clean_text(raw_location) if raw_location else "United States"
    loc_lower = loc_clean.lower()
    work_type_lower = str(work_type_raw or "").lower()

    # Determine Work Mode (Remote / Hybrid / On-site)
    is_remote_flag = bool(remote_allowed == 1.0 or remote_allowed is True or remote_allowed == 1)
    if "hybrid" in loc_lower or "hybrid" in work_type_lower:
        work_mode = "Hybrid"
    elif is_remote_flag or "remote" in loc_lower or "anywhere" in loc_lower or "work from home" in loc_lower:
        work_mode = "Remote"
    else:
        work_mode = "On-site"

    is_remote = (work_mode == "Remote")

    # Determine Real ISO Country Code
    country = "US"  # Default for this US-centric LinkedIn dataset

    # Disambiguation: check if US state indicator exists (e.g. "Ontario, CA" -> CA is California)
    tokens = [t.strip().strip(".,") for t in re.split(r"[,/|;\-]+", loc_lower) if t.strip()]

    has_us_state = any(t in US_STATES for t in tokens[1:]) if len(tokens) > 1 else False
    is_us_metro = any(metro in loc_lower for metro in US_METRO_AREAS)

    if has_us_state or is_us_metro or "united states" in loc_lower or " usa" in loc_lower or loc_lower.endswith(", us"):
        country = "US"
    else:
        # Check for genuine foreign country matches
        matched_foreign = False
        for token in reversed(tokens):
            if token in EXPLICIT_COUNTRIES:
                country = EXPLICIT_COUNTRIES[token]
                matched_foreign = True
                break

        if not matched_foreign:
            for country_name, code in EXPLICIT_COUNTRIES.items():
                if country_name in loc_lower:
                    country = code
                    matched_foreign = True
                    break

        if not matched_foreign:
            country = "US"

    return {
        "location": loc_clean,
        "country": country,
        "work_mode": work_mode,
        "modality": work_mode,  # alias for backward compatibility
        "is_remote": is_remote,
    }


def infer_role_family(title: str | None) -> str:
    """Categorize job title into major role families for stratified sampling & analytics."""
    if not title or not isinstance(title, str):
        return "Operations & Other"

    for family, pattern in ROLE_FAMILIES:
        if pattern.search(title):
            return family

    return "Operations & Other"


ATS_DOMAINS = {
    "greenhouse", "lever", "workday", "linkedin", "indeed", "smartrecruiters",
    "icims", "click2apply", "myworkdayjobs", "oraclecloud", "zoho", "hrmdirect",
    "apploi", "jobvite", "breezy", "ashbyhq", "recruitee", "taleo",
}


def extract_company_name(
    company_raw: Any,
    posting_domain: str | None = None,
    application_url: str | None = None,
) -> str | None:
    """Derive clean company name from raw company or posting_domain.

    Excludes ATS/job-board domains and avoids placeholder IDs.
    Returns None if no authentic name can be determined.
    """
    if company_raw is not None and not (isinstance(company_raw, float) and str(company_raw) == "nan"):
        comp_str = clean_text(str(company_raw))
        if comp_str and not comp_str.isdigit():
            return comp_str

    if posting_domain and isinstance(posting_domain, str) and posting_domain.strip():
        domain = posting_domain.strip().lower()
        for prefix in ["careers-", "careers.", "jobs.", "www."]:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        parts = domain.split(".")
        if parts:
            brand = parts[0]
            if brand not in ATS_DOMAINS and len(brand) > 2:
                return brand.title()

    return None
