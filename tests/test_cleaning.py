"""Tests for cleaning, normalization, title, and location standardization."""

import pytest
from joblens.cleaning.normalizer import (
    strip_html,
    normalize_whitespace,
    clean_text,
    standardize_title,
    standardize_location,
)


def test_strip_html():
    raw = "<p>Join our team as a <b>Python Engineer</b>!<br>Requires experience with &amp; knowledge of SQL.</p>"
    cleaned = strip_html(raw)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "Python Engineer" in cleaned
    assert "SQL" in cleaned


def test_normalize_whitespace_and_unicode():
    raw = "Senior   Developer\u200b with “smart quotes” and – dashes.\n\n\n\nRequires 5 years."
    normalized = normalize_whitespace(raw)
    assert "   " not in normalized
    assert "\u200b" not in normalized
    assert '"smart quotes"' in normalized
    assert " - dashes." in normalized
    assert "\n\n\n" not in normalized


def test_standardize_title():
    assert standardize_title("Sr. SWE [Remote]") == "Senior Software Engineer"
    assert standardize_title("Jr. Python Dev (Full Time)") == "Junior Python Developer"
    assert standardize_title("Lead ML / AI Engineer") == "Lead ML / AI Engineer"
    assert standardize_title("DevOps Engineer (m/w/d)") == "DevOps Engineer"
    assert standardize_title("SDE II - Cloud Platforms") == "Software Development Engineer II - Cloud Platforms"


def test_standardize_location_us():
    loc = standardize_location("San Francisco, CA, United States")
    assert loc["country"] == "US"
    assert loc["modality"] == "On-site"
    assert loc["is_remote"] is False


def test_standardize_location_remote():
    loc = standardize_location("Remote - US")
    assert loc["is_remote"] is True
    assert loc["modality"] == "Remote"
    assert loc["country"] == "US"


def test_standardize_location_international():
    loc_uk = standardize_location("London, UK")
    assert loc_uk["country"] == "GB"

    loc_de = standardize_location("Berlin, Germany (Hybrid)")
    assert loc_de["country"] == "DE"
    assert loc_de["modality"] == "Hybrid"

    loc_in = standardize_location("Bengaluru, India")
    assert loc_in["country"] == "IN"
