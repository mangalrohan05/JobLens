"""Unit tests for real-data edge cases discovered during real LinkedIn dataset ingestion."""

from pathlib import Path
import pandas as pd
import pytest

from joblens.cleaning.experience import parse_experience_with_source
from joblens.cleaning.normalizer import (
    extract_company_name,
    infer_role_family,
    standardize_location,
    standardize_title,
)
from joblens.ingestion.loader import detect_and_map_columns, parse_timestamp_or_date, iter_raw_chunks


def test_epoch_timestamp_parsing():
    """Edge Case 1: Millisecond epoch integers from LinkedIn metadata."""
    series = pd.Series([1692881234000, 1713500000000, "2024-04-19", None])
    parsed = parse_timestamp_or_date(series)
    assert parsed.iloc[0] == "2023-08-24"
    assert parsed.iloc[1] == "2024-04-19"
    assert parsed.iloc[2] == "2024-04-19"
    assert parsed.iloc[3] == "2024-01-01"


def test_us_location_disambiguation_with_state():
    """Edge Case 2: Cities named after foreign nations/regions (e.g. Ontario CA, Mexico NY, Berlin NJ)."""
    loc_ontario = standardize_location("Ontario, CA")
    assert loc_ontario["country"] == "US"  # California, not Canada

    loc_mexico = standardize_location("Mexico, NY")
    assert loc_mexico["country"] == "US"  # New York, not Mexico

    loc_berlin = standardize_location("Berlin, NJ")
    assert loc_berlin["country"] == "US"  # New Jersey, not Germany

    loc_london_oh = standardize_location("London, OH")
    assert loc_london_oh["country"] == "US"  # Ohio, not UK

    loc_real_uk = standardize_location("London, England, United Kingdom")
    assert loc_real_uk["country"] == "GB"

    loc_real_ca = standardize_location("Toronto, ON, Canada")
    assert loc_real_ca["country"] == "CA"


def test_us_metropolitan_areas_parsing():
    """Edge Case 3: Regional metropolitan strings lacking explicit state/country tokens."""
    for metro in [
        "San Francisco Bay Area",
        "New York City Metropolitan Area",
        "Greater Seattle Area",
        "Dallas-Fort Worth Metroplex",
        "Greater Chicago Area",
    ]:
        res = standardize_location(metro)
        assert res["country"] == "US"
        assert res["work_mode"] in ("On-site", "Remote")


def test_company_derivation_from_domain_and_url():
    """Edge Case 4: Numeric company_id fallback to posting_domain branding."""
    # When company_raw is a numeric ID, extract brand from domain
    company_name = extract_company_name(
        company_raw=437225,
        posting_domain="careers.iherb.com",
    )
    assert company_name == "Iherb"

    # When domain is a generic portal or ATS, company must be None (no placeholders)
    company_fallback = extract_company_name(
        company_raw=553718,
        posting_domain="www.click2apply.net",
    )
    assert company_fallback is None

    # When company_name is an explicit string, preserve it
    company_explicit = extract_company_name(
        company_raw="Google LLC",
        posting_domain="careers.google.com",
    )
    assert company_explicit == "Google LLC"


def test_experience_source_attribution():
    """Edge Case 5: Separate tracking for description-derived vs title-fallback requirements."""
    # 1. Description contains explicit years -> source must be 'description'
    min_exp, max_exp, src = parse_experience_with_source(
        description="Requires 3-5 years of industry experience.",
        title="Software Engineer",
    )
    assert min_exp == 3.0
    assert max_exp == 5.0
    assert src == "description"

    # 2. Description lacks years, title has 'Senior' -> source must be 'title'
    min_exp, max_exp, src = parse_experience_with_source(
        description="Exciting engineering role building search systems.",
        title="Senior Software Engineer",
    )
    assert min_exp == 5.0
    assert max_exp is None
    assert src == "title"

    # 3. Formatted level has 'Entry level' -> source must be 'title'
    min_exp, max_exp, src = parse_experience_with_source(
        description="Join our team as an analyst.",
        title="Analyst",
        experience_level_raw="Entry level",
    )
    assert min_exp == 0.0
    assert max_exp == 2.0
    assert src == "title"

    # 4. Neither has experience indicators -> source must be None
    min_exp, max_exp, src = parse_experience_with_source(
        description="Great workplace with collaborative culture.",
        title="General Assistant",
    )
    assert min_exp is None
    assert max_exp is None
    assert src is None


def test_chunked_streaming_and_split_handling(tmp_path):
    """Edge Case 6: Memory-efficient chunked streaming across multiple split CSV files."""
    part1 = tmp_path / "split_part1.csv"
    part2 = tmp_path / "split_part2.csv"

    df1 = pd.DataFrame({
        "job_id": ["job_1", "job_2"],
        "title": ["Software Engineer", "Data Scientist"],
        "location": ["Austin, TX", "London, UK"],
        "description": ["Python development", "PyTorch machine learning"],
    })
    df2 = pd.DataFrame({
        "job_id": ["job_3", "job_4"],
        "title": ["Product Manager", "DevOps Engineer"],
        "location": ["Remote, US", "Berlin, Germany"],
        "description": ["Product leadership", "AWS and Terraform"],
    })
    df1.to_csv(part1, index=False)
    df2.to_csv(part2, index=False)

    chunks = list(iter_raw_chunks(tmp_path, chunksize=1))
    assert len(chunks) == 4
    total_records = sum(len(c) for c in chunks)
    assert total_records == 4
