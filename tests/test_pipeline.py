"""Tests for raw ingestion, pipeline execution, deduplication, and Parquet round-trips."""

from pathlib import Path
import pandas as pd
import pytest

from joblens.cleaning.pipeline import JobDataPipeline
from joblens.ingestion.loader import detect_and_map_columns
from joblens.skills.extractor import SkillExtractor


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "job_title": "Sr. SWE [Remote]",
            "company_name": "Google LLC",
            "job_location": "San Francisco, CA, USA",
            "job_description": "<p>Need 5+ years of experience in Python and AWS.</p>",
            "posted_date": "2024-03-15",
            "employment_type": "Full-time",
            "salary_min": 150000,
            "salary_max": 200000,
        },
        # Duplicate record
        {
            "job_title": "Sr. SWE [Remote]",
            "company_name": "Google LLC",
            "job_location": "San Francisco, CA, USA",
            "job_description": "<p>Need 5+ years of experience in Python and AWS.</p>",
            "posted_date": "2024-03-15",
            "employment_type": "Full-time",
            "salary_min": 150000,
            "salary_max": 200000,
        },
        # Record with missing title
        {
            "job_title": "",
            "company_name": "Ghost Corp",
            "job_location": "Remote",
            "job_description": "Valid description",
        },
        # Record with British location and different role
        {
            "job_title": "Data Scientist",
            "company_name": "DeepMind",
            "job_location": "London, UK",
            "job_description": "Seeking ML engineer with 3-5 years experience in PyTorch and SQL.",
            "posted_date": "2023-11-20",
        },
    ])


def test_column_mapping(sample_raw_df):
    mapped = detect_and_map_columns(sample_raw_df)
    assert "title" in mapped.columns
    assert "company" in mapped.columns
    assert "location" in mapped.columns
    assert "description" in mapped.columns
    assert "job_id" in mapped.columns


def test_pipeline_cleaning_and_deduplication(sample_raw_df):
    mapped = detect_and_map_columns(sample_raw_df)
    pipeline = JobDataPipeline()
    clean_df, report = pipeline.run(mapped)

    # 4 raw rows -> 1 empty title dropped -> 1 duplicate dropped -> 2 clean rows
    assert report.raw_row_count == 4
    assert report.null_rows_dropped == 1
    assert report.duplicates_dropped == 1
    assert report.final_row_count == 2

    # Check standardized columns
    titles = list(clean_df["title"])
    assert "Senior Software Engineer" in titles
    assert "Data Scientist" in titles

    # Check country standardization
    countries = set(clean_df["country"])
    assert "US" in countries
    assert "GB" in countries

    # Check skill extraction
    google_row = clean_df[clean_df["company"] == "Google LLC"].iloc[0]
    assert "python" in google_row["skills"]
    assert "aws" in google_row["skills"]
    assert google_row["min_experience"] == 5.0

    deepmind_row = clean_df[clean_df["company"] == "DeepMind"].iloc[0]
    assert "pytorch" in deepmind_row["skills"]
    assert "sql" in deepmind_row["skills"]
    assert deepmind_row["min_experience"] == 3.0
    assert deepmind_row["max_experience"] == 5.0


def test_parquet_roundtrip(tmp_path, sample_raw_df):
    mapped = detect_and_map_columns(sample_raw_df)
    pipeline = JobDataPipeline()
    clean_df, _ = pipeline.run(mapped)

    # Write partitioned parquet
    out_dir = tmp_path / "processed_test"
    pipeline.save_partitioned_parquet(clean_df, out_dir, partition_cols=["country"])

    # Read back with pandas
    read_df = pd.read_parquet(out_dir)
    assert len(read_df) == 2
    assert "title" in read_df.columns
    assert "country" in read_df.columns

    # Test sample file save
    sample_file = tmp_path / "sample.parquet"
    pipeline.save_sample_parquet(clean_df, sample_file, max_rows=1)
    read_sample = pd.read_parquet(sample_file)
    assert len(read_sample) == 1
