"""Data ingestion, schema detection, and chunked streaming for raw job postings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import pandas as pd

# Mapping of raw column variants to standardized canonical JobLens schema
COLUMN_MAPPINGS = {
    "title": ["title", "job_title", "role", "position", "job_name"],
    "company": ["company_name", "company", "employer", "organization"],
    "company_id": ["company_id"],
    "location": ["location", "job_location", "city", "place", "work_location"],
    "description": ["description", "job_description", "job_details", "text", "body", "details"],
    "posted_date": ["listed_time", "original_listed_time", "posted_date", "date", "date_posted", "created_at"],
    "job_type": ["formatted_work_type", "work_type", "employment_type", "type", "schedule_type"],
    "salary_min": ["min_salary", "salary_min", "salary_from", "pay_min"],
    "salary_max": ["max_salary", "salary_max", "salary_to", "pay_max"],
    "salary_currency": ["currency", "salary_currency", "pay_currency"],
    "job_id": ["job_id", "id", "job_link", "url", "job_posting_url"],
    "remote_allowed": ["remote_allowed", "is_remote"],
    "experience_level_raw": ["formatted_experience_level", "experience_level", "seniority"],
    "posting_domain": ["posting_domain"],
    "application_url": ["application_url"],
}

REQUIRED_CANONICAL_COLUMNS = ["title", "location", "description"]


def parse_timestamp_or_date(series: pd.Series) -> pd.Series:
    """Parse dates from either ISO strings or millisecond epoch integers."""
    # Convert numeric epoch milliseconds if present
    numeric_s = pd.to_numeric(series, errors="coerce")
    is_epoch_ms = numeric_s.notnull() & (numeric_s > 1e11)

    result = pd.Series(index=series.index, dtype="object")
    if is_epoch_ms.any():
        result[is_epoch_ms] = pd.to_datetime(numeric_s[is_epoch_ms], unit="ms", errors="coerce").dt.strftime("%Y-%m-%d")

    non_epoch = ~is_epoch_ms
    if non_epoch.any():
        result[non_epoch] = pd.to_datetime(series[non_epoch], errors="coerce").dt.strftime("%Y-%m-%d")

    return result.fillna("2024-01-01")


def detect_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map flexible source column names to standardized canonical column names."""
    df_cols = {str(col).lower().strip(): col for col in df.columns}
    rename_map = {}

    for canonical, candidates in COLUMN_MAPPINGS.items():
        for candidate in candidates:
            if candidate.lower() in df_cols:
                rename_map[df_cols[candidate.lower()]] = canonical
                break

    mapped_df = df.rename(columns=rename_map)

    # Validate required canonical columns
    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in mapped_df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required canonical columns: {missing}. Available columns: {list(df.columns)}"
        )

    # Ensure optional columns exist
    optional_defaults = {
        "job_id": None,
        "company": None,
        "company_id": None,
        "posted_date": "2024-01-01",
        "job_type": "Full-time",
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "remote_allowed": None,
        "experience_level_raw": None,
        "posting_domain": None,
        "application_url": None,
    }
    for col, default_val in optional_defaults.items():
        if col not in mapped_df.columns:
            mapped_df[col] = default_val

    # Standardize date column
    mapped_df["posted_date"] = parse_timestamp_or_date(mapped_df["posted_date"])

    # Auto-generate job_id if null
    if mapped_df["job_id"].isnull().any() or (mapped_df["job_id"] == "").any():
        mapped_df["job_id"] = [f"job_{idx:06d}" for idx in range(1, len(mapped_df) + 1)]
    else:
        mapped_df["job_id"] = mapped_df["job_id"].astype(str)

    return mapped_df


def find_raw_files(source_path_or_dir: str | Path) -> List[Path]:
    """Discover all raw data files from a given path or directory."""
    path = Path(source_path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        return [path]

    # Search directory for supported formats
    supported_extensions = ["*.csv", "*.parquet", "*.pq", "*.json", "*.jsonl"]
    files = []
    for ext in supported_extensions:
        files.extend(sorted(path.glob(ext)))

    if not files:
        raise FileNotFoundError(f"No supported data files found in directory: {path}")

    return files


def iter_raw_chunks(
    source_path_or_dir: str | Path,
    chunksize: int = 10000,
) -> Generator[pd.DataFrame, None, None]:
    """Stream raw datasets in memory-efficient chunks, supporting split files.

    Args:
        source_path_or_dir: Path to single file or directory containing split CSVs.
        chunksize: Number of records to yield per chunk.

    Yields:
        Standardized chunk DataFrame.
    """
    files = find_raw_files(source_path_or_dir)

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False, encoding="utf-8", on_bad_lines="skip"):
                yield detect_and_map_columns(chunk)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(file_path)
            for i in range(0, len(df), chunksize):
                yield detect_and_map_columns(df.iloc[i : i + chunksize])
        elif suffix in (".json", ".jsonl"):
            try:
                df = pd.read_json(file_path, lines=suffix == ".jsonl")
                for i in range(0, len(df), chunksize):
                    yield detect_and_map_columns(df.iloc[i : i + chunksize])
            except Exception:
                with open(file_path, "r", encoding="utf-8") as f:
                    batch = []
                    for line in f:
                        if line.strip():
                            batch.append(json.loads(line))
                            if len(batch) >= chunksize:
                                yield detect_and_map_columns(pd.DataFrame(batch))
                                batch = []
                    if batch:
                        yield detect_and_map_columns(pd.DataFrame(batch))


def load_raw_dataset(source_path_or_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate all raw data from file or directory."""
    chunks = list(iter_raw_chunks(source_path_or_dir, chunksize=20000))
    if not chunks:
        raise ValueError(f"No records could be loaded from {source_path_or_dir}")
    return pd.concat(chunks, ignore_index=True)
