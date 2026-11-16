"""End-to-end data cleaning, normalization, skill extraction, and quality reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from joblens.cleaning.experience import parse_experience_with_source
from joblens.cleaning.normalizer import (
    clean_text,
    extract_company_name,
    infer_role_family,
    standardize_location,
    standardize_title,
)
from joblens.skills.extractor import SkillExtractor


@dataclass
class DataQualityReport:
    """Detailed data quality metrics before and after cleaning."""

    raw_row_count: int
    null_rows_dropped: int
    duplicates_dropped: int
    final_row_count: int
    null_rates_before: Dict[str, float]
    null_rates_after: Dict[str, float]
    total_skills_extracted: int
    avg_skills_per_job: float
    description_derived_exp_count: int
    description_derived_exp_rate: float
    title_fallback_exp_count: int
    title_fallback_exp_rate: float
    total_experience_parsed_count: int
    total_experience_parsed_rate: float
    country_distribution: Dict[str, int]
    work_mode_distribution: Dict[str, int]
    role_family_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print_summary(self) -> str:
        """Format a human-readable markdown summary."""
        lines = [
            "# JobLens Data Quality Report (Real Dataset)",
            "",
            "## 1. Dataset Scale & Retention",
            f"- **Raw Rows Ingested:** {self.raw_row_count:,}",
            f"- **Null / Corrupt Rows Dropped:** {self.null_rows_dropped:,} ({self.null_rows_dropped / max(self.raw_row_count, 1):.2%})",
            f"- **Duplicate Rows Dropped:** {self.duplicates_dropped:,} ({self.duplicates_dropped / max(self.raw_row_count, 1):.2%})",
            f"- **Final Clean Rows:** {self.final_row_count:,} ({self.final_row_count / max(self.raw_row_count, 1):.1%})",
            "",
            "## 2. Feature Extraction & Parsing Rates",
            f"- **Total Extracted Skill Instances:** {self.total_skills_extracted:,}",
            f"- **Average Extracted Skills per Posting:** {self.avg_skills_per_job:.2f}",
            f"- **Experience Parsed (Description-derived):** {self.description_derived_exp_count:,} ({self.description_derived_exp_rate:.1%})",
            f"- **Experience Parsed (Title / Seniority Fallback):** {self.title_fallback_exp_count:,} ({self.title_fallback_exp_rate:.1%})",
            f"- **Total Experience Requirement Parsed:** {self.total_experience_parsed_count:,} ({self.total_experience_parsed_rate:.1%})",
            "",
            "## 3. ISO Country Distribution (Partitioning Keys)",
        ]
        for country, count in sorted(self.country_distribution.items(), key=lambda x: x[1], reverse=True)[:8]:
            lines.append(f"- **{country}:** {count:,} postings ({count / max(self.final_row_count, 1):.1%})")

        lines.extend([
            "",
            "## 4. Work Mode Distribution",
        ])
        for mode, count in sorted(self.work_mode_distribution.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{mode}:** {count:,} postings ({count / max(self.final_row_count, 1):.1%})")

        lines.extend([
            "",
            "## 5. Role Family Distribution",
        ])
        for role, count in sorted(self.role_family_distribution.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- **{role}:** {count:,} postings ({count / max(self.final_row_count, 1):.1%})")

        lines.extend([
            "",
            "## 6. Null Rates (Before vs. After Cleaning)",
            "| Field | Null Rate Before | Null Rate After |",
            "|---|---|---|",
        ])
        all_cols = sorted(set(self.null_rates_before.keys()).union(set(self.null_rates_after.keys())))
        for col in all_cols:
            before_val = f"{self.null_rates_before.get(col, 0.0):.1%}" if col in self.null_rates_before else "N/A"
            after_val = f"{self.null_rates_after.get(col, 0.0):.1%}" if col in self.null_rates_after else "N/A"
            lines.append(f"| `{col}` | {before_val} | {after_val} |")

        return "\n".join(lines)


class JobDataPipeline:
    """Orchestrates data ingestion, cleaning, normalization, enrichment, and Parquet export."""

    def __init__(self, extractor: Optional[SkillExtractor] = None) -> None:
        self.extractor = extractor or SkillExtractor()

    def process_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Clean and enrich a batch of rows."""
        cleaned_records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            title_clean = standardize_title(str(row["title"]))
            desc_clean = clean_text(str(row["description"]))

            # Company and company_id
            company_raw = row.get("company")
            comp_id_raw = row.get("company_id")
            company_id = str(comp_id_raw).strip() if pd.notnull(comp_id_raw) and str(comp_id_raw).strip() != "" and str(comp_id_raw) != "nan" else None
            company_clean = extract_company_name(
                company_raw,
                posting_domain=row.get("posting_domain"),
                application_url=row.get("application_url"),
            )

            loc_raw = str(row.get("location", ""))
            remote_val = row.get("remote_allowed")
            work_type_raw = str(row.get("job_type", ""))
            loc_info = standardize_location(loc_raw, remote_allowed=remote_val, work_type_raw=work_type_raw)

            exp_level_raw = str(row.get("experience_level_raw", "")) if pd.notnull(row.get("experience_level_raw")) else None
            min_exp, max_exp, exp_source = parse_experience_with_source(
                desc_clean, title=title_clean, experience_level_raw=exp_level_raw
            )

            # Role family categorization
            role_family = infer_role_family(title_clean)

            # Extract canonical skills
            full_text = f"{title_clean} {desc_clean}"
            skills = self.extractor.extract_skills(full_text)

            # Date and year parsing
            raw_date = row.get("posted_date", None)
            parsed_date = pd.to_datetime(raw_date, errors="coerce")
            posting_year = int(parsed_date.year) if pd.notnull(parsed_date) else 2024
            posted_date_str = parsed_date.strftime("%Y-%m-%d") if pd.notnull(parsed_date) else "2024-01-01"

            # Salary normalization - do not impute currency if salary is null
            salary_min = float(row["salary_min"]) if pd.notnull(row.get("salary_min")) and str(row.get("salary_min")).replace(".", "", 1).isdigit() else None
            salary_max = float(row["salary_max"]) if pd.notnull(row.get("salary_max")) and str(row.get("salary_max")).replace(".", "", 1).isdigit() else None
            if salary_min is not None or salary_max is not None:
                curr_val = str(row.get("salary_currency", "")).strip().upper()
                salary_currency = curr_val if curr_val and curr_val != "NAN" else "USD"
            else:
                salary_currency = None

            # Composite deduplication key
            dedup_key = f"{title_clean.lower()}|{str(company_clean or '').lower()}|{loc_info['location'].lower()}|{desc_clean[:100].lower()}"

            cleaned_records.append({
                "job_id": str(row.get("job_id", "")),
                "title": title_clean,
                "company": company_clean,
                "company_id": company_id,
                "location": loc_info["location"],
                "country": loc_info["country"],
                "work_mode": loc_info["work_mode"],
                "is_remote": loc_info["is_remote"],
                "job_type": str(row.get("job_type", "Full-time")),
                "role_family": role_family,
                "min_experience": min_exp,
                "max_experience": max_exp,
                "experience_source": exp_source,
                "skills": skills,
                "skills_count": len(skills),
                "skills_text": " ".join(skills),
                "description": desc_clean,
                "posted_date": posted_date_str,
                "posting_year": posting_year,
                "salary_currency": salary_currency,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "_dedup_key": dedup_key,
            })

        return cleaned_records

    def run(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Execute full cleaning and enrichment pipeline."""
        raw_count = len(raw_df)
        null_rates_before = (raw_df.isnull().mean()).to_dict()

        # Step 1: Filter rows missing essential fields
        valid_mask = (
            raw_df["title"].notnull()
            & (raw_df["title"].astype(str).str.strip() != "")
            & raw_df["description"].notnull()
            & (raw_df["description"].astype(str).str.strip() != "")
        )
        filtered_df = raw_df[valid_mask].copy()
        null_rows_dropped = raw_count - len(filtered_df)

        # Step 2: Clean and enrich records
        cleaned_records = self.process_records(filtered_df)
        processed_df = pd.DataFrame(cleaned_records)

        # Step 3: Deduplication
        pre_dedup_count = len(processed_df)
        processed_df = processed_df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
        duplicates_dropped = pre_dedup_count - len(processed_df)

        final_count = len(processed_df)
        null_rates_after = (processed_df.isnull().mean()).to_dict()

        # Step 4: Quality statistics calculation
        total_skills = int(processed_df["skills_count"].sum()) if final_count > 0 else 0
        avg_skills = float(processed_df["skills_count"].mean()) if final_count > 0 else 0.0

        desc_exp_count = int((processed_df["experience_source"] == "description").sum())
        title_exp_count = int((processed_df["experience_source"] == "title").sum())
        total_exp_count = desc_exp_count + title_exp_count

        desc_exp_rate = desc_exp_count / max(final_count, 1)
        title_exp_rate = title_exp_count / max(final_count, 1)
        total_exp_rate = total_exp_count / max(final_count, 1)

        country_dist = processed_df["country"].value_counts().to_dict()
        work_mode_dist = processed_df["work_mode"].value_counts().to_dict()
        role_dist = processed_df["role_family"].value_counts().to_dict()

        report = DataQualityReport(
            raw_row_count=raw_count,
            null_rows_dropped=null_rows_dropped,
            duplicates_dropped=duplicates_dropped,
            final_row_count=final_count,
            null_rates_before=null_rates_before,
            null_rates_after=null_rates_after,
            total_skills_extracted=total_skills,
            avg_skills_per_job=avg_skills,
            description_derived_exp_count=desc_exp_count,
            description_derived_exp_rate=desc_exp_rate,
            title_fallback_exp_count=title_exp_count,
            title_fallback_exp_rate=title_exp_rate,
            total_experience_parsed_count=total_exp_count,
            total_experience_parsed_rate=total_exp_rate,
            country_distribution=country_dist,
            work_mode_distribution=work_mode_dist,
            role_family_distribution=role_dist,
        )

        return processed_df, report

    @staticmethod
    def save_partitioned_parquet(
        df: pd.DataFrame,
        output_dir: str | Path,
        partition_cols: Optional[List[str]] = None,
    ) -> Path:
        """Write DataFrame to partitioned Parquet files by ISO country code."""
        import shutil
        out_path = Path(output_dir)
        if out_path.exists():
            shutil.rmtree(out_path, ignore_errors=True)
        out_path.mkdir(parents=True, exist_ok=True)
        cols_to_partition = partition_cols or ["country"]

        export_df = df.copy()
        # Drop internal experience_source column before persistence
        if "experience_source" in export_df.columns:
            export_df = export_df.drop(columns=["experience_source"])

        table = pa.Table.from_pandas(export_df, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(out_path),
            partition_cols=cols_to_partition,
            use_dictionary=True,
            compression="SNAPPY",
        )
        return out_path

    @staticmethod
    def save_stratified_sample_parquet(
        df: pd.DataFrame,
        output_file: str | Path,
        max_rows: int = 3500,
        min_per_family: int = 200,
        random_state: int = 42,
    ) -> Path:
        """Save a compact deployable Parquet file (<50MB) with per-role-family minimums.

        Ensures low-frequency families like Data & AI, Cybersecurity, and Cloud & DevOps
        are well represented in the deployable sample.
        """
        out_file = Path(output_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        export_df = df.copy()
        if "experience_source" in export_df.columns:
            export_df = export_df.drop(columns=["experience_source"])

        if len(export_df) <= max_rows:
            sample_df = export_df
        else:
            selected_indices = set()

            # Pass 1: Ensure minimum representation for each role family
            for family, group in export_df.groupby("role_family"):
                take_n = min(len(group), min_per_family)
                sampled_group = group.sample(n=take_n, random_state=random_state)
                selected_indices.update(sampled_group.index)

            # Pass 2: Fill remaining quota proportionally across (country, role_family)
            remaining_quota = max_rows - len(selected_indices)
            if remaining_quota > 0:
                unselected = export_df.loc[~export_df.index.isin(selected_indices)]
                group_cols = ["country", "role_family"]
                frac = min(1.0, remaining_quota / len(unselected))
                extra_sample = (
                    unselected.groupby(group_cols, group_keys=False)
                    .apply(lambda g: g.sample(n=max(1, int(np.ceil(len(g) * frac))), random_state=random_state))
                )
                selected_indices.update(extra_sample.index[:remaining_quota])

            sample_df = export_df.loc[list(selected_indices)]
            if len(sample_df) > max_rows:
                sample_df = sample_df.sample(n=max_rows, random_state=random_state)
            sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        sample_df.to_parquet(out_file, index=False, compression="SNAPPY")
        return out_file

    save_sample_parquet = save_stratified_sample_parquet
