"""CLI script to build, clean, enrich, and partition the JobLens dataset from real data."""

import argparse
import json
import sys
from pathlib import Path

# Add src to sys.path if run directly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from joblens.cleaning.pipeline import JobDataPipeline
from joblens.ingestion.loader import load_raw_dataset
from joblens.skills.extractor import SkillExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="JobLens Dataset Build Pipeline (Real Data)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT_DIR / "data" / "raw"),
        help="Path to raw dataset file or directory containing split CSVs (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "processed"),
        help="Directory to save partitioned Parquet dataset",
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default=str(ROOT_DIR / "data" / "sample" / "jobs_sample.parquet"),
        help="Path to save deployable sample Parquet file",
    )
    parser.add_argument(
        "--partition-by",
        type=str,
        default="country",
        help="Column to partition by (default: country ISO code)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3500,
        help="Number of rows for deployable stratified sample (default: 3500)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Raw dataset path not found at {input_path}")
        sys.exit(1)

    print("=" * 70)
    print(" JobLens Pipeline: Ingesting Real Data & Compiling Taxonomy")
    print("=" * 70)
    print(f"Loading raw dataset from: {input_path} (supporting large/split files)...")
    raw_df = load_raw_dataset(input_path)
    print(f"Loaded {len(raw_df):,} raw records across input files.")

    print("\nInitializing SkillExtractor and compiling 375+ skills taxonomy...")
    extractor = SkillExtractor()
    pipeline = JobDataPipeline(extractor=extractor)

    print("Executing cleaning, normalization, title/location standardization, experience parsing, and skill extraction...")
    clean_df, report = pipeline.run(raw_df)

    # Save partitioned parquet
    processed_dir = Path(args.processed_dir)
    print(f"\nWriting partitioned Parquet dataset to: {processed_dir} (partitioned by: {args.partition_by})...")
    pipeline.save_partitioned_parquet(clean_df, processed_dir, partition_cols=[args.partition_by])

    # Save stratified sample parquet
    sample_file = Path(args.sample_file)
    print(f"Writing stratified sample Parquet (<50MB, country & role_family stratified) to: {sample_file} ...")
    pipeline.save_stratified_sample_parquet(clean_df, sample_file, max_rows=args.sample_size)

    # Save quality report JSON
    report_file = ROOT_DIR / "data" / "data_quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Saved data quality report to: {report_file}")

    print("\n" + "=" * 70)
    print(report.print_summary())
    print("=" * 70)
    print("Pipeline execution on real data completed successfully!")


if __name__ == "__main__":
    main()
