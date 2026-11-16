"""CLI script to build, benchmark, and persist the multi-field inverted index."""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from joblens.ir.bm25 import BM25FRanker
from joblens.ir.inverted_index import InvertedIndex
from joblens.ir.tokenizer import JobTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="JobLens Index Builder & Benchmark")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT_DIR / "data" / "processed"),
        help="Path to partitioned Parquet directory or sample Parquet file (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "index"),
        help="Directory to persist index JSON files (default: data/index)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="Run performance benchmark (build time, peak memory, query latency)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    print("=" * 70)
    print(" JobLens Inverted Index Builder & Benchmark")
    print("=" * 70)
    print(f"Loading data from: {input_path} ...")

    t_load_start = time.time()
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} job records in {time.time() - t_load_start:.2f}s.")

    print("\nInitializing JobTokenizer (Porter stemmer with protected skills bypass)...")
    tokenizer = JobTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)

    print(f"Indexing {len(df):,} documents across fields [title, skills, description]...", flush=True)
    tracemalloc.start()
    t_index_start = time.time()

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        doc_id = str(row["job_id"])
        title = str(row["title"])
        skills = row["skills"] if isinstance(row["skills"], list) else []
        desc = str(row["description"])

        meta = {
            "title": title,
            "company": row.get("company"),
            "company_id": row.get("company_id"),
            "location": str(row.get("location", "")),
            "country": str(row.get("country", "")),
            "work_mode": str(row.get("work_mode", "On-site")),
            "job_type": str(row.get("job_type", "Full-time")),
            "role_family": str(row.get("role_family", "Operations & Other")),
            "min_experience": float(row["min_experience"]) if pd.notnull(row.get("min_experience")) else None,
            "max_experience": float(row["max_experience"]) if pd.notnull(row.get("max_experience")) else None,
            "salary_min": float(row["salary_min"]) if pd.notnull(row.get("salary_min")) else None,
            "salary_max": float(row["salary_max"]) if pd.notnull(row.get("salary_max")) else None,
            "salary_currency": str(row.get("salary_currency")) if pd.notnull(row.get("salary_currency")) else None,
            "posted_date": str(row.get("posted_date", "")),
        }
        index.add_document(doc_id, title, skills, desc, metadata=meta)

        if idx % 5000 == 0 or idx == len(df):
            print(f"  Processed {idx:,}/{len(df):,} documents ({time.time() - t_index_start:.1f}s)...", flush=True)

    index.finalize()
    build_time = time.time() - t_index_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024 * 1024)
    print(f"Index built in {build_time:.2f}s ({len(df)/max(build_time, 0.001):.0f} docs/sec).")
    print(f"Peak memory usage (tracemalloc): {peak_mem_mb:.2f} MB (current: {current_mem / (1024 * 1024):.2f} MB).")

    print(f"\nPersisting index to disk at {out_dir} ...")
    t_save_start = time.time()
    index.save(out_dir)
    save_time = time.time() - t_save_start

    meta_size = (out_dir / "index_meta.json").stat().st_size / (1024 * 1024)
    post_size = (out_dir / "index_postings.json").stat().st_size / (1024 * 1024)
    total_size_mb = meta_size + post_size
    print(f"Index persisted in {save_time:.2f}s. Total disk size: {total_size_mb:.2f} MB (meta: {meta_size:.2f} MB, postings: {post_size:.2f} MB).")

    if args.benchmark:
        print("\n" + "-" * 70)
        print(" Benchmark: BM25F Query Latency")
        print("-" * 70)
        ranker = BM25FRanker(index)
        test_queries = [
            "machine learning engineer python pytorch",
            "senior backend developer go postgresql kubernetes",
            "cloud devops engineer terraform aws",
            "product manager agile scrum",
            "cybersecurity analyst penetration testing network security",
            "data scientist sql pandas tableau",
            "nurse practitioner healthcare",
            "sales marketing manager",
        ]
        latencies = []
        for q in test_queries:
            t0 = time.perf_counter()
            results = ranker.search(q, top_k=10)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dur_ms)
            top_title = results[0].metadata.get("title") if results else "No result"
            print(f"  Query: '{q[:35]}...' -> Top: '{top_title}' ({dur_ms:.2f} ms)")

        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\nMean Query Latency: {avg_latency:.2f} ms | P95 Latency: {p95_latency:.2f} ms")

        # Save benchmark report to data/index_benchmark.json
        bench_data = {
            "total_documents": index.N,
            "build_time_seconds": round(build_time, 2),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "index_disk_size_mb": round(total_size_mb, 2),
            "mean_query_latency_ms": round(avg_latency, 2),
            "p95_query_latency_ms": round(p95_latency, 2),
        }
        with open(ROOT_DIR / "data" / "index_benchmark.json", "w", encoding="utf-8") as f:
            import json
            json.dump(bench_data, f, indent=2)

    print("=" * 70)
    print("Index build and benchmark completed successfully!")


if __name__ == "__main__":
    main()
