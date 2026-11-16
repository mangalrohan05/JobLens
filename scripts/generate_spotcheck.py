"""Generate qrels_spotcheck.md with real top-5 retrieved postings per system for 5 queries."""

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
from joblens.ir.baseline import KeywordBaselineRanker
from joblens.ir.bm25 import BM25FRanker
from joblens.ir.inverted_index import InvertedIndex
from joblens.ir.tokenizer import JobTokenizer


def main():
    tokenizer = JobTokenizer()
    index = InvertedIndex.load(ROOT_DIR / "data" / "index", tokenizer=tokenizer)
    bm25 = BM25FRanker(index)
    baseline = KeywordBaselineRanker(index, match_mode="OR")

    with open(ROOT_DIR / "data" / "evaluation" / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    # 5 queries: 3 tech, 2 non-tech
    spotcheck_ids = ["q01", "q04", "q08", "q16", "q21"]
    spotcheck_queries = [q for q in queries if q["query_id"] in spotcheck_ids]

    qrels_df = pd.read_csv(ROOT_DIR / "data" / "evaluation" / "qrels.csv")
    qrels_map = {(row["query_id"], str(row["doc_id"])): row["relevance"] for _, row in qrels_df.iterrows()}

    lines = []
    lines.append("# JobLens Qrels Spot-Check Form")
    lines.append("")
    lines.append("This document contains the top-5 retrieved postings for 5 diverse evaluation queries (3 technical, 2 non-technical) across two retrieval systems:")
    lines.append("1. **Keyword Baseline Ranker** (raw term-frequency count in OR mode)")
    lines.append("2. **BM25F Default Ranker** (field-weighted Robertson-Spärck Jones BM25)")
    lines.append("")
    lines.append("**Instructions for Project Owner / Evaluator**:")
    lines.append("- Review each retrieved posting title, company, and location.")
    lines.append("- Record your independent manual assessment in the **Manual Verdict** column (`[X] Rel` or `[X] Irrel`).")
    lines.append("- Compare your verdict with the **Rule Grade** (2 = Highly Relevant, 1 = Partially Relevant, 0 = Irrelevant) to evaluate how well rule-based labeling reflects human judgment.")
    lines.append("")

    for q in spotcheck_queries:
        qid = q["query_id"]
        qtext = q["query"]
        qfam = q["role_family"]
        skills_str = ", ".join(q["core_skills"])
        lines.append(f"## Query [{qid}]: \"{qtext}\"")
        lines.append(f"**Target Role Family**: {qfam}  ")
        lines.append(f"**Core Skills**: {skills_str}  ")
        lines.append("")

        # BM25 Top 5
        lines.append("### System 1: BM25F (Default Multi-Field)")
        lines.append("| Rank | Doc ID | Job Title | Company | Location | BM25 Score | Rule Grade | Manual Verdict |")
        lines.append("| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |")
        bm25_res = bm25.search(qtext, top_k=5)
        for rank, r in enumerate(bm25_res, 1):
            meta = r.metadata
            title = str(meta.get("title", "N/A")).replace("|", "-")
            company = str(meta.get("company") or "N/A").replace("|", "-")
            loc = str(meta.get("location", "N/A")).replace("|", "-")
            grade = qrels_map.get((qid, str(r.doc_id)), 0)
            lines.append(f"| {rank} | `{r.doc_id}` | {title} | {company} | {loc} | {r.score:.2f} | **{grade}** | [ ] Rel / [ ] Irrel |")
        lines.append("")

        # Baseline Top 5
        lines.append("### System 2: Keyword Baseline Ranker (Term Count)")
        lines.append("| Rank | Doc ID | Job Title | Company | Location | Term Count | Rule Grade | Manual Verdict |")
        lines.append("| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |")
        base_res = baseline.search(qtext, top_k=5)
        for rank, r in enumerate(base_res, 1):
            meta = r.metadata
            title = str(meta.get("title", "N/A")).replace("|", "-")
            company = str(meta.get("company") or "N/A").replace("|", "-")
            loc = str(meta.get("location", "N/A")).replace("|", "-")
            grade = qrels_map.get((qid, str(r.doc_id)), 0)
            lines.append(f"| {rank} | `{r.doc_id}` | {title} | {company} | {loc} | {r.score:.0f} | **{grade}** | [ ] Rel / [ ] Irrel |")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_file = ROOT_DIR / "data" / "evaluation" / "qrels_spotcheck.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Successfully generated {out_file}")


if __name__ == "__main__":
    main()
