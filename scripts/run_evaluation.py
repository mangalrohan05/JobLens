"""Comprehensive IR evaluation script: pooling, rule-based qrels generation,

metric benchmarking (P@5, P@10, MAP, NDCG@10), parameter grid search, and plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from joblens.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
)
from joblens.ir.baseline import KeywordBaselineRanker
from joblens.ir.bm25 import BM25FRanker
from joblens.ir.inverted_index import InvertedIndex
from joblens.ir.tokenizer import JobTokenizer


# 30 Queries: 15 Tech + 15 Non-Tech across diverse role families
QUERY_DEFINITIONS: List[Dict[str, Any]] = [
    # --- Tech Queries (15) ---
    {
        "query_id": "q01",
        "query": "machine learning engineer python pytorch",
        "role_family": "Data & AI",
        "target_title_regex": r"(?i)\b(machine learning|ml engineer|deep learning|ai engineer)\b",
        "partial_title_regex": r"(?i)\b(data scientist|algorithm engineer|nlp|computer vision)\b",
        "core_skills": ["python", "pytorch", "machine learning", "tensorflow"],
    },
    {
        "query_id": "q02",
        "query": "senior data scientist sql machine learning",
        "role_family": "Data & AI",
        "target_title_regex": r"(?i)\b(data scientist|senior data scientist|lead data scientist)\b",
        "partial_title_regex": r"(?i)\b(machine learning|data analyst|bi analyst|statistician)\b",
        "core_skills": ["python", "sql", "machine learning", "statistics"],
    },
    {
        "query_id": "q03",
        "query": "data engineer spark sql etl pipeline",
        "role_family": "Data & AI",
        "target_title_regex": r"(?i)\b(data engineer|big data engineer|etl developer)\b",
        "partial_title_regex": r"(?i)\b(database engineer|bi developer|data architect|analytics engineer)\b",
        "core_skills": ["spark", "sql", "python", "etl", "airflow"],
    },
    {
        "query_id": "q04",
        "query": "cloud devops engineer terraform kubernetes aws",
        "role_family": "Cloud & DevOps",
        "target_title_regex": r"(?i)\b(devops engineer|cloud engineer|platform engineer|infrastructure engineer)\b",
        "partial_title_regex": r"(?i)\b(systems engineer|sre|cloud architect|release engineer)\b",
        "core_skills": ["terraform", "kubernetes", "aws", "docker", "ci/cd"],
    },
    {
        "query_id": "q05",
        "query": "site reliability engineer sre kubernetes linux",
        "role_family": "Cloud & DevOps",
        "target_title_regex": r"(?i)\b(site reliability engineer|sre|production engineer)\b",
        "partial_title_regex": r"(?i)\b(devops|systems administrator|linux engineer|infrastructure)\b",
        "core_skills": ["kubernetes", "linux", "docker", "python", "ansible"],
    },
    {
        "query_id": "q06",
        "query": "cybersecurity analyst incident response soc",
        "role_family": "Cybersecurity",
        "target_title_regex": r"(?i)\b(cybersecurity analyst|security analyst|soc analyst|infosec analyst)\b",
        "partial_title_regex": r"(?i)\b(security engineer|incident response|information security|threat analyst)\b",
        "core_skills": ["cybersecurity", "incident response", "siem", "firewall", "network security"],
    },
    {
        "query_id": "q07",
        "query": "penetration tester ethical hacker security",
        "role_family": "Cybersecurity",
        "target_title_regex": r"(?i)\b(penetration tester|ethical hacker|pentester|security consultant)\b",
        "partial_title_regex": r"(?i)\b(vulnerability analyst|appsec|red team|security engineer)\b",
        "core_skills": ["penetration testing", "vulnerability assessment", "wireshark", "metasploit"],
    },
    {
        "query_id": "q08",
        "query": "full stack software developer react python",
        "role_family": "Software Engineering",
        "target_title_regex": r"(?i)\b(full stack developer|full stack engineer|software developer|software engineer)\b",
        "partial_title_regex": r"(?i)\b(web developer|frontend engineer|backend engineer|application developer)\b",
        "core_skills": ["python", "react", "javascript", "typescript", "html"],
    },
    {
        "query_id": "q09",
        "query": "frontend developer react typescript css",
        "role_family": "Software Engineering",
        "target_title_regex": r"(?i)\b(frontend developer|frontend engineer|ui developer|web developer)\b",
        "partial_title_regex": r"(?i)\b(software engineer|full stack|javascript developer)\b",
        "core_skills": ["react", "typescript", "css", "html", "javascript"],
    },
    {
        "query_id": "q10",
        "query": "backend developer java spring boot microservices",
        "role_family": "Software Engineering",
        "target_title_regex": r"(?i)\b(backend developer|backend engineer|java developer|java engineer)\b",
        "partial_title_regex": r"(?i)\b(software engineer|systems developer|api developer)\b",
        "core_skills": ["java", "spring boot", "microservices", "sql", "rest api"],
    },
    {
        "query_id": "q11",
        "query": "ios mobile engineer swift swiftui",
        "role_family": "Software Engineering",
        "target_title_regex": r"(?i)\b(ios developer|ios engineer|mobile developer|mobile engineer)\b",
        "partial_title_regex": r"(?i)\b(software developer|application engineer|apple developer)\b",
        "core_skills": ["swift", "ios", "objective-c", "xcode"],
    },
    {
        "query_id": "q12",
        "query": "qa automation engineer selenium cypress",
        "role_family": "Software Engineering",
        "target_title_regex": r"(?i)\b(qa automation|quality assurance engineer|sdet|test automation engineer)\b",
        "partial_title_regex": r"(?i)\b(qa engineer|test engineer|software tester|software engineer in test)\b",
        "core_skills": ["selenium", "cypress", "python", "automation", "jira"],
    },
    {
        "query_id": "q13",
        "query": "technical product manager agile roadmap",
        "role_family": "Product & Management",
        "target_title_regex": r"(?i)\b(technical product manager|product manager|group product manager)\b",
        "partial_title_regex": r"(?i)\b(product owner|project manager|program manager)\b",
        "core_skills": ["product management", "agile", "scrum", "roadmap", "jira"],
    },
    {
        "query_id": "q14",
        "query": "scrum master agile sprint planning",
        "role_family": "Product & Management",
        "target_title_regex": r"(?i)\b(scrum master|agile coach|agile project manager)\b",
        "partial_title_regex": r"(?i)\b(project manager|product owner|delivery manager)\b",
        "core_skills": ["agile", "scrum", "kanban", "jira", "sprint planning"],
    },
    {
        "query_id": "q15",
        "query": "ui ux designer figma user research wireframing",
        "role_family": "Product & Management",
        "target_title_regex": r"(?i)\b(ui ux designer|ux designer|product designer|user experience designer)\b",
        "partial_title_regex": r"(?i)\b(graphic designer|visual designer|web designer|interaction designer)\b",
        "core_skills": ["figma", "wireframing", "prototyping", "user research", "ui design"],
    },

    # --- Non-Tech Queries (15) ---
    {
        "query_id": "q16",
        "query": "registered nurse icu healthcare patient care",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(registered nurse|rn|icu nurse|staff nurse|clinical nurse)\b",
        "partial_title_regex": r"(?i)\b(nurse|nurse supervisor|nurse practitioner|care coordinator)\b",
        "core_skills": ["critical care", "nursing", "patient care", "healthcare"],
    },
    {
        "query_id": "q17",
        "query": "nurse practitioner primary care clinical",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(nurse practitioner|fnp|aprn|family nurse practitioner)\b",
        "partial_title_regex": r"(?i)\b(registered nurse|physician assistant|clinical director)\b",
        "core_skills": ["nursing", "clinical care", "primary care", "patient care"],
    },
    {
        "query_id": "q18",
        "query": "account executive b2b enterprise sales quota",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(account executive|enterprise account executive|sales executive)\b",
        "partial_title_regex": r"(?i)\b(sales representative|business development|sales director)\b",
        "core_skills": ["b2b sales", "crm", "salesforce", "cold calling", "negotiation"],
    },
    {
        "query_id": "q19",
        "query": "business development manager sales pipeline leads",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(business development manager|bdm|director of business development)\b",
        "partial_title_regex": r"(?i)\b(sales manager|account executive|partnership manager)\b",
        "core_skills": ["business development", "lead generation", "sales pipeline", "crm"],
    },
    {
        "query_id": "q20",
        "query": "digital marketing manager seo sem campaigns content",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(digital marketing manager|marketing manager|growth marketer)\b",
        "partial_title_regex": r"(?i)\b(marketing specialist|content marketer|seo specialist|social media manager)\b",
        "core_skills": ["seo", "sem", "google analytics", "digital marketing", "content strategy"],
    },
    {
        "query_id": "q21",
        "query": "financial analyst financial modeling excel forecasting",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(financial analyst|senior financial analyst|finance manager)\b",
        "partial_title_regex": r"(?i)\b(accountant|pricing analyst|investment analyst|budget analyst)\b",
        "core_skills": ["financial modeling", "excel", "forecasting", "variance analysis", "finance"],
    },
    {
        "query_id": "q22",
        "query": "senior accountant audit general ledger reconciliation",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(senior accountant|staff accountant|accounting manager|cpa)\b",
        "partial_title_regex": r"(?i)\b(auditor|bookkeeper|financial accountant|tax accountant)\b",
        "core_skills": ["general ledger", "reconciliation", "gaap", "audit", "excel"],
    },
    {
        "query_id": "q23",
        "query": "human resources manager employee relations talent",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(human resources manager|hr manager|hr director|people partner)\b",
        "partial_title_regex": r"(?i)\b(hr generalist|hr specialist|talent manager|recruiter)\b",
        "core_skills": ["employee relations", "talent management", "hr compliance", "onboarding"],
    },
    {
        "query_id": "q24",
        "query": "talent acquisition specialist technical recruiter sourcing",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(talent acquisition|recruiter|technical recruiter|sourcing specialist)\b",
        "partial_title_regex": r"(?i)\b(hr coordinator|staffing specialist|recruiting coordinator)\b",
        "core_skills": ["sourcing", "interviewing", "talent acquisition", "recruiting", "ats"],
    },
    {
        "query_id": "q25",
        "query": "customer success manager retention onboarding churn",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(customer success manager|csm|client success manager)\b",
        "partial_title_regex": r"(?i)\b(account manager|client relations|customer support lead)\b",
        "core_skills": ["customer retention", "onboarding", "churn reduction", "client management"],
    },
    {
        "query_id": "q26",
        "query": "supply chain logistics manager inventory warehouse",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(supply chain manager|logistics manager|warehouse manager)\b",
        "partial_title_regex": r"(?i)\b(operations manager|procurement manager|inventory analyst)\b",
        "core_skills": ["supply chain", "logistics", "inventory management", "procurement"],
    },
    {
        "query_id": "q27",
        "query": "paralegal legal assistant litigation contract review",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(paralegal|legal assistant|legal specialist|contracts manager)\b",
        "partial_title_regex": r"(?i)\b(legal coordinator|compliance specialist|law clerk)\b",
        "core_skills": ["litigation", "legal research", "contract drafting", "compliance"],
    },
    {
        "query_id": "q28",
        "query": "executive administrative assistant calendar travel office",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(executive assistant|administrative assistant|executive secretary)\b",
        "partial_title_regex": r"(?i)\b(office manager|admin coordinator|personal assistant)\b",
        "core_skills": ["calendar management", "travel arrangements", "office administration", "expense reports"],
    },
    {
        "query_id": "q29",
        "query": "elementary school teacher curriculum classroom",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(teacher|elementary teacher|educator|instructor)\b",
        "partial_title_regex": r"(?i)\b(substitute teacher|tutor|curriculum coordinator|teaching assistant)\b",
        "core_skills": ["classroom management", "curriculum development", "lesson planning", "teaching"],
    },
    {
        "query_id": "q30",
        "query": "store manager retail operations merchandising sales",
        "role_family": "Operations & Other",
        "target_title_regex": r"(?i)\b(store manager|retail store manager|assistant store manager)\b",
        "partial_title_regex": r"(?i)\b(retail supervisor|department manager|merchandising manager)\b",
        "core_skills": ["retail operations", "merchandising", "store management", "inventory control"],
    },
]


def assign_graded_relevance(
    doc_meta: Dict[str, Any],
    query_def: Dict[str, Any],
) -> int:
    """Assign rule-based graded relevance (0, 1, or 2) for a query and document metadata.

    Signals:
    - Target title regex match
    - Partial title regex match
    - Core skills overlap (extracted skills or title/desc text)
    - Role family alignment
    """
    title = str(doc_meta.get("title", ""))
    role_fam = str(doc_meta.get("role_family", ""))
    skills = [s.lower() for s in doc_meta.get("skills", [])] if isinstance(doc_meta.get("skills"), list) else []

    target_title_re = query_def["target_title_regex"]
    partial_title_re = query_def["partial_title_regex"]
    target_role_fam = query_def["role_family"]
    core_skills = [s.lower() for s in query_def.get("core_skills", [])]

    is_target_title = bool(re.search(target_title_re, title))
    is_partial_title = bool(re.search(partial_title_re, title))
    is_family_match = (role_fam.lower() == target_role_fam.lower())

    # Count matching skills
    skill_matches = sum(1 for cs in core_skills if cs in skills or cs in title.lower())

    # Rule 1: Highly Relevant (Grade 2)
    # Direct title match AND (family match or at least 1 core skill match)
    if is_target_title and (is_family_match or skill_matches >= 1):
        return 2

    # Rule 2: Partially Relevant (Grade 1)
    # Either target title with no skill overlap, or partial title with family/skill signal,
    # or family match with strong core skills match (>= 2)
    if is_target_title:
        return 1
    if is_partial_title and (is_family_match or skill_matches >= 1):
        return 1
    if is_family_match and skill_matches >= 2:
        return 1

    # Rule 3: Irrelevant (Grade 0)
    return 0


def run_evaluation(
    index_path: Path,
    output_dir: Path,
    pool_depth: int = 20,
) -> None:
    """Run full evaluation suite."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" JobLens Information Retrieval Evaluation & Grid Search")
    print("=" * 70)
    print(f"Loading InvertedIndex from: {index_path} ...")
    tokenizer = JobTokenizer()
    index = InvertedIndex.load(index_path, tokenizer=tokenizer)
    print(f"Loaded index with {index.N:,} documents.")

    # 1. Save Queries JSON
    queries_path = output_dir / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(QUERY_DEFINITIONS, f, indent=2)
    print(f"Saved 30 evaluation queries to: {queries_path}")

    # 2. Save Labeling Rules CSV
    rules_csv = output_dir / "labeling_rules.csv"
    with open(rules_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "role_family", "target_title_regex", "partial_title_regex", "core_skills"])
        for qd in QUERY_DEFINITIONS:
            writer.writerow([
                qd["query_id"],
                qd["query"],
                qd["role_family"],
                qd["target_title_regex"],
                qd["partial_title_regex"],
                "; ".join(qd["core_skills"]),
            ])
    print(f"Saved labeling rules to: {rules_csv}")

    # 3. Model Configurations (All 9 Evaluated Configurations)
    models: List[Dict[str, Any]] = [
        {"name": "Keyword Baseline (Term Count)", "type": "baseline"},
        {
            "name": "BM25 (Ablation: Description Only)",
            "type": "bm25",
            "k1": 1.2,
            "b_title": 0.0,
            "b_skills": 0.0,
            "b_desc": 0.75,
            "w_title": 0.0,
            "w_skills": 0.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (Default: k1=1.2, b_desc=0.75, balanced)",
            "type": "bm25",
            "k1": 1.2,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.75,
            "w_title": 3.0,
            "w_skills": 2.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (Title-Heavy: w_title=4.0, w_skills=2.0)",
            "type": "bm25",
            "k1": 1.2,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.75,
            "w_title": 4.0,
            "w_skills": 2.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (Skills-Heavy: w_title=2.0, w_skills=3.0)",
            "type": "bm25",
            "k1": 1.2,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.75,
            "w_title": 2.0,
            "w_skills": 3.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (Low k1=0.8, b_desc=0.75)",
            "type": "bm25",
            "k1": 0.8,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.75,
            "w_title": 3.0,
            "w_skills": 2.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (High k1=1.6, b_desc=0.75)",
            "type": "bm25",
            "k1": 1.6,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.75,
            "w_title": 3.0,
            "w_skills": 2.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (Low b_desc=0.50, k1=1.2)",
            "type": "bm25",
            "k1": 1.2,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.50,
            "w_title": 3.0,
            "w_skills": 2.0,
            "w_desc": 1.0,
        },
        {
            "name": "BM25F (High b_desc=0.90, k1=1.2)",
            "type": "bm25",
            "k1": 1.2,
            "b_title": 0.3,
            "b_skills": 0.0,
            "b_desc": 0.90,
            "w_title": 3.0,
            "w_skills": 2.0,
            "w_desc": 1.0,
        },
    ]

    # Pre-instantiate rankers for pooling and evaluation
    rankers_by_model: Dict[str, Any] = {}
    for m in models:
        if m["type"] == "baseline":
            rankers_by_model[m["name"]] = KeywordBaselineRanker(index, match_mode="OR")
        else:
            rankers_by_model[m["name"]] = BM25FRanker(
                index,
                k1=m["k1"],
                w_title=m["w_title"],
                w_skills=m["w_skills"],
                w_desc=m["w_desc"],
                b_title=m["b_title"],
                b_skills=m["b_skills"],
                b_desc=m["b_desc"],
            )

    # 4. Pooling: Union of top-20 results from ALL evaluated configs plus the baseline
    print(f"\nPooling top-{pool_depth} results across ALL {len(models)} evaluated configs...")
    pooled_qrels: Dict[str, Dict[str, int]] = {}  # query_id -> {doc_id: grade}
    qrels_rows = []

    for qd in QUERY_DEFINITIONS:
        q_id = qd["query_id"]
        q_text = qd["query"]
        candidate_doc_ids: Set[str] = set()

        for m_name, ranker in rankers_by_model.items():
            res = ranker.search(q_text, top_k=pool_depth)
            for r in res:
                candidate_doc_ids.add(r.doc_id)

        pooled_qrels[q_id] = {}
        for doc_id in sorted(candidate_doc_ids):
            doc_meta = index.doc_metadata.get(doc_id, {})
            grade = assign_graded_relevance(doc_meta, qd)
            pooled_qrels[q_id][doc_id] = grade
            qrels_rows.append({"query_id": q_id, "doc_id": doc_id, "relevance": grade})

    # Save qrels CSV
    qrels_csv = output_dir / "qrels.csv"
    pd.DataFrame(qrels_rows).to_csv(qrels_csv, index=False)
    total_rel_docs = sum(1 for r in qrels_rows if r["relevance"] >= 1)
    print(f"Saved {len(qrels_rows):,} pooled qrel judgments ({total_rel_docs:,} relevant, grade >= 1) to: {qrels_csv}")

    # 5. Benchmarking Baseline vs BM25 Configurations
    binary_qrels: Dict[str, Set[str]] = {
        q_id: {d for d, g in docs.items() if g >= 1}
        for q_id, docs in pooled_qrels.items()
    }

    print("\nRunning Evaluation Benchmark across Model Configurations...")

    results_table = []

    for m in models:
        t0 = time.time()
        m_name = m["name"]
        retrieved_dict: Dict[str, List[str]] = {}
        ranker = rankers_by_model[m_name]

        for qd in QUERY_DEFINITIONS:
            q_id = qd["query_id"]
            res = ranker.search(qd["query"], top_k=20)
            retrieved_dict[q_id] = [r.doc_id for r in res]

        # Compute Metrics across 30 queries
        p5_scores = [precision_at_k(retrieved_dict[q_id], binary_qrels[q_id], k=5) for q_id in binary_qrels]
        p10_scores = [precision_at_k(retrieved_dict[q_id], binary_qrels[q_id], k=10) for q_id in binary_qrels]
        map_score = mean_average_precision(retrieved_dict, binary_qrels, k=20)
        ndcg10_scores = [
            ndcg_at_k(retrieved_dict[q_id], pooled_qrels[q_id], k=10)
            for q_id in pooled_qrels
        ]

        mean_p5 = sum(p5_scores) / len(p5_scores) if p5_scores else 0.0
        mean_p10 = sum(p10_scores) / len(p10_scores) if p10_scores else 0.0
        mean_ndcg10 = sum(ndcg10_scores) / len(ndcg10_scores) if ndcg10_scores else 0.0
        elapsed_ms = (time.time() - t0) * 1000.0 / len(QUERY_DEFINITIONS)

        results_table.append({
            "Model / Configuration": m_name,
            "P@5": round(mean_p5, 4),
            "P@10": round(mean_p10, 4),
            "MAP@20": round(map_score, 4),
            "NDCG@10": round(mean_ndcg10, 4),
            "Avg Query Latency (ms)": round(elapsed_ms, 2),
        })

    df_results = pd.DataFrame(results_table)
    results_csv = output_dir / "results.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"\nSaved evaluation metrics to: {results_csv}")
    print("\n" + df_results.to_string(index=False))

    # 5. Generate Evaluation Comparison Chart
    print("\nGenerating evaluation comparison chart...")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    model_labels = [
        "Baseline",
        "BM25 Desc-Only",
        "BM25F Default",
        "Title-Heavy",
        "Skills-Heavy",
        "k1=0.8",
        "k1=1.6",
        "b=0.50",
        "b=0.90",
    ]
    x = range(len(model_labels))
    width = 0.25

    p5_vals = [r["P@5"] for r in results_table]
    p10_vals = [r["P@10"] for r in results_table]
    map_vals = [r["MAP@20"] for r in results_table]

    ax.bar([i - width for i in x], p5_vals, width=width, label="P@5", color="#3b82f6", alpha=0.9)
    ax.bar([i for i in x], p10_vals, width=width, label="P@10", color="#10b981", alpha=0.9)
    ax.bar([i + width for i in x], map_vals, width=width, label="MAP@20", color="#8b5cf6", alpha=0.9)

    ax.set_title("JobLens IR Evaluation: BM25F vs Keyword Baseline", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    chart_path = output_dir / "eval_chart.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved evaluation chart to: {chart_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="JobLens Evaluation Suite")
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "index"),
        help="Path to inverted index directory (default: data/index)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "data" / "evaluation"),
        help="Directory to save evaluation artifacts (default: data/evaluation)",
    )
    parser.add_argument(
        "--pool-depth",
        type=int,
        default=20,
        help="Pooling depth per system (default: 20)",
    )
    args = parser.parse_args()
    run_evaluation(
        index_path=Path(args.index_dir),
        output_dir=Path(args.output_dir),
        pool_depth=args.pool_depth,
    )


if __name__ == "__main__":
    main()
