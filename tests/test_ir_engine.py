"""Tests for the IR engine: tokenizer, inverted index, hand-computed BM25F, and keyword baseline."""

import math
from pathlib import Path
import pytest

from joblens.ir.baseline import KeywordBaselineRanker
from joblens.ir.bm25 import BM25FRanker
from joblens.ir.inverted_index import InvertedIndex
from joblens.ir.tokenizer import JobTokenizer


@pytest.fixture
def tokenizer() -> JobTokenizer:
    return JobTokenizer()


def test_protected_tokens_survive_stemming(tokenizer: JobTokenizer):
    """Verify that taxonomy skills and special punctuation tokens bypass stemming."""
    text = "Senior developer using pandas, kubernetes, aws, c++, c#, .net, node.js, and ci/cd for engineering."
    tokens = tokenizer.tokenize(text)

    # Protected tokens must survive unstemmed
    assert "pandas" in tokens
    assert "kubernetes" in tokens
    assert "aws" in tokens
    assert "c++" in tokens
    assert "c#" in tokens
    assert ".net" in tokens
    assert "node.js" in tokens
    assert "ci/cd" in tokens

    # Non-protected English words must be stemmed
    assert "develop" in tokens  # 'developer' -> 'develop'
    assert "use" in tokens      # 'using' -> 'use'
    assert "engineering" in tokens  # taxonomy skill word is protected


def test_hand_computed_bm25_corpus():
    """Verify BM25F calculation against an exact manual calculation on a 3-document corpus."""
    tok = JobTokenizer()
    index = InvertedIndex(tokenizer=tok)

    # Doc 1: Python web engineer
    index.add_document(
        doc_id="doc_1",
        title="Python Engineer",
        skills=["python"],
        description="Python backend web development",
        metadata={"title": "Python Engineer", "location": "Austin, TX"},
    )
    # Doc 2: Java developer (no python)
    index.add_document(
        doc_id="doc_2",
        title="Java Developer",
        skills=["java"],
        description="Java spring backend services",
        metadata={"title": "Java Developer", "location": "Austin, TX"},
    )
    # Doc 3: Python data scientist
    index.add_document(
        doc_id="doc_3",
        title="Python Data Scientist",
        skills=["python", "machine learning"],
        description="Python pandas data analysis",
        metadata={"title": "Python Data Scientist", "location": "Remote"},
    )
    index.finalize()

    assert index.N == 3
    # 'python' appears in doc_1 and doc_3 -> df = 2
    df_python = index.get_document_frequency("python")
    assert df_python == 2

    # Theoretical IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
    expected_idf = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))  # ln(1 + 1.5/2.5) = ln(1.6)
    ranker = BM25FRanker(
        index,
        k1=1.2,
        w_title=3.0,
        w_skills=2.0,
        w_desc=1.0,
        b_title=0.3,
        b_skills=0.0,
        b_desc=0.75,
    )
    computed_idf = ranker.compute_idf("python")
    assert abs(computed_idf - expected_idf) < 1e-9

    # Query: "python"
    results = ranker.search("python", top_k=10)
    assert len(results) == 2  # doc_1 and doc_3 match; doc_2 has 0 score

    # Check that doc_1 and doc_3 both receive positive relevance scores
    assert results[0].score > 0.0
    assert results[1].score > 0.0
    assert "python" in results[0].term_scores

    # Compute manual score for doc_1:
    # title tokens: ['python', 'engin'] -> len=2, tf=1
    # skills tokens: ['python'] -> len=1, tf=1
    # desc tokens: ['python', 'backend', 'web', 'develop'] -> len=4, tf=1
    avg_title_len = (2 + 2 + 3) / 3.0  # doc1: 2, doc2: 2, doc3: 3
    avg_skills_len = (1 + 1 + 3) / 3.0
    avg_desc_len = (4 + 4 + 4) / 3.0

    b_title, b_skills, b_desc = 0.3, 0.0, 0.75
    norm_title = 1.0 - b_title + b_title * (2 / avg_title_len)
    norm_skills = 1.0 - b_skills + b_skills * (1 / avg_skills_len)  # exactly 1.0
    norm_desc = 1.0 - b_desc + b_desc * (4 / avg_desc_len)

    tilde_tf_doc1 = 3.0 * (1 / norm_title) + 2.0 * (1 / norm_skills) + 1.0 * (1 / norm_desc)
    expected_doc1_score = expected_idf * (tilde_tf_doc1 / (1.2 + tilde_tf_doc1))

    # Match doc_1 score from ranker
    doc_1_res = next(r for r in results if r.doc_id == "doc_1")
    assert abs(doc_1_res.score - expected_doc1_score) < 1e-4


def test_index_persistence_roundtrip(tmp_path):
    """Verify that InvertedIndex saves to JSON and reloads with 100% fidelity."""
    tok = JobTokenizer()
    index = InvertedIndex(tokenizer=tok)
    index.add_document("job_1", "Senior React Developer", ["react", "typescript"], "Building user interfaces with React.")
    index.add_document("job_2", "Backend Go Engineer", ["go", "docker"], "Building microservices in Go.")
    index.finalize()

    save_dir = tmp_path / "test_index"
    index.save(save_dir)

    # Reload from disk
    loaded_index = InvertedIndex.load(save_dir, tokenizer=tok)
    assert loaded_index.N == 2
    assert loaded_index.doc_freq["react"] == 1
    assert loaded_index.doc_lengths["title"]["job_1"] == index.doc_lengths["title"]["job_1"]

    ranker_orig = BM25FRanker(index)
    ranker_loaded = BM25FRanker(loaded_index)

    orig_results = ranker_orig.search("react developer")
    loaded_results = ranker_loaded.search("react developer")

    assert len(orig_results) == len(loaded_results)
    assert orig_results[0].doc_id == loaded_results[0].doc_id
    assert abs(orig_results[0].score - loaded_results[0].score) < 1e-6


def test_bm25_filters():
    """Verify pre/post filters for location, work mode, and experience."""
    tok = JobTokenizer()
    index = InvertedIndex(tokenizer=tok)
    index.add_document(
        "doc_remote", "Cloud Architect", ["aws"], "AWS cloud architecture.",
        metadata={"location": "Austin, TX, US", "country": "US", "work_mode": "Remote", "min_experience": 5.0, "max_experience": None},
    )
    index.add_document(
        "doc_onsite", "Cloud Architect", ["aws"], "AWS cloud engineering on-site.",
        metadata={"location": "London, GB", "country": "GB", "work_mode": "On-site", "min_experience": 3.0, "max_experience": 5.0},
    )
    index.finalize()

    ranker = BM25FRanker(index)

    # Filter: Remote only
    remote_only = ranker.search("cloud architect", filters={"work_mode": "Remote"})
    assert len(remote_only) == 1
    assert remote_only[0].doc_id == "doc_remote"

    # Filter: Country GB only
    gb_only = ranker.search("cloud architect", filters={"country": "GB"})
    assert len(gb_only) == 1
    assert gb_only[0].doc_id == "doc_onsite"

    # Filter: State TX only
    tx_only = ranker.search("cloud architect", filters={"state": "TX"})
    assert len(tx_only) == 1
    assert tx_only[0].doc_id == "doc_remote"

    # Filter: City Austin only
    austin_only = ranker.search("cloud architect", filters={"city": "Austin"})
    assert len(austin_only) == 1
    assert austin_only[0].doc_id == "doc_remote"

    # Filter: Candidate experience = 4 years
    # doc_remote requires min 5 yrs (candidate has 4 -> excluded)
    # doc_onsite requires min 3 yrs, max 5 yrs (candidate has 4 -> included)
    exp_filtered = ranker.search("cloud architect", filters={"candidate_experience": 4.0})
    assert len(exp_filtered) == 1
    assert exp_filtered[0].doc_id == "doc_onsite"


def test_baseline_keyword_ranker():
    """Verify KeywordBaselineRanker in AND and OR modes."""
    tok = JobTokenizer()
    index = InvertedIndex(tokenizer=tok)
    index.add_document("doc_a", "Python Developer", ["python"], "Python Python programming")
    index.add_document("doc_b", "Python React Developer", ["python", "react"], "Python and React development")
    index.add_document("doc_c", "React Developer", ["react"], "React frontend only")
    index.finalize()

    baseline = KeywordBaselineRanker(index)

    # OR mode: matches all docs with python or react
    res_or = baseline.search("python react", match_mode="OR")
    assert len(res_or) == 3

    # AND mode: only doc_b has both python and react
    res_and = baseline.search("python react", match_mode="AND")
    assert len(res_and) == 1
    assert res_and[0].doc_id == "doc_b"

    # doc_a has multiple occurrences of python (higher tf score)
    res_py = baseline.search("python")
    assert res_py[0].doc_id == "doc_a"
    assert res_py[0].score > res_py[1].score
