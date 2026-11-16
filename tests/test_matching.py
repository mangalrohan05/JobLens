"""Tests for candidate matching, explainable breakdowns, skill gap aggregation, and SQLite persistence."""

from pathlib import Path
import pytest

from joblens.ir.bm25 import BM25FRanker
from joblens.ir.inverted_index import InvertedIndex
from joblens.ir.tokenizer import JobTokenizer
from joblens.matching.matcher import CandidateMatcher
from joblens.storage.db import JobLensDatabase


@pytest.fixture
def mock_index() -> InvertedIndex:
    tok = JobTokenizer()
    index = InvertedIndex(tokenizer=tok)

    # Job 1: Junior Python Dev (Austin, TX, 0-2 yrs)
    index.add_document(
        doc_id="job_junior_py",
        title="Junior Python Developer",
        skills=["python", "sql", "git"],
        description="Entry level python software developer role.",
        metadata={
            "title": "Junior Python Developer",
            "company": "TechCorp",
            "location": "Austin, TX",
            "country": "US",
            "work_mode": "On-site",
            "min_experience": 0.0,
            "max_experience": 2.0,
            "skills": ["python", "sql", "git"],
        },
    )

    # Job 2: Senior Python & Cloud Architect (Remote, 5+ yrs)
    index.add_document(
        doc_id="job_senior_cloud",
        title="Senior Python Cloud Architect",
        skills=["python", "aws", "docker", "kubernetes", "terraform"],
        description="Lead cloud infrastructure and backend microservices.",
        metadata={
            "title": "Senior Python Cloud Architect",
            "company": "CloudScale",
            "location": "Remote",
            "country": "US",
            "work_mode": "Remote",
            "min_experience": 5.0,
            "max_experience": None,
            "skills": ["python", "aws", "docker", "kubernetes", "terraform"],
        },
    )

    # Job 3: London On-site Frontend (London, UK, 3-5 yrs)
    index.add_document(
        doc_id="job_london_frontend",
        title="Frontend React Engineer",
        skills=["react", "typescript", "css"],
        description="Frontend web development in central London.",
        metadata={
            "title": "Frontend React Engineer",
            "company": "FinTech London",
            "location": "London, UK",
            "country": "GB",
            "work_mode": "On-site",
            "min_experience": 3.0,
            "max_experience": 5.0,
            "skills": ["react", "typescript", "css"],
        },
    )

    # Job 4: Office Coordinator with zero extracted skills
    index.add_document(
        doc_id="job_zero_skills",
        title="Administrative Office Coordinator",
        skills=[],
        description="General office coordination, filing, and team scheduling.",
        metadata={
            "title": "Administrative Office Coordinator",
            "company": "City Services",
            "location": "Austin, TX, US",
            "country": "US",
            "work_mode": "On-site",
            "min_experience": 1.0,
            "max_experience": 3.0,
            "skills": [],
        },
    )

    index.finalize()
    return index


def test_standard_candidate_match(mock_index: InvertedIndex):
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": ["python", "aws", "docker"],
        "years_experience": 6.0,
        "preferred_roles": ["Cloud Architect", "Senior Engineer"],
        "preferred_locations": ["US", "Remote"],
        "preferred_work_modes": ["Remote"],
    }

    matches = matcher.match_candidate(profile, top_k=5)
    assert len(matches) > 0

    top_match = matches[0]
    assert top_match.job_id == "job_senior_cloud"
    assert "python" in top_match.matched_skills
    assert "aws" in top_match.matched_skills
    assert "terraform" in top_match.missing_skills
    assert "kubernetes" in top_match.missing_skills
    assert "Direct experience match" in top_match.experience_fit_note
    assert "Matches preference for Remote work" in top_match.location_fit_note
    assert top_match.match_score > 0.60


def test_edge_case_no_skills(mock_index: InvertedIndex):
    """Edge Case: Candidate profile with no skills entered."""
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": [],
        "years_experience": 1.0,
        "preferred_roles": ["Junior Developer", "Python"],
        "preferred_locations": ["Austin, TX"],
        "preferred_work_modes": ["On-site"],
    }

    matches = matcher.match_candidate(profile, top_k=5)
    assert len(matches) > 0

    # Junior job should rank high based on experience and location fit despite 0 candidate skills
    assert any(m.job_id == "job_junior_py" for m in matches)
    for m in matches:
        assert m.skill_score == 0.0 or len(m.matched_skills) == 0
        assert m.match_score >= 0.0


def test_edge_case_over_qualified(mock_index: InvertedIndex):
    """Edge Case: Candidate with 14 years experience applying for a 0-2 year junior role."""
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": ["python", "sql", "git"],
        "years_experience": 14.0,
        "preferred_roles": ["Python Developer"],
        "preferred_locations": ["Austin, TX"],
        "preferred_work_modes": ["On-site"],
    }

    matches = matcher.match_candidate(profile, top_k=5)
    junior_match = next((m for m in matches if m.job_id == "job_junior_py"), None)
    assert junior_match is not None

    # Over-qualification should produce a note and experience score decay (< 1.0)
    assert "over-qualified" in junior_match.experience_fit_note.lower()
    assert junior_match.experience_score < 1.0


def test_edge_case_location_mismatch(mock_index: InvertedIndex):
    """Edge Case: Candidate preferred US Remote only, job is London On-site."""
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": ["react", "typescript"],
        "years_experience": 4.0,
        "preferred_roles": ["Frontend Engineer"],
        "preferred_locations": ["Austin, TX"],
        "preferred_work_modes": ["Remote"],
    }

    matches = matcher.match_candidate(profile, top_k=5)
    london_match = next((m for m in matches if m.job_id == "job_london_frontend"), None)
    if london_match:
        assert london_match.location_score <= 0.25
        assert "mismatch" in london_match.location_fit_note.lower()


def test_edge_case_zero_required_skills(mock_index: InvertedIndex):
    """Edge Case: Job posting has zero extracted skills; use neutral 0.5 and rebalance weights in explanation."""
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": ["python", "sql"],
        "years_experience": 2.0,
        "preferred_roles": ["Office Coordinator"],
        "preferred_locations": ["Austin, TX, US"],
        "preferred_work_modes": ["On-site"],
    }

    matches = matcher.match_candidate(profile, top_k=5)
    zero_skill_match = next((m for m in matches if m.job_id == "job_zero_skills"), None)
    assert zero_skill_match is not None

    # Check neutral 0.5 skill score and empty skill sets
    assert zero_skill_match.skill_score == 0.50
    assert len(zero_skill_match.matched_skills) == 0
    assert len(zero_skill_match.missing_skills) == 0

    # Check that explanation mentions neutral 0.5 and rebalanced weights
    assert "neutral 0.5 skill fit" in zero_skill_match.rationale
    assert "rebalanced" in zero_skill_match.rationale


def test_city_and_state_location_fit(mock_index: InvertedIndex):
    """Verify city and state matching in LocFit."""
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": ["python"],
        "years_experience": 1.0,
        "preferred_roles": ["Developer"],
        "preferred_cities": ["Austin"],
        "preferred_states": ["TX"],
        "preferred_work_modes": ["On-site"],
    }

    matches = matcher.match_candidate(profile, top_k=5)
    austin_match = next((m for m in matches if m.job_id == "job_junior_py"), None)
    assert austin_match is not None
    assert austin_match.location_score == 1.0


def test_aggregate_skill_gap(mock_index: InvertedIndex):
    matcher = CandidateMatcher(mock_index)

    profile = {
        "skills": ["python"],
        "years_experience": 5.0,
        "preferred_roles": ["Engineer"],
    }
    matches = matcher.match_candidate(profile, top_k=10)
    gap_summary = matcher.aggregate_skill_gap(matches, top_n=5)

    assert isinstance(gap_summary, list)
    if gap_summary:
        assert "skill" in gap_summary[0]
        assert "missing_in_jobs" in gap_summary[0]
        assert "percentage" in gap_summary[0]


def test_sqlite_persistence(tmp_path):
    """Verify profile, saved jobs, and search history operations in SQLite."""
    db_path = tmp_path / "test_joblens.db"
    db = JobLensDatabase(db_path)

    # 1. Profile save & load
    db.save_profile(
        user_id="user_42",
        skills=["python", "docker", "aws"],
        years_experience=3.5,
        preferred_roles=["Backend Engineer"],
        preferred_locations=["Remote", "US"],
        preferred_work_modes=["Remote"],
        min_salary=120000,
    )
    prof = db.get_profile("user_42")
    assert prof is not None
    assert prof["user_id"] == "user_42"
    assert "docker" in prof["skills"]
    assert prof["years_experience"] == 3.5

    # 2. Save & retrieve job
    saved = db.save_job(
        user_id="user_42",
        job_id="job_001",
        title="Senior Python Engineer",
        company="Databricks",
        location="Remote",
        match_score=94.5,
        notes="High match for distributed systems.",
    )
    assert saved is True
    # Attempting to save duplicate returns False
    assert db.save_job(user_id="user_42", job_id="job_001", title="Duplicate") is False

    saved_list = db.get_saved_jobs("user_42")
    assert len(saved_list) == 1
    assert saved_list[0]["job_id"] == "job_001"
    assert saved_list[0]["company"] == "Databricks"

    # Remove saved job
    assert db.remove_saved_job("user_42", "job_001") is True
    assert len(db.get_saved_jobs("user_42")) == 0

    # 3. Search history
    db.record_search("user_42", "python engineer", filters={"work_mode": "Remote"}, result_count=15)
    history = db.get_search_history("user_42")
    assert len(history) == 1
    assert history[0]["query"] == "python engineer"
    assert history[0]["result_count"] == 15
