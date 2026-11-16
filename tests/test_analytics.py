"""Tests for market analytics and insights module."""

import pandas as pd
import pytest

from joblens.analytics.market_insights import (
    get_experience_distribution,
    get_location_distribution,
    get_salary_insights,
    get_skill_cooccurrence,
    get_skills_by_role_family,
    get_top_roles,
    get_top_skills,
)


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "job_id": "1",
            "title": "Machine Learning Engineer",
            "role_family": "Data & AI",
            "skills": ["python", "pytorch", "docker", "sql"],
            "country": "US",
            "location": "San Francisco, CA, US",
            "work_mode": "Hybrid",
            "min_experience": 3.0,
            "max_experience": 5.0,
            "salary_min": 140000,
            "salary_max": 180000,
        },
        {
            "job_id": "2",
            "title": "Data Scientist",
            "role_family": "Data & AI",
            "skills": ["python", "sql", "tableau", "statistics"],
            "country": "US",
            "location": "Austin, TX, US",
            "work_mode": "Remote",
            "min_experience": 2.0,
            "max_experience": 4.0,
            "salary_min": 120000,
            "salary_max": 150000,
        },
        {
            "job_id": "3",
            "title": "DevOps Engineer",
            "role_family": "Cloud & DevOps",
            "skills": ["docker", "kubernetes", "aws", "terraform", "python"],
            "country": "GB",
            "location": "London, GB",
            "work_mode": "On-site",
            "min_experience": 5.0,
            "max_experience": 8.0,
            "salary_min": 90000,
            "salary_max": 120000,
        },
        {
            "job_id": "4",
            "title": "Senior Software Engineer",
            "role_family": "Software Engineering",
            "skills": ["python", "docker", "fastapi", "react"],
            "country": "US",
            "location": "New York, NY, US",
            "work_mode": "Hybrid",
            "min_experience": 6.0,
            "max_experience": None,
            "salary_min": None,
            "salary_max": None,
        },
        {
            "job_id": "5",
            "title": "Administrative Assistant",
            "role_family": "Operations & Other",
            "skills": ["excel", "communication"],
            "country": "US",
            "location": "Chicago, IL, US",
            "work_mode": "On-site",
            "min_experience": 1.0,
            "max_experience": 2.0,
            "salary_min": 50000,
            "salary_max": 65000,
        },
    ])


def test_get_top_roles(sample_market_df: pd.DataFrame):
    roles = get_top_roles(sample_market_df, n=3)
    assert len(roles) == 3
    assert all("role" in r and "count" in r for r in roles)

    data_roles = get_top_roles(sample_market_df, role_family="Data & AI")
    assert len(data_roles) == 2
    assert {r["role"] for r in data_roles} == {"Machine Learning Engineer", "Data Scientist"}


def test_get_top_skills(sample_market_df: pd.DataFrame):
    skills = get_top_skills(sample_market_df, n=5)
    assert len(skills) == 5
    # Python appears in 4 jobs
    assert skills[0]["skill"] == "python"
    assert skills[0]["count"] == 4
    assert skills[0]["demand_share"] == 80.0

    cloud_skills = get_top_skills(sample_market_df, role_family="Cloud & DevOps")
    assert any(s["skill"] == "kubernetes" for s in cloud_skills)


def test_get_skill_cooccurrence(sample_market_df: pd.DataFrame):
    pairs = get_skill_cooccurrence(sample_market_df, top_n_skills=10, top_n_pairs=5)
    assert len(pairs) > 0
    # python and docker co-occur in jobs 1, 3, 4
    py_docker = next((p for p in pairs if ("python" in p["pair"] and "docker" in p["pair"])), None)
    assert py_docker is not None
    assert py_docker["cooccurrence_count"] == 3


def test_get_location_distribution(sample_market_df: pd.DataFrame):
    dist = get_location_distribution(sample_market_df)
    assert len(dist["by_country"]) == 2  # US (4), GB (1)
    assert dist["by_country"][0]["country"] == "US"
    assert dist["by_country"][0]["count"] == 4

    modes = {m["work_mode"]: m["count"] for m in dist["by_work_mode"]}
    assert modes.get("Hybrid") == 2
    assert modes.get("Remote") == 1
    assert modes.get("On-site") == 2


def test_get_experience_distribution(sample_market_df: pd.DataFrame):
    exp_dist = get_experience_distribution(sample_market_df)
    assert "tiers" in exp_dist
    assert len(exp_dist["tiers"]) == 5
    assert len(exp_dist["avg_by_family"]) > 0


def test_get_skills_by_role_family(sample_market_df: pd.DataFrame):
    family_skills = get_skills_by_role_family(sample_market_df)
    assert "Data & AI" in family_skills
    assert "Cloud & DevOps" in family_skills
    assert any(s["skill"] == "pytorch" for s in family_skills["Data & AI"])


def test_get_salary_insights(sample_market_df: pd.DataFrame):
    insights = get_salary_insights(sample_market_df)
    assert insights["available"] is True
    assert insights["disclosed_count"] == 4
    assert len(insights["by_family"]) == 3
    assert insights["overall_median"] > 0
