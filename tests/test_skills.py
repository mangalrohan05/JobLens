"""Tests for the skill taxonomy and extractor."""

import pytest
from joblens.skills.extractor import SkillExtractor


@pytest.fixture
def extractor() -> SkillExtractor:
    return SkillExtractor()


def test_taxonomy_size_and_structure(extractor: SkillExtractor):
    stats = extractor.get_stats()
    assert stats["total_canonical_skills"] >= 300, f"Expected >= 300 skills, found {stats['total_canonical_skills']}"
    assert stats["total_aliases"] >= 300
    assert len(stats["categories"]) >= 10


def test_alias_mapping(extractor: SkillExtractor):
    text = "We are seeking a developer with experience in JS, TS, K8s, and Postgres."
    skills = extractor.extract_skills(text)
    assert "javascript" in skills
    assert "typescript" in skills
    assert "kubernetes" in skills
    assert "postgresql" in skills


def test_special_characters_extraction(extractor: SkillExtractor):
    text = "Senior Software Engineer skilled in C++, C#, .NET, Node.js, and CI/CD pipelines."
    skills = extractor.extract_skills(text)
    assert "c++" in skills
    assert "c#" in skills
    assert "asp.net" in skills  # .net maps to asp.net
    assert "node.js" in skills
    assert "continuous integration" in skills


def test_no_false_positives_for_common_words(extractor: SkillExtractor):
    # Ensure words like "going", "algorithm", "cat", "good" don't trigger false positives
    text = "The candidate should have good interpersonal relations and ongoing passion for problem solving."
    skills = extractor.extract_skills(text)
    assert "go" not in skills
    assert "c" not in skills
    assert "r" not in skills
    assert "problem solving" in skills


def test_ambiguous_short_tokens_in_valid_context(extractor: SkillExtractor):
    # Test valid contexts for C, R, and Go
    text_c = "Strong background in C programming and memory management."
    assert "c" in extractor.extract_skills(text_c)

    text_r = "Data analysis using R programming and Python."
    skills_r = extractor.extract_skills(text_r)
    assert "r" in skills_r
    assert "python" in skills_r

    text_go = "Backend microservices built with Go developer tools and Golang."
    assert "go" in extractor.extract_skills(text_go)


def test_soft_skills_extraction(extractor: SkillExtractor):
    text = "Must possess excellent communication, leadership, and critical thinking."
    skills = extractor.extract_skills(text)
    assert "communication" in skills
    assert "leadership" in skills
    assert "critical thinking" in skills


def test_category_lookup(extractor: SkillExtractor):
    assert extractor.get_category("python") == "programming_languages"
    assert extractor.get_category("react") == "web_frontend"
    assert extractor.get_category("docker") == "cloud_infrastructure"
    assert extractor.get_category("apache spark") == "big_data_engineering"
