"""Tests for experience range parsing."""

import pytest
from joblens.cleaning.experience import parse_experience_range


def test_parse_range():
    desc = "Candidates should have 3-5 years of professional software engineering experience."
    min_exp, max_exp = parse_experience_range(desc)
    assert min_exp == 3.0
    assert max_exp == 5.0


def test_parse_range_word_to():
    desc = "Looking for an engineer with 4 to 6 years of experience in distributed systems."
    min_exp, max_exp = parse_experience_range(desc)
    assert min_exp == 4.0
    assert max_exp == 6.0


def test_parse_minimum_plus():
    desc = "Requirements: 5+ years of Python and cloud development experience."
    min_exp, max_exp = parse_experience_range(desc)
    assert min_exp == 5.0
    assert max_exp is None


def test_parse_at_least():
    desc = "Must have at least 7 years of industry experience leading technical teams."
    min_exp, max_exp = parse_experience_range(desc)
    assert min_exp == 7.0
    assert max_exp is None


def test_parse_no_experience():
    desc = "Entry-level position. No experience required; training will be provided."
    min_exp, max_exp = parse_experience_range(desc)
    assert min_exp == 0.0
    assert max_exp == 0.0


def test_parse_up_to():
    desc = "Ideal for graduates with up to 2 years of experience."
    min_exp, max_exp = parse_experience_range(desc)
    assert min_exp == 0.0
    assert max_exp == 2.0


def test_fallback_to_title():
    desc = "Exciting opportunity to join our core backend team."
    min_exp, max_exp = parse_experience_range(desc, title="Senior Software Engineer")
    assert min_exp == 5.0
    assert max_exp is None

    min_intern, max_intern = parse_experience_range(desc, title="Summer Software Engineer Intern")
    assert min_intern == 0.0
    assert max_intern == 1.0


def test_missing_experience():
    desc = "We build high performance distributed search engines with passion."
    min_exp, max_exp = parse_experience_range(desc, title="Software Engineer")
    assert min_exp is None
    assert max_exp is None
