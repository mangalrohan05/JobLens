"""Smoke tests for Streamlit application pages using Streamlit AppTest framework."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_main_app_smoke():
    """Smoke test for the main Streamlit application entry point."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"AppTest raised exceptions: {[e.value for e in at.exception]}"
    # Verify title or header was rendered
    assert len(at.title) > 0 or len(at.header) > 0


def test_page_search_smoke():
    """Smoke test for Page 1: Search."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "1_Search.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Page 1 raised exceptions: {[e.value for e in at.exception]}"


def test_page_profile_smoke():
    """Smoke test for Page 2: Profile."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "2_Profile.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Page 2 raised exceptions: {[e.value for e in at.exception]}"


def test_page_recommendations_smoke():
    """Smoke test for Page 3: Recommendations."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "3_Recommendations.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Page 3 raised exceptions: {[e.value for e in at.exception]}"


def test_page_saved_jobs_smoke():
    """Smoke test for Page 4: Saved Jobs."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "4_Saved_Jobs.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Page 4 raised exceptions: {[e.value for e in at.exception]}"


def test_page_market_insights_smoke():
    """Smoke test for Page 5: Market Insights."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "5_Market_Insights.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Page 5 raised exceptions: {[e.value for e in at.exception]}"


def test_page_evaluation_smoke():
    """Smoke test for Page 6: Evaluation."""
    at = AppTest.from_file(str(ROOT_DIR / "app" / "pages" / "6_Evaluation.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"Page 6 raised exceptions: {[e.value for e in at.exception]}"
