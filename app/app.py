"""JobLens: Personalized Information Retrieval and Job Market Analytics Engine.

Streamlit multi-page dashboard entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

from joblens.ui.dashboard import (
    get_db,
    inject_custom_css,
    load_all_taxonomy_skills,
    load_dataset,
    load_search_engine,
    render_analytics_page,
    render_evaluation_page,
    render_profile_page,
    render_recommendations_page,
    render_saved_jobs_page,
    render_search_page,
)

st.set_page_config(
    page_title="JobLens | Personalized Placement Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    inject_custom_css()
    df = load_dataset()
    index, bm25_ranker, baseline_ranker, matcher = load_search_engine()
    db = get_db()
    all_taxonomy_skills = load_all_taxonomy_skills()

    st.sidebar.title("JobLens")
    st.sidebar.caption(f"Information Retrieval & Market Analytics  \nIndexed: **{index.N:,}** job postings")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Search",
            "My Profile",
            "Recommendations",
            "Saved Jobs",
            "Market Insights",
            "Evaluation",
        ],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption("System Status: **Ready (Local / Streamlit Cloud)**")
    st.sidebar.caption("Tech Stack: SQLite, Parquet, BM25F, Pandas")

    # Dispatch to page renderer
    if page == "Search":
        render_search_page(index, bm25_ranker, baseline_ranker, db)
    elif page == "My Profile":
        render_profile_page(db, all_taxonomy_skills)
    elif page == "Recommendations":
        render_recommendations_page(matcher, db)
    elif page == "Saved Jobs":
        render_saved_jobs_page(db)
    elif page == "Market Insights":
        render_analytics_page(df)
    elif page == "Evaluation":
        render_evaluation_page()


if __name__ == "__main__":
    main()
