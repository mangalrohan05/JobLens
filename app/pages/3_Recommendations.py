"""Page 3: Personalized Recommendations & Skill Gap Analysis."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

st.set_page_config(
    page_title="Recommendations | JobLens",
    page_icon="🎯",
    layout="wide",
)

from joblens.ui.dashboard import get_db, load_search_engine, render_recommendations_page

index, bm25_ranker, baseline_ranker, matcher = load_search_engine()
db = get_db()

st.sidebar.caption(f"Candidate Matching Engine  \nIndexed: **{index.N:,}** job postings")
st.sidebar.divider()
st.sidebar.caption("Scoring: **SkillCoverage + BM25 + ExpFit + LocFit**")

render_recommendations_page(matcher, db)
