"""Page 1: Relevance-Ranked Job Search."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

st.set_page_config(
    page_title="Job Search | JobLens",
    page_icon="🔍",
    layout="wide",
)

from joblens.ui.dashboard import get_db, load_search_engine, render_search_page

index, bm25_ranker, baseline_ranker, matcher = load_search_engine()
db = get_db()

st.sidebar.caption(f"Information Retrieval Engine  \nIndexed: **{index.N:,}** job postings")
st.sidebar.divider()
st.sidebar.caption("System Status: **Ready**")

render_search_page(index, bm25_ranker, baseline_ranker, db)
