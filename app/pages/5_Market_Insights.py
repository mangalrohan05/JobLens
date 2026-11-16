"""Page 5: Job Market Analytics & In-Demand Skills."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

st.set_page_config(
    page_title="Market Insights | JobLens",
    page_icon="📊",
    layout="wide",
)

from joblens.ui.dashboard import load_dataset, render_analytics_page

df = load_dataset()

st.sidebar.caption(f"Market Analytics  \nCorpus: **{len(df):,}** job postings")
st.sidebar.divider()
st.sidebar.caption("Visualizations: Roles, Skills, Locations, Experience")

render_analytics_page(df)
