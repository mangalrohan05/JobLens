"""Page 4: Saved Jobs Management."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

st.set_page_config(
    page_title="Saved Jobs | JobLens",
    page_icon="⭐",
    layout="wide",
)

from joblens.ui.dashboard import get_db, render_saved_jobs_page

db = get_db()

st.sidebar.caption("Saved Postings & Application Tracker")
st.sidebar.divider()
st.sidebar.caption("Database: **SQLite (data/joblens.db)**")

render_saved_jobs_page(db)
