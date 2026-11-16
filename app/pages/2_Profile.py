"""Page 2: Candidate Profile Management."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

st.set_page_config(
    page_title="Candidate Profile | JobLens",
    page_icon="👤",
    layout="wide",
)

from joblens.ui.dashboard import get_db, load_all_taxonomy_skills, render_profile_page

db = get_db()
taxonomy_skills = load_all_taxonomy_skills()

st.sidebar.caption("Candidate Profile & Skill Management")
st.sidebar.divider()
st.sidebar.caption("Database: **SQLite (data/joblens.db)**")

render_profile_page(db, taxonomy_skills)
