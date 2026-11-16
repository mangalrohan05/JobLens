"""Page 6: Information Retrieval Engine Evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import streamlit as st

st.set_page_config(
    page_title="IR Evaluation | JobLens",
    page_icon="📈",
    layout="wide",
)

from joblens.ui.dashboard import render_evaluation_page

st.sidebar.caption("IR System Evaluation & Benchmark")
st.sidebar.divider()
st.sidebar.caption("Metrics: **P@5, P@10, MAP@20**")

render_evaluation_page()
