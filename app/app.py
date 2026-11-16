"""JobLens: Personalized Information Retrieval and Job Market Analytics Engine.

Streamlit multi-page dashboard featuring:
- Relevance-ranked job search (custom BM25F vs Keyword Baseline)
- Candidate profiling and SQLite persistence
- Explainable match scoring and aggregate skill-gap analysis
- Saved job management
- Interactive job market insights and co-occurrence analytics
- Information retrieval evaluation metrics and parameter comparison
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from joblens.analytics.market_insights import (
    get_experience_distribution,
    get_location_distribution,
    get_salary_insights,
    get_skill_cooccurrence,
    get_skills_by_role_family,
    get_top_roles,
    get_top_skills,
)
from joblens.ir.baseline import KeywordBaselineRanker
from joblens.ir.bm25 import BM25FRanker, SearchResult
from joblens.ir.inverted_index import InvertedIndex
from joblens.ir.tokenizer import JobTokenizer
from joblens.matching.matcher import CandidateMatcher, JobMatch
from joblens.storage.db import JobLensDatabase

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="JobLens | Personalized Placement Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished dark UI
st.markdown(
    """
    <style>
    /* Card styling */
    .joblens-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .joblens-card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .joblens-meta {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 6px;
    }
    .badge-blue { background-color: #1e3a8a; color: #93c5fd; }
    .badge-green { background-color: #064e3b; color: #6ee7b7; }
    .badge-amber { background-color: #78350f; color: #fde68a; }
    .badge-purple { background-color: #581c87; color: #d8b4fe; }
    .badge-slate { background-color: #334155; color: #cbd5e1; }
    .skill-pill {
        display: inline-block;
        padding: 2px 7px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    .skill-matched { background-color: #064e3b; color: #a7f3d0; border: 1px solid #059669; }
    .skill-missing { background-color: #451a03; color: #fde68a; border: 1px solid #d97706; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Resources and Data Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading job dataset...")
def load_dataset() -> pd.DataFrame:
    """Load sample dataset (preferred for cloud demo) or fallback to full processed dataset."""
    sample_path = ROOT_DIR / "data" / "sample" / "jobs_sample.parquet"
    if sample_path.exists():
        return pd.read_parquet(sample_path)
    processed_path = ROOT_DIR / "data" / "processed"
    if processed_path.exists():
        return pd.read_parquet(processed_path)
    raise FileNotFoundError("Neither sample nor processed dataset found. Run build_dataset.py first.")


@st.cache_resource(show_spinner="Initializing JobLens Inverted Index and IR Engine...")
def load_search_engine() -> Tuple[InvertedIndex, BM25FRanker, KeywordBaselineRanker, CandidateMatcher]:
    """Build and cache the multi-field inverted index and rankers in memory."""
    df = load_dataset()
    tokenizer = JobTokenizer()
    index = InvertedIndex(tokenizer=tokenizer)

    for _, row in df.iterrows():
        doc_id = str(row["job_id"])
        title = str(row["title"])
        raw_skills = row.get("skills")
        if isinstance(raw_skills, (list, tuple, set)) or hasattr(raw_skills, "__iter__") and not isinstance(raw_skills, str):
            skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        elif isinstance(raw_skills, str):
            skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
        else:
            skills = []
        desc = str(row["description"])

        meta = {
            "title": title,
            "company": row.get("company"),
            "company_id": row.get("company_id"),
            "location": str(row.get("location", "")),
            "country": str(row.get("country", "")),
            "work_mode": str(row.get("work_mode", "On-site")),
            "job_type": str(row.get("job_type", "Full-time")),
            "role_family": str(row.get("role_family", "Operations & Other")),
            "min_experience": float(row["min_experience"]) if pd.notnull(row.get("min_experience")) else None,
            "max_experience": float(row["max_experience"]) if pd.notnull(row.get("max_experience")) else None,
            "salary_min": float(row["salary_min"]) if pd.notnull(row.get("salary_min")) else None,
            "salary_max": float(row["salary_max"]) if pd.notnull(row.get("salary_max")) else None,
            "salary_currency": str(row.get("salary_currency")) if pd.notnull(row.get("salary_currency")) else None,
            "skills": skills,
            "posted_date": str(row.get("posted_date", "")),
        }
        index.add_document(doc_id, title, skills, desc, metadata=meta)

    index.finalize()

    bm25 = BM25FRanker(
        index,
        k1=1.2,
        w_title=3.0,
        w_skills=2.0,
        w_desc=1.0,
        b_title=0.3,
        b_skills=0.0,
        b_desc=0.75,
    )
    baseline = KeywordBaselineRanker(index, match_mode="OR")
    matcher = CandidateMatcher(index, ranker=bm25)

    return index, bm25, baseline, matcher


@st.cache_resource
def get_db() -> JobLensDatabase:
    """Initialize and cache the SQLite database access layer."""
    db_path = ROOT_DIR / "data" / "joblens.db"
    return JobLensDatabase(db_path)


@st.cache_data
def load_all_taxonomy_skills() -> List[str]:
    """Load curated skill taxonomy canonical names for profile selection."""
    from joblens.skills.extractor import SkillExtractor
    try:
        extractor = SkillExtractor()
        return sorted(extractor.canonical_skills)
    except Exception:
        tax_path = ROOT_DIR / "src" / "joblens" / "skills" / "taxonomy.yaml"
        if tax_path.exists():
            import yaml
            with open(tax_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                skills = set()
                for cat, items in data.items():
                    for item in items:
                        if isinstance(item, dict) and "canonical" in item:
                            skills.add(item["canonical"])
                return sorted(skills)
        return ["python", "sql", "aws", "docker", "kubernetes", "react", "java"]


# -----------------------------------------------------------------------------
# Navigation & Sidebar Setup
# -----------------------------------------------------------------------------
def main():
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


# -----------------------------------------------------------------------------
# Page 1: Relevance-Ranked Search
# -----------------------------------------------------------------------------
def render_search_page(
    index: InvertedIndex,
    bm25_ranker: BM25FRanker,
    baseline_ranker: KeywordBaselineRanker,
    db: JobLensDatabase,
):
    st.title("Job Search")
    st.markdown("Search across multi-field postings with custom **BM25F** ranking or compare against the **Keyword Baseline**.")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter job title, skills, or target keywords:", placeholder="e.g. machine learning engineer python pytorch")
    with col2:
        engine_mode = st.selectbox("Ranking Engine", ["BM25F (Field-Weighted)", "Keyword Baseline (Term Count)"])

    # Example query quick-click buttons
    st.caption("Quick Queries:")
    quick_cols = st.columns(5)
    if quick_cols[0].button("ML Engineer"):
        query = "machine learning engineer python pytorch"
    if quick_cols[1].button("Cloud DevOps"):
        query = "cloud devops engineer terraform kubernetes aws"
    if quick_cols[2].button("Frontend React"):
        query = "frontend developer react typescript css"
    if quick_cols[3].button("Registered Nurse"):
        query = "registered nurse icu healthcare patient care"
    if quick_cols[4].button("Financial Analyst"):
        query = "financial analyst financial modeling excel forecasting"

    # Search Filters Expander
    with st.expander("Search Filters (Location, Work Mode, Experience, Role Family)", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            family_filter = st.selectbox(
                "Role Family",
                ["All", "Software Engineering", "Data & AI", "Cloud & DevOps", "Cybersecurity", "Product & Management", "Operations & Other"],
            )
            work_mode_filter = st.selectbox("Work Mode", ["All", "Remote", "Hybrid", "On-site"])
        with fcol2:
            country_filter = st.selectbox("Country", ["All", "US", "GB", "CA", "DE", "IN", "BR"])
            state_filter = st.text_input("State / Province (e.g. CA, TX, NY)", "")
        with fcol3:
            city_filter = st.text_input("City (e.g. San Francisco, Austin, London)", "")
            max_exp_filter = st.slider("Maximum Required Experience (Years)", 0.0, 15.0, 15.0, 0.5)

    if not query.strip():
        st.info("Enter a query above or click a quick-query preset to explore postings.")
        return

    # Build active filters
    active_filters: Dict[str, Any] = {}
    if family_filter != "All":
        active_filters["role_family"] = family_filter
    if work_mode_filter != "All":
        active_filters["work_mode"] = work_mode_filter
    if country_filter != "All":
        active_filters["country"] = country_filter
    if state_filter.strip():
        active_filters["state"] = state_filter.strip()
    if city_filter.strip():
        active_filters["city"] = city_filter.strip()
    if max_exp_filter < 15.0:
        active_filters["max_experience"] = max_exp_filter

    t0 = time.time()
    if "BM25F" in engine_mode:
        results = bm25_ranker.search(query, top_k=25, filters=active_filters)
    else:
        results = baseline_ranker.search(query, top_k=25, filters=active_filters, match_mode="OR")
    latency_ms = (time.time() - t0) * 1000.0

    st.markdown(f"**Found {len(results)} matching jobs** in `{latency_ms:.1f} ms` using **{engine_mode}**.")

    if not results:
        st.warning("No postings matched all query terms and filter criteria. Try broadening your query or clearing filters.")
        return

    # Display Ranked Search Results
    for rank, res in enumerate(results, start=1):
        meta = res.metadata
        title = meta.get("title", "Untitled Posting")
        company = meta.get("company") or "Undisclosed Company"
        loc = meta.get("location", "Location Not Listed")
        mode = meta.get("work_mode", "On-site")
        role_fam = meta.get("role_family", "General")
        min_exp = meta.get("min_experience")
        max_exp = meta.get("max_experience")
        skills = meta.get("skills", [])
        salary_min = meta.get("salary_min")
        salary_max = meta.get("salary_max")
        salary_curr = meta.get("salary_currency") or "USD"

        exp_str = "Not specified"
        if min_exp is not None and max_exp is not None:
            exp_str = f"{min_exp:.0f} - {max_exp:.0f} yrs"
        elif min_exp is not None:
            exp_str = f"{min_exp:.0f}+ yrs"

        salary_str = ""
        if salary_min is not None:
            salary_str = f" | 💵 {salary_curr} {salary_min:,.0f}" + (f" - {salary_max:,.0f}" if salary_max else "")

        with st.container():
            st.markdown(
                f"""
                <div class="joblens-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="joblens-card-header">#{rank}. {title}</span>
                        <span class="badge badge-blue">Relevance: {res.score:.2f}</span>
                    </div>
                    <div class="joblens-meta">
                        🏢 <b>{company}</b> &nbsp;|&nbsp; 📍 {loc} &nbsp;|&nbsp; 💼 {exp_str}{salary_str}
                    </div>
                    <div>
                        <span class="badge badge-purple">{role_fam}</span>
                        <span class="badge badge-green">{mode}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Details & Actions
            rcol1, rcol2 = st.columns([4, 1])
            with rcol1:
                # Per-term contribution breakdown expander
                if res.term_scores:
                    with st.expander(f"View scoring contribution breakdown for #{rank}", expanded=False):
                        breakdown_df = pd.DataFrame([
                            {"Query Term": term, "Score Contribution": round(score, 4)}
                            for term, score in sorted(res.term_scores.items(), key=lambda x: x[1], reverse=True)
                        ])
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                if skills:
                    pills = " ".join([f"<span class='skill-pill badge-slate'>{s}</span>" for s in skills[:12]])
                    st.markdown(f"**Required Skills**: {pills}", unsafe_allow_html=True)

            with rcol2:
                btn_key = f"save_{res.doc_id}_{rank}"
                if st.button("⭐ Save Job", key=btn_key):
                    saved = db.save_job(
                        user_id="default_user",
                        job_id=res.doc_id,
                        title=title,
                        company=company,
                        location=loc,
                        match_score=float(res.score),
                        notes="Saved from Search",
                    )
                    if saved:
                        st.success("Saved!")
                    else:
                        st.info("Already saved.")


# -----------------------------------------------------------------------------
# Page 2: Candidate Profile Management
# -----------------------------------------------------------------------------
def render_profile_page(db: JobLensDatabase, taxonomy_skills: List[str]):
    st.title("Candidate Profile")
    st.markdown("Set up your profile to unlock **explainable personalized job recommendations** and **skill-gap analysis**.")

    profile = db.get_profile("default_user") or {
        "user_id": "default_user",
        "skills": ["python", "sql", "aws", "docker"],
        "years_experience": 3.0,
        "preferred_roles": ["Data Scientist", "Software Engineer"],
        "preferred_locations": ["United States"],
        "preferred_cities": ["San Francisco", "Austin"],
        "preferred_states": ["CA", "TX"],
        "preferred_work_modes": ["Remote", "Hybrid"],
        "min_salary": 110000,
    }

    with st.form("profile_form"):
        col1, col2 = st.columns(2)

        with col1:
            selected_skills = st.multiselect(
                "Your Skills (Select or type custom skills)",
                options=taxonomy_skills,
                default=[s for s in profile.get("skills", []) if s in taxonomy_skills],
            )
            custom_skills_input = st.text_input(
                "Additional Custom Skills (comma-separated)",
                value=", ".join([s for s in profile.get("skills", []) if s not in taxonomy_skills]),
            )
            years_exp = st.slider("Years of Professional Experience", 0.0, 25.0, float(profile.get("years_experience", 2.0)), 0.5)
            min_salary = st.number_input("Minimum Desired Annual Salary (USD)", min_value=0, max_value=500000, value=int(profile.get("min_salary", 0) or 0), step=5000)

        with col2:
            preferred_roles_input = st.text_input(
                "Preferred Job Titles (comma-separated)",
                value=", ".join(profile.get("preferred_roles", [])),
                placeholder="e.g. Machine Learning Engineer, Data Scientist",
            )
            preferred_cities_input = st.text_input(
                "Preferred Cities (comma-separated)",
                value=", ".join(profile.get("preferred_cities", [])),
                placeholder="e.g. San Francisco, Austin, New York",
            )
            preferred_states_input = st.text_input(
                "Preferred States (comma-separated, 2-letter codes)",
                value=", ".join(profile.get("preferred_states", [])),
                placeholder="e.g. CA, TX, NY",
            )
            work_modes = st.multiselect(
                "Acceptable Work Modes",
                options=["Remote", "Hybrid", "On-site"],
                default=profile.get("preferred_work_modes", ["Remote", "Hybrid"]),
            )

        submit = st.form_submit_button("💾 Save Profile", use_container_width=True)

    if submit:
        # Merge custom skills with taxonomy skills
        merged_skills = list(selected_skills)
        if custom_skills_input.strip():
            for cs in custom_skills_input.split(","):
                clean_s = cs.strip().lower()
                if clean_s and clean_s not in merged_skills:
                    merged_skills.append(clean_s)

        pref_roles = [r.strip() for r in preferred_roles_input.split(",") if r.strip()]
        pref_cities = [c.strip() for c in preferred_cities_input.split(",") if c.strip()]
        pref_states = [s.strip().upper() for s in preferred_states_input.split(",") if s.strip()]

        db.save_profile(
            user_id="default_user",
            skills=merged_skills,
            years_experience=years_exp,
            preferred_roles=pref_roles,
            preferred_locations=["United States"],
            preferred_cities=pref_cities,
            preferred_states=pref_states,
            preferred_work_modes=work_modes,
            min_salary=min_salary if min_salary > 0 else None,
        )
        st.success("Candidate profile successfully updated in SQLite database!")
        st.rerun()

    # Profile Summary Card
    st.divider()
    st.subheader("Active Profile Overview")
    scol1, scol2, scol3 = st.columns(3)
    scol1.metric("Registered Skills", len(profile.get("skills", [])))
    scol2.metric("Experience", f"{profile.get('years_experience', 0):.1f} Years")
    scol3.metric("Target Work Modes", ", ".join(profile.get("preferred_work_modes", [])) or "Any")

    if profile.get("skills"):
        st.markdown("**Skills on File:**")
        pills = " ".join([f"<span class='skill-pill badge-blue'>{s}</span>" for s in profile["skills"]])
        st.markdown(pills, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Page 3: Candidate Recommendations & Skill-Gap Analysis
# -----------------------------------------------------------------------------
def render_recommendations_page(matcher: CandidateMatcher, db: JobLensDatabase):
    st.title("Personalized Recommendations")
    st.markdown("Matches your profile against all indexed job postings with **transparent fit factors** and **skill-gap insights**.")

    profile = db.get_profile("default_user")
    if not profile or not profile.get("skills"):
        st.warning("Please configure your candidate profile first in **My Profile** to generate personalized recommendations.")
        return

    # Weight customizer expander
    with st.expander("Scoring Weights Adjustment", expanded=False):
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        w_skills = wcol1.slider("Skill Coverage Weight", 0.0, 1.0, 0.40, 0.05)
        w_bm25 = wcol2.slider("BM25 Relevance Weight", 0.0, 1.0, 0.30, 0.05)
        w_exp = wcol3.slider("Experience Fit Weight", 0.0, 1.0, 0.20, 0.05)
        w_loc = wcol4.slider("Location Fit Weight", 0.0, 1.0, 0.10, 0.05)

        matcher.w_skills = w_skills
        matcher.w_bm25 = w_bm25
        matcher.w_exp = w_exp
        matcher.w_loc = w_loc

    top_k = st.slider("Number of recommendations to retrieve", 5, 30, 10)

    matches = matcher.match_candidate(profile, top_k=top_k)
    if not matches:
        st.info("No matching roles found. Try adjusting your preferred roles or skills in My Profile.")
        return

    # Top-N Aggregate Skill Gap Widget
    skill_gaps = matcher.aggregate_skill_gap(matches, top_n=top_k)
    if skill_gaps:
        st.subheader("Top Missing Skills in Target Roles (Skill Gap Analysis)")
        st.caption(f"Frequently required skills across your top-{top_k} recommended positions that are currently missing from your profile:")
        gap_cols = st.columns(min(len(skill_gaps), 5))
        for idx, gap in enumerate(skill_gaps[:5]):
            with gap_cols[idx]:
                st.metric(
                    label=gap["skill"].upper(),
                    value=f"{gap['percentage']}%",
                    help=f"Required in {gap['missing_in_jobs']} of your top {top_k} recommended postings.",
                )
        st.divider()

    st.subheader(f"Top {len(matches)} Recommended Job Matches")

    for rank, m in enumerate(matches, start=1):
        meta = m.metadata
        title = meta.get("title", "Role")
        company = meta.get("company") or "Undisclosed Employer"
        loc = meta.get("location", "United States")
        mode = meta.get("work_mode", "On-site")
        role_fam = meta.get("role_family", "General")

        with st.container():
            st.markdown(
                f"""
                <div class="joblens-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="joblens-card-header">#{rank}. {title}</span>
                        <span class="badge badge-green">Match Score: {m.match_score * 100:.1f}%</span>
                    </div>
                    <div class="joblens-meta">
                        🏢 <b>{company}</b> &nbsp;|&nbsp; 📍 {loc} &nbsp;|&nbsp; <span class="badge badge-purple">{role_fam}</span> <span class="badge badge-blue">{mode}</span>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.88rem; color: #cbd5e1;">
                        💡 <i>{m.rationale}</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Factor breakdown metrics
            bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([1, 1, 1, 1, 1])
            bcol1.metric("Skill Coverage", f"{m.skill_score * 100:.0f}%")
            bcol2.metric("BM25 Fit", f"{m.bm25_score * 100:.0f}%")
            bcol3.metric("Experience Fit", f"{m.experience_score * 100:.0f}%")
            bcol4.metric("Location Fit", f"{m.location_score * 100:.0f}%")

            with bcol5:
                if st.button("⭐ Save Job", key=f"rec_save_{m.job_id}_{rank}"):
                    db.save_job(
                        user_id="default_user",
                        job_id=m.job_id,
                        title=title,
                        company=company,
                        location=loc,
                        match_score=round(m.match_score * 100.0, 1),
                        notes=f"Recommended #{rank} ({m.match_score * 100:.1f}%)",
                    )
                    st.success("Saved!")

            # Skill overlap pills
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                if m.matched_skills:
                    pills = " ".join([f"<span class='skill-pill skill-matched'>✓ {s}</span>" for s in m.matched_skills])
                    st.markdown(f"**Matched Skills**: {pills}", unsafe_allow_html=True)
                else:
                    st.caption("No overlapping skills extracted.")
            with pcol2:
                if m.missing_skills:
                    pills = " ".join([f"<span class='skill-pill skill-missing'>✗ {s}</span>" for s in m.missing_skills])
                    st.markdown(f"**Missing Skills (Skill Gap)**: {pills}", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Page 4: Saved Jobs Management
# -----------------------------------------------------------------------------
def render_saved_jobs_page(db: JobLensDatabase):
    st.title("Saved Jobs")
    st.markdown("Track and manage postings you have bookmarked for applications.")

    saved_list = db.get_saved_jobs("default_user")

    if not saved_list:
        st.info("You have no saved jobs yet. Search for postings or generate recommendations to bookmark roles.")
        return

    st.markdown(f"**You have {len(saved_list)} saved job(s)** on record.")

    for item in saved_list:
        job_id = item["job_id"]
        title = item["title"]
        company = item["company"] or "Company"
        loc = item["location"] or "Location"
        score = item["match_score"]
        saved_at = item.get("saved_at", "")[:10]
        notes = item.get("notes") or ""

        with st.container():
            st.markdown(
                f"""
                <div class="joblens-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="joblens-card-header">{title}</span>
                        <span class="badge badge-blue">Saved on: {saved_at}</span>
                    </div>
                    <div class="joblens-meta">
                        🏢 <b>{company}</b> &nbsp;|&nbsp; 📍 {loc} &nbsp;|&nbsp; Score: <b>{score:.1f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ncol1, ncol2 = st.columns([4, 1])
            with ncol1:
                new_notes = st.text_input("Application Notes", value=notes, key=f"notes_{job_id}")
                if new_notes != notes:
                    db.save_job(
                        user_id="default_user",
                        job_id=job_id,
                        title=title,
                        company=company,
                        location=loc,
                        match_score=score,
                        notes=new_notes,
                    )
                    st.success("Notes updated!")
            with ncol2:
                if st.button("🗑️ Remove", key=f"del_{job_id}"):
                    db.remove_saved_job("default_user", job_id)
                    st.rerun()


# -----------------------------------------------------------------------------
# Page 5: Job Market Analytics
# -----------------------------------------------------------------------------
def render_analytics_page(df: pd.DataFrame):
    st.title("Job Market Analytics")
    st.markdown("Aggregated market intelligence computed directly from the job posting corpus.")

    # Role family selector
    family_options = ["All"] + sorted(df["role_family"].dropna().unique().tolist())
    selected_fam = st.selectbox("Filter Analytics by Role Family", family_options)

    # Top-level metric cards
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    filtered_df = df if selected_fam == "All" else df[df["role_family"] == selected_fam]

    mcol1.metric("Postings Analyzed", f"{len(filtered_df):,}")
    mcol2.metric("Unique Job Titles", f"{filtered_df['title'].nunique():,}")
    top_s = get_top_skills(filtered_df, n=1)
    top_skill_name = top_s[0]["skill"].upper() if top_s else "N/A"
    mcol3.metric("Most In-Demand Skill", top_skill_name)
    salary_data = get_salary_insights(filtered_df)
    mcol4.metric(
        "Median Disclosed Salary",
        f"${salary_data['overall_median']:,}" if salary_data["available"] else "Disclosed in 33%",
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Top Roles",
        "In-Demand Skills & Co-occurrence",
        "Locations & Work Modes",
        "Experience & Salaries",
    ])

    with tab1:
        st.subheader(f"Top 10 Job Titles ({selected_fam})")
        top_roles = get_top_roles(df, n=10, role_family=selected_fam)
        if top_roles:
            roles_df = pd.DataFrame(top_roles)
            st.bar_chart(roles_df.set_index("role")["count"], color="#3b82f6")
            st.dataframe(roles_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Most In-Demand Technical & Soft Skills")
        top_skills = get_top_skills(df, role_family=selected_fam, n=15)
        if top_skills:
            skills_df = pd.DataFrame(top_skills)
            st.bar_chart(skills_df.set_index("skill")["count"], color="#10b981")

        st.subheader("Skill Co-occurrence Patterns")
        st.caption("Pairs of skills that frequently appear together in the same posting:")
        cooccur_pairs = get_skill_cooccurrence(df, role_family=selected_fam, top_n_pairs=10)
        if cooccur_pairs:
            cooccur_df = pd.DataFrame(cooccur_pairs)[["pair", "cooccurrence_count", "cooccurrence_rate"]]
            st.dataframe(cooccur_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Geographic & Work Mode Distributions")
        loc_dist = get_location_distribution(filtered_df, top_n=8)

        lcol1, lcol2 = st.columns(2)
        with lcol1:
            st.markdown("**Work Mode Distribution**")
            mode_df = pd.DataFrame(loc_dist["by_work_mode"])
            if not mode_df.empty:
                st.bar_chart(mode_df.set_index("work_mode")["count"], color="#8b5cf6")

        with lcol2:
            st.markdown("**Top States / Regions**")
            state_df = pd.DataFrame(loc_dist["by_state"])
            if not state_df.empty:
                st.bar_chart(state_df.set_index("state")["count"], color="#f59e0b")

    with tab4:
        st.subheader("Experience Requirements & Career Tiers")
        exp_dist = get_experience_distribution(filtered_df)
        tier_df = pd.DataFrame(exp_dist["tiers"])
        if not tier_df.empty:
            st.bar_chart(tier_df.set_index("tier")["count"], color="#06b6d4")

        st.subheader("Salary Insights by Role Family")
        if salary_data["available"]:
            st.dataframe(pd.DataFrame(salary_data["by_family"]), use_container_width=True, hide_index=True)
        else:
            st.info("Disclosed salary data is limited in this subset.")


# -----------------------------------------------------------------------------
# Page 6: Information Retrieval Evaluation
# -----------------------------------------------------------------------------
def render_evaluation_page():
    st.title("Information Retrieval Engine Evaluation")
    st.markdown("Systematic evaluation comparing our custom **BM25F** multi-field ranker against the **Keyword Baseline**.")

    st.info(
        "**Evaluation Methodology & Indicative Nature**:  \n"
        "Ground truth relevance labels were constructed via pooling top-20 retrieved postings across all evaluated configurations "
        "and applying rule-based graded labeling (0 = Irrelevant, 1 = Partially Relevant, 2 = Highly Relevant) on title regex and skill signals.  \n"
        "Because labels are rule-based and favor title/skill matches, and parameter variants were evaluated on the same 30 queries, "
        "results are indicative of relative retrieval characteristics. Parameter configuration differences are within noise bounds for 30 queries."
    )

    results_csv = ROOT_DIR / "data" / "evaluation" / "results.csv"
    if results_csv.exists():
        st.subheader("Benchmark Metrics Table (30 Labelled Queries)")
        results_df = pd.read_csv(results_csv)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Evaluation results CSV not found. Run scripts/run_evaluation.py to generate benchmarks.")

    chart_path = ROOT_DIR / "data" / "evaluation" / "eval_chart.png"
    if chart_path.exists():
        st.subheader("Retrieval Performance Comparison (P@5, P@10, MAP@20)")
        st.image(str(chart_path), use_container_width=True)

    # Interactive Query Inspector
    queries_path = ROOT_DIR / "data" / "evaluation" / "queries.json"
    if queries_path.exists():
        st.divider()
        st.subheader("Interactive Query Inspector")
        with open(queries_path, "r", encoding="utf-8") as f:
            queries_data = json.load(f)

        query_titles = [f"[{q['query_id']}] {q['query']} ({q['role_family']})" for q in queries_data]
        selected_q_idx = st.selectbox("Inspect an Evaluation Query", range(len(query_titles)), format_func=lambda i: query_titles[i])
        selected_q = queries_data[selected_q_idx]

        qcol1, qcol2 = st.columns(2)
        with qcol1:
            st.markdown(f"**Query**: `{selected_q['query']}`")
            st.markdown(f"**Role Family**: `{selected_q['role_family']}`")
        with qcol2:
            st.markdown(f"**Core Skills**: {', '.join(selected_q['core_skills'])}")
            st.markdown(f"**Title Pattern**: `{selected_q['target_title_regex']}`")


if __name__ == "__main__":
    main()
