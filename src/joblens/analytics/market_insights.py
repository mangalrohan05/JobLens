"""Job market analytics and insights engine.

Provides cached analytical aggregations over job postings datasets:
- Top roles and role distribution
- In-demand skills overall and segmented by role family
- Skill co-occurrence patterns (pairwise matrix / graph edges)
- Location distribution (country, state, city) and work mode trends
- Experience requirements distribution and senior-level thresholds
- Salary insights across roles, locations, and experience brackets
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def get_top_roles(
    df: pd.DataFrame,
    n: int = 10,
    role_family: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the most frequent job titles overall or within a specific role family."""
    if df.empty:
        return []

    subset = df
    if role_family and role_family.lower() != "all":
        subset = df[df["role_family"].str.lower() == role_family.lower()]

    if subset.empty:
        return []

    counts = subset["title"].value_counts().head(n)
    total = len(subset)

    return [
        {
            "role": str(title),
            "count": int(count),
            "percentage": round((count / total) * 100.0, 1),
        }
        for title, count in counts.items()
    ]


def get_top_skills(
    df: pd.DataFrame,
    role_family: Optional[str] = None,
    n: int = 15,
) -> List[Dict[str, Any]]:
    """Compute most frequently in-demand skills overall or for a given role family."""
    if df.empty:
        return []

    subset = df
    if role_family and role_family.lower() != "all":
        subset = df[df["role_family"].str.lower() == role_family.lower()]

    if subset.empty:
        return []

    total_postings = len(subset)
    skill_counter: Counter[str] = Counter()

    for skills_list in subset["skills"]:
        if isinstance(skills_list, (list, np.ndarray)):
            # Unique skills per posting so count represents posting frequency
            unique_in_doc = set(str(s).strip() for s in skills_list if str(s).strip())
            skill_counter.update(unique_in_doc)

    return [
        {
            "skill": skill,
            "count": count,
            "demand_share": round((count / total_postings) * 100.0, 1),
        }
        for skill, count in skill_counter.most_common(n)
    ]


def get_skill_cooccurrence(
    df: pd.DataFrame,
    role_family: Optional[str] = None,
    top_n_skills: int = 25,
    top_n_pairs: int = 15,
) -> List[Dict[str, Any]]:
    """Calculate skill co-occurrence frequencies among the most popular skills."""
    if df.empty:
        return []

    subset = df
    if role_family and role_family.lower() != "all":
        subset = df[df["role_family"].str.lower() == role_family.lower()]

    if subset.empty:
        return []

    # First find top_n_skills to restrict combinatorial space
    top_skills_list = [item["skill"] for item in get_top_skills(subset, n=top_n_skills)]
    top_skills_set = set(top_skills_list)

    pair_counter: Counter[tuple[str, str]] = Counter()

    for skills_list in subset["skills"]:
        if isinstance(skills_list, (list, np.ndarray)):
            matched = sorted(set(str(s).strip() for s in skills_list if str(s).strip() in top_skills_set))
            if len(matched) >= 2:
                for s1, s2 in combinations(matched, 2):
                    pair_counter[(s1, s2)] += 1

    total_postings = len(subset)
    return [
        {
            "skill_a": pair[0],
            "skill_b": pair[1],
            "pair": f"{pair[0]} + {pair[1]}",
            "cooccurrence_count": count,
            "cooccurrence_rate": round((count / total_postings) * 100.0, 2),
        }
        for pair, count in pair_counter.most_common(top_n_pairs)
    ]


def get_location_distribution(df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    """Aggregate distribution across countries, states, top cities, and work modes."""
    if df.empty:
        return {"by_country": [], "by_state": [], "by_city": [], "by_work_mode": []}

    total = len(df)

    # 1. By Country
    country_counts = df["country"].value_counts().head(top_n)
    by_country = [
        {"country": str(k), "count": int(v), "percentage": round((v / total) * 100.0, 1)}
        for k, v in country_counts.items()
    ]

    # 2. By Work Mode
    mode_counts = df["work_mode"].value_counts()
    by_work_mode = [
        {"work_mode": str(k), "count": int(v), "percentage": round((v / total) * 100.0, 1)}
        for k, v in mode_counts.items()
    ]

    # Parse city and state from location string: "City, ST, Country"
    cities: List[str] = []
    states: List[str] = []

    for loc in df["location"]:
        if not loc or not isinstance(loc, str):
            continue
        parts = [p.strip() for p in loc.split(",") if p.strip()]
        if len(parts) >= 2:
            cities.append(parts[0])
            states.append(parts[1])
        elif len(parts) == 1:
            cities.append(parts[0])

    city_counts = pd.Series(cities).value_counts().head(top_n) if cities else pd.Series(dtype=int)
    state_counts = pd.Series(states).value_counts().head(top_n) if states else pd.Series(dtype=int)

    by_city = [
        {"city": str(k), "count": int(v), "percentage": round((v / total) * 100.0, 1)}
        for k, v in city_counts.items()
    ]

    by_state = [
        {"state": str(k), "count": int(v), "percentage": round((v / total) * 100.0, 1)}
        for k, v in state_counts.items()
    ]

    return {
        "by_country": by_country,
        "by_state": by_state,
        "by_city": by_city,
        "by_work_mode": by_work_mode,
    }


def get_experience_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate experience requirement distributions and tier breakdowns."""
    if df.empty:
        return {"tiers": [], "avg_by_family": []}

    # Categorize experience into career tiers
    tiers = {
        "Entry-level (0 - 2 yrs)": 0,
        "Mid-level (3 - 5 yrs)": 0,
        "Senior (6 - 8 yrs)": 0,
        "Lead / Principal (9+ yrs)": 0,
        "Unspecified / Not parsed": 0,
    }

    min_exp_series = pd.to_numeric(df["min_experience"], errors="coerce")
    for val in min_exp_series:
        if pd.isna(val):
            tiers["Unspecified / Not parsed"] += 1
        elif val <= 2.0:
            tiers["Entry-level (0 - 2 yrs)"] += 1
        elif val <= 5.0:
            tiers["Mid-level (3 - 5 yrs)"] += 1
        elif val <= 8.0:
            tiers["Senior (6 - 8 yrs)"] += 1
        else:
            tiers["Lead / Principal (9+ yrs)"] += 1

    total = len(df)
    tier_list = [
        {"tier": tier, "count": count, "percentage": round((count / total) * 100.0, 1)}
        for tier, count in tiers.items()
    ]

    # Average experience per role family
    family_exp = []
    if "role_family" in df.columns:
        valid_exp = df.dropna(subset=["min_experience"]).copy()
        if not valid_exp.empty:
            grouped = valid_exp.groupby("role_family")["min_experience"].agg(["mean", "median", "count"])
            for family, row in grouped.iterrows():
                family_exp.append({
                    "role_family": str(family),
                    "mean_years": round(float(row["mean"]), 1),
                    "median_years": round(float(row["median"]), 1),
                    "sample_size": int(row["count"]),
                })

    return {
        "tiers": tier_list,
        "avg_by_family": sorted(family_exp, key=lambda x: x["mean_years"], reverse=True),
    }


def get_skills_by_role_family(df: pd.DataFrame, top_n: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve top in-demand skills segmented across all distinct role families."""
    if df.empty or "role_family" not in df.columns:
        return {}

    result = {}
    families = sorted(df["role_family"].dropna().unique())
    for fam in families:
        skills = get_top_skills(df, role_family=str(fam), n=top_n)
        result[str(fam)] = skills

    return result


def get_salary_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute salary distribution statistics for postings with disclosed compensation."""
    if df.empty or "salary_min" not in df.columns:
        return {"available": False, "disclosed_count": 0, "by_family": []}

    # Filter to valid positive salaries
    has_salary = df[df["salary_min"].notna() & (df["salary_min"] > 0)].copy()
    if has_salary.empty:
        return {"available": False, "disclosed_count": 0, "by_family": []}

    has_salary["midpoint"] = (has_salary["salary_min"] + has_salary["salary_max"].fillna(has_salary["salary_min"])) / 2.0

    by_family = []
    if "role_family" in has_salary.columns:
        grouped = has_salary.groupby("role_family")["midpoint"].agg(["median", "mean", "min", "max", "count"])
        for family, row in grouped.iterrows():
            by_family.append({
                "role_family": str(family),
                "median_salary": int(round(row["median"])),
                "mean_salary": int(round(row["mean"])),
                "min_salary": int(round(row["min"])),
                "max_salary": int(round(row["max"])),
                "disclosed_postings": int(row["count"]),
            })

    overall_median = int(round(has_salary["midpoint"].median()))
    overall_mean = int(round(has_salary["midpoint"].mean()))

    return {
        "available": True,
        "disclosed_count": len(has_salary),
        "disclosed_rate": round((len(has_salary) / len(df)) * 100.0, 1),
        "overall_median": overall_median,
        "overall_mean": overall_mean,
        "by_family": sorted(by_family, key=lambda x: x["median_salary"], reverse=True),
    }
