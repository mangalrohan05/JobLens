"""Candidate matching engine with explainable match breakdowns and skill gap aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from joblens.ir.bm25 import BM25FRanker, SearchResult
from joblens.ir.inverted_index import InvertedIndex


@dataclass
class JobMatch:
    """Detailed candidate-to-job match result with explainable factors."""

    job_id: str
    match_score: float  # Normalized 0.0 - 1.0
    skill_score: float
    bm25_score: float
    experience_score: float
    location_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    experience_fit_note: str
    location_fit_note: str
    rationale: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "match_score": round(self.match_score * 100.0, 1),
            "skill_score": round(self.skill_score * 100.0, 1),
            "bm25_score": round(self.bm25_score * 100.0, 1),
            "experience_score": round(self.experience_score * 100.0, 1),
            "location_score": round(self.location_score * 100.0, 1),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "experience_fit_note": self.experience_fit_note,
            "location_fit_note": self.location_fit_note,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


class CandidateMatcher:
    """Matches candidate profile against job postings with explainable components."""

    def __init__(
        self,
        index: InvertedIndex,
        ranker: Optional[BM25FRanker] = None,
        weight_skills: float = 0.40,
        weight_bm25: float = 0.30,
        weight_experience: float = 0.20,
        weight_location: float = 0.10,
    ) -> None:
        self.index = index
        self.ranker = ranker or BM25FRanker(index)
        self.w_skills = weight_skills
        self.w_bm25 = weight_bm25
        self.w_exp = weight_experience
        self.w_loc = weight_location

    def _compute_experience_fit(
        self, candidate_years: float, job_min: Optional[float], job_max: Optional[float]
    ) -> Tuple[float, str]:
        """Compute experience fit score [0.0, 1.0] and generate a human-readable explanation."""
        if job_min is None and job_max is None:
            return 1.0, "No strict experience requirements specified."

        if job_min is not None:
            if candidate_years < job_min:
                shortfall = job_min - candidate_years
                score = max(0.0, 1.0 - (shortfall * 0.25))
                return score, f"Experience shortfall: requires {job_min:.0f}+ yrs, profile has {candidate_years:.1f} yrs."
            elif job_max is not None and candidate_years > job_max:
                over = candidate_years - job_max
                if over > 3.0:
                    score = max(0.70, 1.0 - (over * 0.04))
                    return score, f"Slightly over-qualified: requires {job_min:.0f}-{job_max:.0f} yrs, profile has {candidate_years:.1f} yrs."
                return 1.0, f"Strong experience fit: profile ({candidate_years:.1f} yrs) meets required {job_min:.0f}-{job_max:.0f} yrs."
            else:
                return 1.0, f"Direct experience match: profile ({candidate_years:.1f} yrs) meets required {job_min:.0f}+ yrs."

        if job_max is not None:
            if candidate_years <= job_max:
                return 1.0, f"Within maximum range (up to {job_max:.0f} yrs)."
            else:
                return 0.85, f"Profile experience ({candidate_years:.1f} yrs) exceeds target {job_max:.0f} yrs."

        return 1.0, "Experience fits role expectations."

    def _compute_location_fit(
        self,
        job_loc: str,
        job_country: str,
        job_work_mode: str,
        pref_locations: List[str],
        pref_modes: List[str],
        pref_cities: Optional[List[str]] = None,
        pref_states: Optional[List[str]] = None,
    ) -> Tuple[float, str]:
        """Compute location/work-mode fit score [0.0, 1.0] and generate explanation."""
        pref_modes_clean = [m.lower() for m in pref_modes]
        job_mode_lower = job_work_mode.lower()

        # Check work mode preference
        mode_match = job_mode_lower in pref_modes_clean or not pref_modes_clean

        # Parse city, state, country from standardized location (e.g. "San Francisco, CA, US")
        loc_parts = [p.strip() for p in job_loc.split(",") if p.strip()]
        job_city = loc_parts[0].lower() if len(loc_parts) >= 2 else ""
        job_state = loc_parts[1].lower() if len(loc_parts) >= 2 else ""
        loc_clean = job_loc.lower()
        country_clean = job_country.lower()

        # Check city and state preferences
        city_match = True
        if pref_cities:
            city_match = any(c.lower() == job_city or c.lower() in loc_clean for c in pref_cities)

        state_match = True
        if pref_states:
            state_match = any(s.lower() == job_state or s.lower() in loc_clean for s in pref_states)

        # Check general location preference
        loc_pref_match = True
        if pref_locations:
            loc_pref_match = any(
                p.lower() in loc_clean
                or p.lower() == country_clean
                or p.lower() == job_state
                or p.lower() == job_city
                for p in pref_locations
            )

        loc_match = loc_pref_match and city_match and state_match

        if job_work_mode == "Remote" and ("remote" in pref_modes_clean or not pref_modes_clean):
            return 1.0, "Matches preference for Remote work."
        elif loc_match and mode_match:
            return 1.0, f"Matches location ({job_loc}) and work mode ({job_work_mode})."
        elif loc_match:
            return 0.75, f"Located in {job_loc}, but work mode is {job_work_mode}."
        elif mode_match and job_work_mode == "Hybrid":
            return 0.50, f"Hybrid role located in {job_loc}."
        elif mode_match and job_work_mode == "Remote":
            return 1.0, "Remote position."
        else:
            return 0.20, f"Location mismatch: posting in {job_loc} ({job_work_mode})."

    def match_candidate(
        self,
        profile: Dict[str, Any],
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[JobMatch]:
        """Match a candidate profile against all indexed jobs and return ranked explainable matches."""
        cand_skills = set(s.lower().strip() for s in profile.get("skills", []) if s.strip())
        cand_exp = float(profile.get("years_experience", 0.0))
        pref_roles = profile.get("preferred_roles", [])
        pref_locations = profile.get("preferred_locations", [])
        pref_cities = profile.get("preferred_cities", [])
        pref_states = profile.get("preferred_states", [])
        pref_modes = profile.get("preferred_work_modes", [])

        # Build profile query for BM25F
        query_parts = []
        if pref_roles:
            query_parts.extend(pref_roles)
        if cand_skills:
            query_parts.extend(list(cand_skills)[:10])

        profile_query = " ".join(query_parts) if query_parts else "software"

        # Search candidates using BM25
        bm25_results = self.ranker.search(profile_query, top_k=top_k * 3, filters=filters)
        if not bm25_results:
            return []

        max_bm25_score = max((r.score for r in bm25_results), default=1.0)
        max_bm25_score = max(max_bm25_score, 1e-4)

        matches: List[JobMatch] = []

        for res in bm25_results:
            meta = res.metadata
            job_skills = set(s.lower().strip() for s in meta.get("skills", []) if s.strip())

            # 1. Skill Overlap Fit: |candidate ∩ required| / |required|
            if job_skills:
                matched_skills_set = job_skills.intersection(cand_skills)
                missing_skills_set = job_skills.difference(cand_skills)
                skill_score = len(matched_skills_set) / len(job_skills)
                skill_note = f"matches {len(matched_skills_set)}/{len(job_skills)} required skills ({int(skill_score * 100)}%)"
            else:
                # Zero required skills: use neutral 0.5 and rebalance remaining weights in explanation
                matched_skills_set = set()
                missing_skills_set = set()
                skill_score = 0.50
                sum_other = self.w_bm25 + self.w_exp + self.w_loc
                if sum_other > 0:
                    pct_bm25 = int(round((self.w_bm25 / sum_other) * 100))
                    pct_exp = int(round((self.w_exp / sum_other) * 100))
                    pct_loc = int(round((self.w_loc / sum_other) * 100))
                else:
                    pct_bm25 = pct_exp = pct_loc = 33
                skill_note = (
                    f"no explicit skills required (neutral 0.5 skill fit; "
                    f"weights rebalanced to {pct_bm25}% BM25 relevance, {pct_exp}% experience, {pct_loc}% location)"
                )

            matched_skills = sorted(matched_skills_set)
            missing_skills = sorted(missing_skills_set)

            # 2. BM25 Relevance Fit
            norm_bm25 = min(1.0, res.score / max_bm25_score)

            # 3. Experience Fit
            job_min_exp = meta.get("min_experience")
            job_max_exp = meta.get("max_experience")
            exp_score, exp_note = self._compute_experience_fit(cand_exp, job_min_exp, job_max_exp)

            # 4. Location & Work Mode Fit
            job_loc = meta.get("location", "United States")
            job_country = meta.get("country", "US")
            job_work_mode = meta.get("work_mode", "On-site")
            loc_score, loc_note = self._compute_location_fit(
                job_loc,
                job_country,
                job_work_mode,
                pref_locations,
                pref_modes,
                pref_cities=pref_cities,
                pref_states=pref_states,
            )

            # Total Weighted Match Score
            total_match_score = (
                self.w_skills * skill_score
                + self.w_bm25 * norm_bm25
                + self.w_exp * exp_score
                + self.w_loc * loc_score
            )

            # Generate plain-language rationale
            title = meta.get("title", "Role")
            company = meta.get("company") or "the employer"
            rationale = f"{title} at {company}: {skill_note}. {exp_note} {loc_note}"

            matches.append(
                JobMatch(
                    job_id=res.doc_id,
                    match_score=total_match_score,
                    skill_score=skill_score,
                    bm25_score=norm_bm25,
                    experience_score=exp_score,
                    location_score=loc_score,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    experience_fit_note=exp_note,
                    location_fit_note=loc_note,
                    rationale=rationale,
                    metadata=meta,
                )
            )

        # Sort descending by match_score, tie-break by ascending doc_id
        matches.sort(key=lambda m: (-m.match_score, m.job_id))
        return matches[:top_k]

    @staticmethod
    def aggregate_skill_gap(matches: List[JobMatch], top_n: int = 10) -> List[Dict[str, Any]]:
        """Compute aggregate skill gap across the top-N recommended jobs.

        Identifies the skills that appear most frequently among missing skills.
        """
        top_slice = matches[:top_n]
        if not top_slice:
            return []

        gap_counts: Dict[str, int] = {}
        for m in top_slice:
            for s in m.missing_skills:
                gap_counts[s] = gap_counts.get(s, 0) + 1

        total_jobs = len(top_slice)
        gap_summary = []
        for skill, count in sorted(gap_counts.items(), key=lambda x: x[1], reverse=True):
            gap_summary.append({
                "skill": skill,
                "missing_in_jobs": count,
                "percentage": round((count / total_jobs) * 100.0, 1),
            })

        return gap_summary
