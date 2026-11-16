"""BM25F ranker with field weighting, per-term score breakdown, and pre/post filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from joblens.ir.inverted_index import InvertedIndex


@dataclass
class SearchResult:
    """Ranked search result with relevance score and explainable term contributions."""

    doc_id: str
    score: float
    term_scores: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "term_scores": {k: round(v, 4) for k, v in self.term_scores.items()},
            "metadata": self.metadata,
        }


class BM25FRanker:
    """BM25F ranker supporting multi-field weighting (title, skills, description)."""

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.2,
        w_title: float = 3.0,
        w_skills: float = 2.0,
        w_desc: float = 1.0,
        b_title: float = 0.3,
        b_skills: float = 0.0,
        b_desc: float = 0.75,
    ) -> None:
        self.index = index
        self.k1 = k1
        self.field_weights = {
            "title": w_title,
            "skills": w_skills,
            "description": w_desc,
        }
        self.b_params = {
            "title": b_title,
            "skills": b_skills,
            "description": b_desc,
        }

    def compute_idf(self, term: str) -> float:
        """Compute Robertson-Spärck Jones IDF with Robertson addition: ln(1 + (N - df + 0.5) / (df + 0.5))."""
        df = self.index.get_document_frequency(term)
        if df == 0:
            return 0.0
        n = self.index.N
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def _apply_filters(self, doc_id: str, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if candidate document satisfies all filter criteria."""
        if not filters:
            return True

        meta = self.index.doc_metadata.get(doc_id, {})

        # Filter by country
        if "country" in filters and filters["country"]:
            req_country = filters["country"].strip().upper()
            if meta.get("country", "").upper() != req_country:
                return False

        # Filter by state (e.g. "CA", "TX", "NY")
        if "state" in filters and filters["state"]:
            req_state = filters["state"].strip().lower()
            loc_str = meta.get("location", "").lower()
            loc_parts = [p.strip() for p in loc_str.split(",") if p.strip()]
            job_state = loc_parts[1].lower() if len(loc_parts) >= 2 else ""
            if req_state != job_state and req_state not in loc_parts:
                return False

        # Filter by city (e.g. "San Francisco", "Austin")
        if "city" in filters and filters["city"]:
            req_city = filters["city"].strip().lower()
            loc_str = meta.get("location", "").lower()
            loc_parts = [p.strip() for p in loc_str.split(",") if p.strip()]
            job_city = loc_parts[0].lower() if len(loc_parts) >= 2 else loc_str
            if req_city != job_city and req_city not in loc_parts and req_city not in loc_str:
                return False

        # Filter by work mode (Remote / Hybrid / On-site)
        if "work_mode" in filters and filters["work_mode"]:
            req_mode = filters["work_mode"].lower()
            if meta.get("work_mode", "").lower() != req_mode:
                return False

        # Filter by role family
        if "role_family" in filters and filters["role_family"]:
            req_family = filters["role_family"].lower()
            if meta.get("role_family", "").lower() != req_family:
                return False

        # Filter by candidate years of experience
        # (Job must accept candidate: job min_exp <= cand_exp, and job max_exp >= cand_exp if max_exp exists)
        if "candidate_experience" in filters and filters["candidate_experience"] is not None:
            cand_exp = float(filters["candidate_experience"])
            job_min = meta.get("min_experience")
            job_max = meta.get("max_experience")
            if job_min is not None and cand_exp < float(job_min):
                return False
            if job_max is not None and cand_exp > float(job_max):
                return False

        # Direct max experience filter (job's required min_experience <= max_allowed)
        if "max_experience" in filters and filters["max_experience"] is not None:
            job_min = meta.get("min_experience")
            if job_min is not None and float(job_min) > float(filters["max_experience"]):
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Perform BM25F relevance-ranked search with per-term score breakdown."""
        query_terms = self.index.tokenizer.tokenize(query)
        if not query_terms or self.index.N == 0:
            return []

        # Find candidate documents containing at least one query term in any field
        candidate_docs: Set[str] = set()
        term_idfs: Dict[str, float] = {}

        for term in set(query_terms):
            idf = self.compute_idf(term)
            term_idfs[term] = idf
            if idf <= 0.0:
                continue
            for field in self.index.FIELDS:
                postings = self.index.get_field_postings(field, term)
                candidate_docs.update(postings.keys())

        if not candidate_docs:
            return []

        doc_scores: Dict[str, float] = {}
        doc_term_breakdowns: Dict[str, Dict[str, float]] = {}

        for doc_id in candidate_docs:
            # Apply pre-ranking filters
            if not self._apply_filters(doc_id, filters):
                continue

            total_score = 0.0
            term_scores: Dict[str, float] = {}

            for term in set(query_terms):
                idf = term_idfs.get(term, 0.0)
                if idf <= 0.0:
                    continue

                # Compute normalized term frequency across all fields
                tilde_tf = 0.0
                for field in self.index.FIELDS:
                    postings = self.index.get_field_postings(field, term)
                    if doc_id in postings:
                        tf, _ = postings[doc_id]
                        w_f = self.field_weights[field]
                        b_f = self.b_params[field]
                        doc_len = self.index.doc_lengths[field].get(doc_id, 0)
                        avg_len = self.index.avg_doc_lengths.get(field, 1.0)

                        if avg_len > 0:
                            len_norm = 1.0 - b_f + b_f * (doc_len / avg_len)
                        else:
                            len_norm = 1.0

                        len_norm = max(len_norm, 1e-4)
                        tilde_tf += w_f * (tf / len_norm)

                if tilde_tf > 0.0:
                    term_score = idf * (tilde_tf / (self.k1 + tilde_tf))
                    total_score += term_score
                    term_scores[term] = term_score

            if total_score > 0.0:
                doc_scores[doc_id] = total_score
                doc_term_breakdowns[doc_id] = term_scores

        if not doc_scores:
            return []

        # Sort: descending by score, tie-break by ascending doc_id
        ranked_doc_ids = sorted(
            doc_scores.keys(),
            key=lambda d: (-doc_scores[d], d),
        )

        results: List[SearchResult] = []
        for doc_id in ranked_doc_ids[:top_k]:
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=doc_scores[doc_id],
                    term_scores=doc_term_breakdowns[doc_id],
                    metadata=self.index.doc_metadata.get(doc_id, {}),
                )
            )

        return results
