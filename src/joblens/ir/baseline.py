"""Boolean and term-count keyword search baseline ranker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from joblens.ir.bm25 import SearchResult
from joblens.ir.inverted_index import InvertedIndex


class KeywordBaselineRanker:
    """Baseline retrieval engine using Boolean AND/OR matching and raw term frequency counts."""

    def __init__(self, index: InvertedIndex, match_mode: str = "OR") -> None:
        """Initialize baseline with inverted index and match mode ('OR' or 'AND')."""
        self.index = index
        self.match_mode = match_mode.upper()

    def _apply_filters(self, doc_id: str, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if candidate document satisfies all filter criteria."""
        if not filters:
            return True

        meta = self.index.doc_metadata.get(doc_id, {})

        if "country" in filters and filters["country"]:
            if meta.get("country", "").upper() != filters["country"].strip().upper():
                return False

        if "state" in filters and filters["state"]:
            req_state = filters["state"].strip().lower()
            loc_str = meta.get("location", "").lower()
            loc_parts = [p.strip() for p in loc_str.split(",") if p.strip()]
            job_state = loc_parts[1].lower() if len(loc_parts) >= 2 else ""
            if req_state != job_state and req_state not in loc_parts:
                return False

        if "city" in filters and filters["city"]:
            req_city = filters["city"].strip().lower()
            loc_str = meta.get("location", "").lower()
            loc_parts = [p.strip() for p in loc_str.split(",") if p.strip()]
            job_city = loc_parts[0].lower() if len(loc_parts) >= 2 else loc_str
            if req_city != job_city and req_city not in loc_parts and req_city not in loc_str:
                return False

        if "work_mode" in filters and filters["work_mode"]:
            if meta.get("work_mode", "").lower() != filters["work_mode"].lower():
                return False

        if "role_family" in filters and filters["role_family"]:
            if meta.get("role_family", "").lower() != filters["role_family"].lower():
                return False

        if "candidate_experience" in filters and filters["candidate_experience"] is not None:
            cand_exp = float(filters["candidate_experience"])
            job_min = meta.get("min_experience")
            job_max = meta.get("max_experience")
            if job_min is not None and cand_exp < float(job_min):
                return False
            if job_max is not None and cand_exp > float(job_max):
                return False

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        match_mode: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute baseline keyword search scoring by total matched term count."""
        mode = (match_mode or self.match_mode).upper()
        query_terms = self.index.tokenizer.tokenize(query)
        if not query_terms or self.index.N == 0:
            return []

        unique_query_terms = list(dict.fromkeys(query_terms))

        # Collect term matches per document
        # doc_term_tfs[doc_id][term] = sum of tf across all fields
        doc_term_tfs: Dict[str, Dict[str, int]] = {}

        for term in unique_query_terms:
            for field in self.index.FIELDS:
                postings = self.index.get_field_postings(field, term)
                for doc_id, (tf, _) in postings.items():
                    if doc_id not in doc_term_tfs:
                        doc_term_tfs[doc_id] = {}
                    doc_term_tfs[doc_id][term] = doc_term_tfs[doc_id].get(term, 0) + tf

        doc_scores: Dict[str, float] = {}
        doc_breakdowns: Dict[str, Dict[str, float]] = {}

        for doc_id, tfs in doc_term_tfs.items():
            if not self._apply_filters(doc_id, filters):
                continue

            matched_terms_count = len(tfs)

            # In AND mode, all query terms must appear in the document
            if mode == "AND" and matched_terms_count < len(unique_query_terms):
                continue

            # Score is total term occurrences
            score = float(sum(tfs.values()))
            if score > 0.0:
                doc_scores[doc_id] = score
                doc_breakdowns[doc_id] = {k: float(v) for k, v in tfs.items()}

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
                    term_scores=doc_breakdowns[doc_id],
                    metadata=self.index.doc_metadata.get(doc_id, {}),
                )
            )

        return results
