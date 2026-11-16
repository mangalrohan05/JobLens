"""Information retrieval evaluation metrics implemented from scratch.

Provides manual implementations for:
- Precision@K (P@K)
- Average Precision (AP)
- Mean Average Precision (MAP)
- Discounted Cumulative Gain (DCG@K) and Normalized DCG (NDCG@K)
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Union


def precision_at_k(
    retrieved: List[str],
    relevant_set: Set[str],
    k: int = 10,
) -> float:
    """Calculate Precision@K: proportion of top-K retrieved docs that are relevant."""
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return relevant_retrieved / float(k)


def average_precision(
    retrieved: List[str],
    relevant_set: Set[str],
    k: Optional[int] = None,
) -> float:
    """Calculate Average Precision (AP) for a single ranked list.

    AP = sum_{k=1}^N (P@k * I(doc_k in relevant_set)) / total_relevant
    If total_relevant == 0, returns 0.0.
    """
    if not relevant_set:
        return 0.0

    eval_list = retrieved[:k] if k is not None else retrieved
    if not eval_list:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for idx, doc_id in enumerate(eval_list, start=1):
        if doc_id in relevant_set:
            hit_count += 1
            precision_sum += hit_count / float(idx)

    total_relevant = len(relevant_set)
    return precision_sum / float(total_relevant)


def mean_average_precision(
    query_retrieved: Dict[str, List[str]],
    query_relevant: Dict[str, Set[str]],
    k: Optional[int] = None,
) -> float:
    """Calculate Mean Average Precision (MAP) across all evaluated queries."""
    if not query_retrieved:
        return 0.0

    ap_scores: List[float] = []
    for q_id, retrieved in query_retrieved.items():
        rel_set = query_relevant.get(q_id, set())
        ap = average_precision(retrieved, rel_set, k=k)
        ap_scores.append(ap)

    return sum(ap_scores) / float(len(ap_scores)) if ap_scores else 0.0


def dcg_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int = 10,
) -> float:
    """Calculate Discounted Cumulative Gain (DCG@K) using graded relevance."""
    top_k = retrieved[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(top_k, start=1):
        rel = qrels.get(doc_id, 0)
        # Standard formulation: (2^rel - 1) / log2(idx + 1)
        dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 1.0)
    return dcg


def ndcg_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int = 10,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
    actual_dcg = dcg_at_k(retrieved, qrels, k=k)

    # Ideal ranking: sort all available qrel documents descending by relevance
    sorted_rels = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = 0.0
    for idx, rel in enumerate(sorted_rels, start=1):
        ideal_dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 1.0)

    if ideal_dcg <= 0.0:
        return 0.0

    return actual_dcg / ideal_dcg
