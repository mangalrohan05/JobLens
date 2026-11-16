"""Tests for information retrieval evaluation metrics."""

import math
import pytest

from joblens.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    mean_average_precision,
    ndcg_at_k,
    precision_at_k,
)


def test_precision_at_k_hand_calculated():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5", "doc7"}

    # P@1: doc1 is rel -> 1/1 = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    # P@2: doc1 rel, doc2 not -> 1/2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    # P@3: doc1, doc3 rel -> 2/3
    assert abs(precision_at_k(retrieved, relevant, k=3) - 2 / 3) < 1e-6
    # P@5: doc1, doc3, doc5 rel -> 3/5 = 0.6
    assert precision_at_k(retrieved, relevant, k=5) == 0.6
    # Empty retrieved
    assert precision_at_k([], relevant, k=5) == 0.0


def test_average_precision_hand_calculated():
    # Ranks:
    # 1: doc1 (Rel) -> P@1 = 1/1
    # 2: doc2 (Not)
    # 3: doc3 (Rel) -> P@3 = 2/3
    # 4: doc4 (Not)
    # 5: doc5 (Rel) -> P@5 = 3/5
    # Total relevant in ground truth = 4 (doc1, doc3, doc5, doc7)
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5", "doc7"}

    expected_ap = (1.0 / 1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 4.0
    computed_ap = average_precision(retrieved, relevant)
    assert abs(computed_ap - expected_ap) < 1e-6

    # Test when relevant set is empty
    assert average_precision(retrieved, set()) == 0.0


def test_mean_average_precision():
    q_retrieved = {
        "q1": ["d1", "d2"],  # d1 rel -> AP = (1/1) / 1 = 1.0
        "q2": ["d3", "d4"],  # d4 rel -> AP = (1/2) / 1 = 0.5
    }
    q_relevant = {
        "q1": {"d1"},
        "q2": {"d4"},
    }

    map_score = mean_average_precision(q_retrieved, q_relevant)
    assert abs(map_score - 0.75) < 1e-6


def test_ndcg_at_k_hand_calculated():
    retrieved = ["d1", "d2", "d3"]
    # Graded relevance: d1=2, d2=0, d3=1
    qrels = {"d1": 2, "d2": 0, "d3": 1}

    # DCG@3:
    # rank 1: (2^2 - 1) / log2(2) = 3 / 1.0 = 3.0
    # rank 2: (2^0 - 1) / log2(3) = 0
    # rank 3: (2^1 - 1) / log2(4) = 1 / 2.0 = 0.5
    # Actual DCG = 3.5
    assert abs(dcg_at_k(retrieved, qrels, k=3) - 3.5) < 1e-6

    # Ideal ranking: d1 (2), d3 (1), d2 (0)
    # IDCG@3:
    # rank 1: 3 / 1 = 3.0
    # rank 2: 1 / log2(3) = 1 / 1.58496 = 0.63092975
    # rank 3: 0
    # IDCG = 3.63092975
    # NDCG = 3.5 / 3.63092975 = ~0.96394
    ndcg = ndcg_at_k(retrieved, qrels, k=3)
    assert 0.96 < ndcg < 0.97
