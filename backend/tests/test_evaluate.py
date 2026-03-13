import pytest
import os
import csv
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.app.evaluation.evaluate import (
    ndcg_at_k,
    recall_at_k,
    mrr_at_k,
    run_evaluation,
)


# ── individual metric helpers ──────────────────────────────────────────────

def test_ndcg_perfect():
    """If retrieval order matches ideal order, nDCG should be 1.0."""
    qrel = {"d1": 3, "d2": 2, "d3": 1}
    retrieved = ["d1", "d2", "d3"]
    assert ndcg_at_k(retrieved, qrel, k=3) == 1.0


def test_ndcg_reversed():
    """Worst ordering should yield nDCG < 1."""
    qrel = {"d1": 3, "d2": 2, "d3": 1}
    retrieved = ["d3", "d2", "d1"]
    score = ndcg_at_k(retrieved, qrel, k=3)
    assert 0.0 < score < 1.0


def test_recall_at_k():
    qrel = {"d1": 1, "d2": 1, "d3": 0}  # 2 relevant docs
    retrieved = ["d1", "d4"]
    assert recall_at_k(retrieved, qrel, k=2) == 0.5  # 1 of 2 relevant


def test_mrr_at_k():
    qrel = {"d3": 1}
    # First relevant doc is at position 3
    retrieved = ["d1", "d2", "d3"]
    assert mrr_at_k(retrieved, qrel, k=3) == pytest.approx(1 / 3)


def test_mrr_no_relevant():
    qrel = {"d5": 1}
    retrieved = ["d1", "d2", "d3"]
    assert mrr_at_k(retrieved, qrel, k=3) == 0.0


# ── end-to-end evaluation ─────────────────────────────────────────────────

def test_run_evaluation_toy():
    queries = [
        {"query_id": "q1", "text": "machine learning"},
        {"query_id": "q2", "text": "deep learning"},
    ]
    qrels = {
        "q1": {"d1": 2, "d2": 1},
        "q2": {"d3": 1},
    }

    # Fake search function returning predictable order
    def fake_search(query_text, top_k, alpha):
        if "machine" in query_text:
            return [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d4"}]
        return [{"doc_id": "d3"}, {"doc_id": "d4"}]

    with TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "experiments.csv")

        with patch("backend.app.evaluation.evaluate.EXPERIMENTS_CSV", csv_path):
            metrics = run_evaluation(queries, qrels, fake_search, alpha=0.5, top_k=3)

        assert metrics["num_queries"] == 2
        assert 0.0 <= metrics["ndcg@10"] <= 1.0
        assert 0.0 <= metrics["recall@10"] <= 1.0
        assert 0.0 <= metrics["mrr@10"] <= 1.0

        # CSV should have been created with one data row
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["alpha"] == "0.5"
            assert "ndcg@10" in rows[0]
