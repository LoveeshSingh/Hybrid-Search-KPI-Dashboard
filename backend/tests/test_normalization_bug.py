"""
Tests demonstrating and fixing the divide-by-zero bug in min-max normalization.

Bug:  When all BM25 (or vector) scores are identical, min == max and
      (score - min) / (max - min) produces a ZeroDivisionError or NaN.

Fix:  The normalization function must detect equal ranges and return a
      safe fallback (0.0) instead of dividing by zero.
"""
import math
import pytest
from backend.app.search.hybrid import hybrid_rank, min_max_normalize


# ── 1. Reproduce the bug with the raw helper ──────────────────────────────

def _broken_min_max_normalize(scores):
    """
    INTENTIONALLY BUGGY version that does NOT guard against equal scores.
    This is what the code looked like before the fix.
    """
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    # BUG: no guard for max_val == min_val → ZeroDivisionError
    return [(s - min_val) / (max_val - min_val) for s in scores]


def test_broken_normalization_produces_error():
    """Demonstrate the ZeroDivisionError when all scores are equal."""
    identical_scores = [5.0, 5.0, 5.0]
    with pytest.raises(ZeroDivisionError):
        _broken_min_max_normalize(identical_scores)


def test_broken_normalization_nan_from_numpy():
    """
    When numpy is used for the division, we might get NaN/inf
    instead of an exception.
    """
    import numpy as np
    scores = [5.0, 5.0, 5.0]
    arr = np.array(scores)
    min_v, max_v = arr.min(), arr.max()
    # numpy silently returns nan instead of raising
    result = (arr - min_v) / (max_v - min_v)
    assert all(math.isnan(v) for v in result)


# ── 2. Verify the fix produces valid scores ───────────────────────────────

def test_fixed_normalization_equal_scores():
    """After the fix, identical scores should return 0.0 (no signal)."""
    result = min_max_normalize([5.0, 5.0, 5.0])
    assert all(v == 0.0 for v in result)


def test_fixed_normalization_single_score():
    """A single score means range = 0; should still be safe."""
    result = min_max_normalize([42.0])
    assert result == [0.0]


# ── 3. End-to-end: hybrid_rank with identical BM25 scores ────────────────

def test_hybrid_rank_identical_bm25_scores():
    """
    All three BM25 docs have the same score.
    Vector scores differ → hybrid ranking should still work, driven entirely
    by the vector component.
    """
    bm25_results = [
        {"doc_id": "d1", "score": 5.0},
        {"doc_id": "d2", "score": 5.0},
        {"doc_id": "d3", "score": 5.0},
    ]
    vector_results = [
        {"doc_id": "d1", "score": 0.9},
        {"doc_id": "d2", "score": 0.5},
        {"doc_id": "d3", "score": 0.1},
    ]

    results = hybrid_rank(bm25_results, vector_results, alpha=0.5)

    # No NaN or Inf anywhere
    for r in results:
        assert not math.isnan(r["hybrid_score"]), f"NaN in {r}"
        assert not math.isinf(r["hybrid_score"]), f"Inf in {r}"

    # d1 should rank first because it has the best vector score
    assert results[0]["doc_id"] == "d1"


def test_hybrid_rank_all_scores_identical():
    """
    Both BM25 *and* vector scores are identical.
    All hybrid scores should equal 0.0 (no discriminating signal).
    """
    bm25_results = [
        {"doc_id": "d1", "score": 3.0},
        {"doc_id": "d2", "score": 3.0},
    ]
    vector_results = [
        {"doc_id": "d1", "score": 0.7},
        {"doc_id": "d2", "score": 0.7},
    ]

    results = hybrid_rank(bm25_results, vector_results, alpha=0.5)

    for r in results:
        assert r["hybrid_score"] == 0.0
        assert not math.isnan(r["hybrid_score"])
