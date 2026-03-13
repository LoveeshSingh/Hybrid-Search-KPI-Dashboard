import pytest
import os
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.app.api.main import app
from backend.app.db.query_store import QueryStore

client = TestClient(app)

@patch("backend.app.api.main.query_store")
def test_metrics_endpoint_structure(mock_store):
    mock_store.get_metrics.return_value = {
        "total_search_requests": 50,
        "average_latency_ms": 32.5,
        "p50_latency": 28.0,
        "p95_latency": 85.0,
        "zero_result_queries": 2,
    }

    response = client.get("/metrics")
    assert response.status_code == 200

    data = response.json()
    assert data["total_search_requests"] == 50
    assert data["average_latency_ms"] == 32.5
    assert data["p50_latency"] == 28.0
    assert data["p95_latency"] == 85.0
    assert data["zero_result_queries"] == 2

def test_query_store_metrics_computation():
    with TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "metrics_test.db")
        store = QueryStore(db_path=db_path)

        # Empty state
        metrics = store.get_metrics()
        assert metrics["total_search_requests"] == 0

        # Seed data – varied latencies and one zero-result query
        store.log_query(query="q1", latency_ms=10.0, top_k=5, alpha=0.5, result_count=3)
        store.log_query(query="q2", latency_ms=20.0, top_k=5, alpha=0.5, result_count=5)
        store.log_query(query="q3", latency_ms=100.0, top_k=5, alpha=0.5, result_count=0)
        store.log_query(query="q4", latency_ms=30.0, top_k=5, alpha=0.5, result_count=2)

        metrics = store.get_metrics()

        assert metrics["total_search_requests"] == 4
        assert metrics["average_latency_ms"] == 40.0  # (10+20+100+30)/4
        # sorted: [10, 20, 30, 100] → p50 idx=int(4*0.50)=2 → 30.0
        assert metrics["p50_latency"] == 30.0
        assert metrics["zero_result_queries"] == 1
        # p95 should be the highest latency bucket
        assert metrics["p95_latency"] == 100.0
