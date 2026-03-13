import pytest
import os
from tempfile import TemporaryDirectory

from backend.app.db.query_store import QueryStore

def test_log_and_retrieve_queries():
    with TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_queries.db")
        store = QueryStore(db_path=db_path)

        # Initially empty
        recent = store.get_recent_queries()
        assert recent == []

        # Log two queries
        rid1 = store.log_query(query="hello world", latency_ms=12.5, top_k=5, alpha=0.5, result_count=3)
        rid2 = store.log_query(query="deep learning", latency_ms=25.0, top_k=10, alpha=0.8, result_count=7)

        assert rid1 != rid2  # unique request IDs

        recent = store.get_recent_queries(limit=10)
        assert len(recent) == 2

        # Newest first
        assert recent[0]["query"] == "deep learning"
        assert recent[1]["query"] == "hello world"

        # Verify all fields present
        row = recent[0]
        assert row["request_id"] == rid2
        assert row["latency_ms"] == 25.0
        assert row["top_k"] == 10
        assert row["alpha"] == 0.8
        assert row["result_count"] == 7
        assert "timestamp" in row
