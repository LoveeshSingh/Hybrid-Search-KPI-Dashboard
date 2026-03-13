import pytest
import os
import sqlite3
from tempfile import TemporaryDirectory
from backend.app.db.query_store import QueryStore

def test_query_store_operations():
    with TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_queries.db")
        store = QueryStore(db_path=db_path)
        
        # Database file should be physically created
        assert os.path.exists(db_path)
        
        # Store some queries
        req1 = store.log_query(
            query="test q1", latency_ms=10.5, top_k=5, alpha=0.5, result_count=3
        )
        req2 = store.log_query(
            query="test q2", latency_ms=25.0, top_k=10, alpha=0.9, result_count=10
        )
        
        # Verify they were saved and get_recent_queries can retrieve them
        recent = store.get_recent_queries(limit=5)
        
        assert len(recent) == 2
        
        # Queries should be retrieved descending (req2 should be first, then req1)
        # Because we inserted them incredibly fast they might share the same second.
        # But we can check they are both in there.
        ids = [row["request_id"] for row in recent]
        assert req1 in ids
        assert req2 in ids
        
        # Verify schema accuracy manually for one object
        q2 = [r for r in recent if r["request_id"] == req2][0]
        assert q2["query"] == "test q2"
        assert q2["latency_ms"] == 25.0
        assert q2["top_k"] == 10
        assert q2["alpha"] == 0.9
        assert q2["result_count"] == 10
        assert "timestamp" in q2
