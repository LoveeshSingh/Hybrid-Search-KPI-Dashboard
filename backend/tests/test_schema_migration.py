"""
Tests for the QueryStore schema migration mechanism.

Demonstrates:
  1. The failure that occurs when v2 code encounters a v1 database.
  2. The automatic migration that upgrades v1 → v2.
  3. Fresh databases start at the latest schema version.
"""
import pytest
import sqlite3
import os
from tempfile import TemporaryDirectory

from backend.app.db.query_store import QueryStore


def _create_v1_database(db_path: str):
    """
    Manually create a v1 database (no schema_version table, no user_agent column).
    This simulates the state before migration support was added.
    """
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("""
        CREATE TABLE queries (
            request_id   TEXT PRIMARY KEY,
            query        TEXT NOT NULL,
            latency_ms   REAL,
            top_k        INTEGER,
            alpha        REAL,
            result_count INTEGER,
            timestamp    TEXT NOT NULL
        )
    """)
    # Seed one row to prove data is preserved after migration
    conn.execute("""
        INSERT INTO queries (request_id, query, latency_ms, top_k, alpha, result_count, timestamp)
        VALUES ('old-id', 'legacy query', 10.0, 5, 0.5, 3, '2026-01-01T00:00:00+00:00')
    """)
    conn.close()


def test_v2_insert_fails_on_v1_database():
    """
    FAILURE SCENARIO — before migration:
    If we try to INSERT with user_agent into a v1 table that
    has no such column, SQLite raises OperationalError.
    """
    with TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "v1.db")
        _create_v1_database(db_path)

        conn = sqlite3.connect(db_path, isolation_level=None)
        with pytest.raises(sqlite3.OperationalError, match="user_agent"):
            conn.execute("""
                INSERT INTO queries
                    (request_id, query, latency_ms, top_k, alpha, result_count, timestamp, user_agent)
                VALUES ('new-id', 'test', 1.0, 5, 0.5, 1, '2026-03-13T00:00:00+00:00', 'curl/7.0')
            """)
        conn.close()


def test_auto_migration_v1_to_v2():
    """
    Opening a v1 database with the new QueryStore should automatically
    add the user_agent column and set schema_version to 2.
    """
    with TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "v1_migrate.db")
        _create_v1_database(db_path)

        # Constructing QueryStore triggers _init_db → _migrate
        store = QueryStore(db_path=db_path)

        # Legacy data should still be there
        recent = store.get_recent_queries(limit=10)
        assert len(recent) == 1
        assert recent[0]["query"] == "legacy query"
        assert recent[0]["user_agent"] == "unknown"  # default filled by migration

        # New inserts with user_agent should work
        rid = store.log_query(
            query="post-migration query",
            latency_ms=5.0,
            top_k=3,
            alpha=0.6,
            result_count=2,
            user_agent="TestClient/1.0",
        )
        assert rid  # non-empty UUID

        recent = store.get_recent_queries(limit=10)
        assert len(recent) == 2
        assert recent[0]["user_agent"] == "TestClient/1.0"

        # Schema version should now be 2
        conn = sqlite3.connect(db_path, isolation_level=None)
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        conn.close()
        assert version == 2


def test_fresh_database_starts_at_v2():
    """A brand-new database should be at the latest schema version."""
    with TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "fresh.db")
        store = QueryStore(db_path=db_path)

        conn = sqlite3.connect(db_path, isolation_level=None)
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        conn.close()
        assert version == 2

        # user_agent insert should work immediately
        store.log_query(
            query="fresh db query",
            latency_ms=2.0,
            top_k=5,
            alpha=0.5,
            result_count=1,
            user_agent="browser/1.0",
        )
        recent = store.get_recent_queries(limit=1)
        assert recent[0]["user_agent"] == "browser/1.0"
