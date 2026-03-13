import sqlite3
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = "data/metrics/query_store.db"

class QueryStore:
    """
    SQLite-backed persistence for search request logs.
    
    Schema:
        request_id  TEXT PRIMARY KEY
        query       TEXT
        latency_ms  REAL
        top_k       INTEGER
        alpha       REAL
        result_count INTEGER
        timestamp   TEXT (ISO 8601 UTC)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(str(self.db_path), isolation_level=None)

    def _init_db(self):
        try:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS queries (
                        request_id   TEXT PRIMARY KEY,
                        query        TEXT NOT NULL,
                        latency_ms   REAL,
                        top_k        INTEGER,
                        alpha        REAL,
                        result_count INTEGER,
                        timestamp    TEXT NOT NULL
                    )
                """)
        except Exception as e:
            logger.error(f"Failed to initialize QueryStore database: {e}")

    def log_query(
        self,
        query: str,
        latency_ms: float,
        top_k: int,
        alpha: float,
        result_count: int,
    ) -> str:
        """Insert a search request record and return the generated request_id."""
        request_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO queries
                        (request_id, query, latency_ms, top_k, alpha, result_count, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (request_id, query, latency_ms, top_k, alpha, result_count, ts),
                )
        except Exception as e:
            logger.error(f"Failed to log query {request_id}: {e}")
        return request_id

    def get_recent_queries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent `limit` logged queries, newest first."""
        try:
            with self._conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to retrieve recent queries: {e}")
            return []
