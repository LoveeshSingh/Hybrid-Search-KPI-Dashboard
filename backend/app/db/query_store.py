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

    Schema v2:
        request_id  TEXT PRIMARY KEY
        query       TEXT NOT NULL
        latency_ms  REAL
        top_k       INTEGER
        alpha       REAL
        result_count INTEGER
        timestamp   TEXT NOT NULL
        user_agent  TEXT NOT NULL DEFAULT 'unknown'
    """

    CURRENT_SCHEMA_VERSION = 2

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(str(self.db_path), isolation_level=None)

    # ── schema bootstrap & migration ──────────────────────────────────────

    def _init_db(self):
        """Create tables if needed, then run any pending migrations."""
        try:
            with self._conn() as conn:
                # Version tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER NOT NULL
                    )
                """)

                version = self._get_schema_version(conn)

                if version == 0:
                    # Fresh database → create schema at latest version
                    self._create_v2_schema(conn)
                    self._set_schema_version(conn, self.CURRENT_SCHEMA_VERSION)
                    logger.info("Initialized QueryStore with schema v2.")
                elif version < self.CURRENT_SCHEMA_VERSION:
                    self._migrate(conn, version)
                else:
                    logger.info(f"QueryStore schema is up-to-date (v{version}).")
        except Exception as e:
            logger.error(f"Failed to initialize QueryStore database: {e}")

    @staticmethod
    def _get_schema_version(conn) -> int:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0

    @staticmethod
    def _set_schema_version(conn, version: int):
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    @staticmethod
    def _create_v2_schema(conn):
        """Create the queries table at the latest schema version."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                request_id   TEXT PRIMARY KEY,
                query        TEXT NOT NULL,
                latency_ms   REAL,
                top_k        INTEGER,
                alpha        REAL,
                result_count INTEGER,
                timestamp    TEXT NOT NULL,
                user_agent   TEXT NOT NULL DEFAULT 'unknown'
            )
        """)

    def _migrate(self, conn, from_version: int):
        """Run incremental migrations from from_version to CURRENT."""
        logger.info(f"Migrating QueryStore schema from v{from_version} to v{self.CURRENT_SCHEMA_VERSION}...")

        if from_version < 2:
            self._migrate_v1_to_v2(conn)

        self._set_schema_version(conn, self.CURRENT_SCHEMA_VERSION)
        logger.info("Migration complete.")

    @staticmethod
    def _migrate_v1_to_v2(conn):
        """v1 → v2: add the user_agent column."""
        logger.info("  v1 → v2: Adding 'user_agent' column...")
        conn.execute(
            "ALTER TABLE queries ADD COLUMN user_agent TEXT NOT NULL DEFAULT 'unknown'"
        )

    # ── public API ────────────────────────────────────────────────────────

    def log_query(
        self,
        query: str,
        latency_ms: float,
        top_k: int,
        alpha: float,
        result_count: int,
        user_agent: str = "unknown",
    ) -> str:
        """Insert a search request record and return the generated request_id."""
        request_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO queries
                        (request_id, query, latency_ms, top_k, alpha, result_count, timestamp, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (request_id, query, latency_ms, top_k, alpha, result_count, ts, user_agent),
                )
        except Exception as e:
            logger.error(f"Failed to log query {request_id}: {e}")
        return request_id

    def get_metrics(self) -> Dict[str, Any]:
        """Compute system-wide search metrics from the query log."""
        try:
            with self._conn() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM queries")
                total = cursor.fetchone()[0]

                if total == 0:
                    return {
                        "total_search_requests": 0,
                        "average_latency_ms": 0.0,
                        "p50_latency": 0.0,
                        "p95_latency": 0.0,
                        "zero_result_queries": 0,
                    }

                cursor.execute("SELECT AVG(latency_ms) FROM queries")
                avg_latency = cursor.fetchone()[0] or 0.0

                # Percentiles via sorted latency values
                cursor.execute("SELECT latency_ms FROM queries ORDER BY latency_ms ASC")
                latencies = [row[0] for row in cursor.fetchall()]

                p50_idx = int(len(latencies) * 0.50)
                p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
                p50 = latencies[p50_idx]
                p95 = latencies[p95_idx]

                cursor.execute("SELECT COUNT(*) FROM queries WHERE result_count = 0")
                zero_results = cursor.fetchone()[0]

                return {
                    "total_search_requests": total,
                    "average_latency_ms": round(avg_latency, 2),
                    "p50_latency": round(p50, 2),
                    "p95_latency": round(p95, 2),
                    "zero_result_queries": zero_results,
                }
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            return {"error": str(e)}

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
