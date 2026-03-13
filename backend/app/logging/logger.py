import sqlite3
import uuid
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SQLiteLogger:
    def __init__(self, db_path: str = "data/metrics/search_logs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self):
        # Isolation level None means auto-commit
        return sqlite3.connect(str(self.db_path), isolation_level=None)

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS search_logs (
                        request_id TEXT PRIMARY KEY,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        query TEXT,
                        latency_ms REAL,
                        top_k INTEGER,
                        alpha REAL,
                        result_count INTEGER
                    )
                """)
        except Exception as e:
            logger.error(f"Failed to initialize SQLite log database: {e}")

    def log_search(self, query: str, latency_ms: float, top_k: int, alpha: float, result_count: int) -> str:
        """Log a search request and return the generated request_id."""
        request_id = str(uuid.uuid4())
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO search_logs 
                    (request_id, query, latency_ms, top_k, alpha, result_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (request_id, query, latency_ms, top_k, alpha, result_count))
        except Exception as e:
            logger.error(f"Failed to log search request {request_id}: {e}")
            
        return request_id

    def get_metrics(self) -> Dict[str, Any]:
        """Aggregate basic metrics from the search_logs table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM search_logs")
                total_requests = cursor.fetchone()[0]
                
                if total_requests == 0:
                    return {
                        "total_requests": 0,
                        "avg_latency_ms": 0.0,
                        "avg_results_returned": 0.0
                    }
                
                cursor.execute("SELECT AVG(latency_ms) FROM search_logs")
                avg_latency = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(result_count) FROM search_logs")
                avg_results = cursor.fetchone()[0]
                
                return {
                    "total_requests": total_requests,
                    "avg_latency_ms": round(avg_latency, 2) if avg_latency else 0.0,
                    "avg_results_returned": round(avg_results, 2) if avg_results else 0.0
                }
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "avg_results_returned": 0.0,
                "error": str(e)
            }
