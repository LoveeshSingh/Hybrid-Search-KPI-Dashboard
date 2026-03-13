import sqlite3
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class QueryStore:
    def __init__(self, db_path: str = "data/db/search_queries.db"):
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
                    CREATE TABLE IF NOT EXISTS queries (
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
            logger.error(f"Failed to initialize SQLite query database: {e}")

    def log_query(self, query: str, latency_ms: float, top_k: int, alpha: float, result_count: int) -> str:
        """Log a search query and return the generated request_id."""
        request_id = str(uuid.uuid4())
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO queries 
                    (request_id, query, latency_ms, top_k, alpha, result_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (request_id, query, latency_ms, top_k, alpha, result_count))
        except Exception as e:
            logger.error(f"Failed to log query {request_id}: {e}")
            
        return request_id

    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent queries ordered by timestamp."""
        results = []
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row  # To return dict-like objects
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT request_id, timestamp, query, latency_ms, top_k, alpha, result_count 
                    FROM queries 
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                for row in rows:
                    results.append(dict(row))
                    
        except Exception as e:
            logger.error(f"Failed to get recent queries: {e}")
            
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Aggregate basic metrics from the queries table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM queries")
                total_requests = cursor.fetchone()[0]
                
                if total_requests == 0:
                    return {
                        "total_requests": 0,
                        "avg_latency_ms": 0.0,
                        "avg_results_returned": 0.0
                    }
                
                cursor.execute("SELECT AVG(latency_ms) FROM queries")
                avg_latency = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(result_count) FROM queries")
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
