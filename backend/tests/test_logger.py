import pytest
import os
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.app.logging.logger import SQLiteLogger
from backend.app.api.main import app

def test_sqlite_logger():
    with TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_logs.db")
        logger = SQLiteLogger(db_path=db_path)
        
        # Initial metrics should be 0
        metrics = logger.get_metrics()
        assert metrics["total_requests"] == 0
        
        # Log a few requests
        logger.log_search(query="test query 1", latency_ms=10.5, top_k=5, alpha=0.5, result_count=3)
        logger.log_search(query="test query 2", latency_ms=20.0, top_k=10, alpha=0.8, result_count=5)
        
        # Check metrics again
        metrics = logger.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["avg_latency_ms"] == 15.25
        assert metrics["avg_results_returned"] == 4.0

client = TestClient(app)

@patch("backend.app.api.main.search_logger")
def test_metrics_endpoint(mock_logger):
    # Setup mock
    mock_logger.get_metrics.return_value = {
        "total_requests": 100,
        "avg_latency_ms": 45.5,
        "avg_results_returned": 5.0
    }
    
    response = client.get("/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_requests"] == 100
    assert data["avg_latency_ms"] == 45.5

@patch("backend.app.api.main.search_logger")
@patch("backend.app.api.main.bm25_index")
@patch("backend.app.api.main.vector_index")
@patch("backend.app.api.main.embedding_pipeline")
def test_search_endpoint_logging(mock_embedding, mock_vector, mock_bm25, mock_logger):
    # Setup mocks
    mock_bm25.bm25 = True
    mock_bm25.query.return_value = [{"doc_id": "d1", "score": 10.0}]
    
    mock_vector.index = True
    mock_vector.query.return_value = [{"doc_id": "d1", "score": 0.9}]
    
    payload = {"query": "logging check", "top_k": 2, "alpha": 0.5}

    response = client.post("/search", json=payload)
    
    assert response.status_code == 200
    
    # Verify logger was called once
    mock_logger.log_search.assert_called_once()
    
    # Verify logger args
    _, kwargs = mock_logger.log_search.call_args
    assert kwargs["query"] == "logging check"
    assert "latency_ms" in kwargs
    assert kwargs["top_k"] == 2
    assert kwargs["alpha"] == 0.5
    assert kwargs["result_count"] == 1
