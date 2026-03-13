import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@patch("backend.app.api.main.hybrid_search")
def test_search_endpoint(mock_hybrid):
    # Setup mocks
    mock_hybrid.bm25_index.bm25 = True
    mock_hybrid.vector_index.index = True
    
    mock_hybrid.search.return_value = [
        {"doc_id": "d2", "bm25_score": 0.5, "vector_score": 0.88, "hybrid_score": 0.69},
        {"doc_id": "d1", "bm25_score": 1.0, "vector_score": 0.0, "hybrid_score": 0.5}
    ]

    payload = {
        "query": "artificial intelligence",
        "top_k": 2,
        "alpha": 0.5
    }

    response = client.post("/search", json=payload)
    
    assert response.status_code == 200
    results = response.json()
    
    assert len(results) == 2
    
    # Verify the structure matches SearchResult model
    assert "doc_id" in results[0]
    assert "bm25_score" in results[0]
    assert "vector_score" in results[0]
    assert "hybrid_score" in results[0]
    
    # d2 should be top because it scores well on both
    assert results[0]["doc_id"] == "d2"

@patch("backend.app.api.main.hybrid_search")
def test_search_endpoint_uninitialized(mock_hybrid):
    # Setup uninitialized state
    mock_hybrid.bm25_index.bm25 = None
    mock_hybrid.vector_index.index = None
    
    payload = {
        "query": "test",
        "top_k": 5,
        "alpha": 0.5
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 503
