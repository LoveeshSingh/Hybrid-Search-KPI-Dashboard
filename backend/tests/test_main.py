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

@patch("backend.app.api.main.bm25_index")
@patch("backend.app.api.main.vector_index")
@patch("backend.app.api.main.embedding_pipeline")
def test_search_endpoint(mock_embedding, mock_vector, mock_bm25):
    # Setup mocks
    mock_bm25.bm25 = True  # Just to pass the "is None" check
    mock_bm25.query.return_value = [
        {"doc_id": "d1", "score": 10.0},
        {"doc_id": "d2", "score": 5.0}
    ]
    
    mock_vector.index = True
    mock_vector.query.return_value = [
        {"doc_id": "d2", "score": 0.9},
        {"doc_id": "d3", "score": 0.8}
    ]
    
    mock_embedding.embed_query.return_value = np.array([0.1, 0.2])

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

def test_search_endpoint_uninitialized():
    # If app hasn't loaded indices (like on a fresh start), it should return 503
    payload = {
        "query": "test",
        "top_k": 5,
        "alpha": 0.5
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 503
