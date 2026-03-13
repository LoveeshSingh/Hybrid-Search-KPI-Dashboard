import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.app.api.main import app

client = TestClient(app)

def test_health_check():
    with patch("backend.app.api.main.get_git_commit", return_value="1234abc"):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["commit"] == "1234abc"

@patch("backend.app.api.main.hybrid_search")
@patch("backend.app.api.main.search_logger")
def test_api_search_endpoint(mock_logger, mock_hybrid):
    # Setup dependencies
    mock_hybrid.bm25_index.bm25 = True
    mock_hybrid.vector_index.index = True
    
    mock_hybrid.search.return_value = [
        {"doc_id": "doc_1", "bm25_score": 0.8, "vector_score": 0.9, "hybrid_score": 0.85},
        {"doc_id": "doc_2", "bm25_score": 0.5, "vector_score": 0.5, "hybrid_score": 0.5}
    ]

    payload = {
        "query": "test query",
        "top_k": 2,
        "alpha": 0.5
    }

    response = client.post("/search", json=payload)
    
    # Assert network status
    assert response.status_code == 200
    
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2
    
    # Verify exact schema payload presence
    doc = results[0]
    assert doc["doc_id"] == "doc_1"
    assert "bm25_score" in doc
    assert "vector_score" in doc
    assert "hybrid_score" in doc
    
    # Verify the inner logic was executed using passed constraints
    mock_hybrid.search.assert_called_once_with(query="test query", top_k=2, alpha=0.5)
