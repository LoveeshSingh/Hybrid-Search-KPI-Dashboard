import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.app.search.hybrid_search import HybridSearch

def test_hybrid_search_load_failure():
    # Service with empty mocks should fail gracefully to load
    mock_bm25 = MagicMock()
    mock_bm25.model_path.exists.return_value = False
    
    mock_vec = MagicMock()
    mock_vec.index_path.exists.return_value = False
    
    service = HybridSearch(bm25_index=mock_bm25, vector_index=mock_vec)
    assert service.load() is False

def test_hybrid_search_execution():
    # Setup mocks
    mock_bm25 = MagicMock()
    mock_bm25.bm25 = True  # bypass is loaded check
    mock_bm25.query.return_value = [
        {"doc_id": "d1", "score": 10.0},
        {"doc_id": "d2", "score": 5.0}
    ]
    
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = np.array([0.1, 0.2, 0.3])
    
    mock_vec = MagicMock()
    mock_vec.index = True # bypass is loaded check
    mock_vec.query.return_value = [
        {"doc_id": "d2", "score": 0.9},
        {"doc_id": "d3", "score": 0.8}
    ]
    
    service = HybridSearch(
        bm25_index=mock_bm25,
        embedding_pipeline=mock_embedder,
        vector_index=mock_vec
    )
    
    # Execute query
    results = service.search("test query", top_k=2, alpha=0.5)
    
    # Assert
    assert len(results) == 2
    
    # d2 should win due to presence in both strong sets
    assert results[0]["doc_id"] == "d2"
    assert "hybrid_score" in results[0]
    assert "bm25_score" in results[0]
    assert "vector_score" in results[0]
    
    # verify inner calls used expected expanded retrieval pool size
    mock_bm25.query.assert_called_once_with("test query", top_k=20)
    mock_vec.query.assert_called_once()
