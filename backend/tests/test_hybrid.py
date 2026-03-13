import pytest
import math
from backend.app.search.hybrid import min_max_normalize, z_score_normalize, hybrid_rank

def test_min_max_normalize():
    scores = [10.0, 20.0, 30.0]
    norm = min_max_normalize(scores)
    assert norm == [0.0, 0.5, 1.0]
    
    # Test identical scores
    norm2 = min_max_normalize([5.0, 5.0])
    assert norm2 == [0.5, 0.5]

def test_z_score_normalize():
    scores = [10.0, 20.0, 30.0]  # mean: 20, std: ~8.16
    norm = z_score_normalize(scores)
    assert math.isclose(norm[1], 0.0, abs_tol=1e-5) # 20 is the mean
    assert norm[0] < 0  # 10 is below mean
    assert norm[2] > 0  # 30 is above mean
    
    # Test identical scores
    norm2 = z_score_normalize([5.0, 5.0])
    assert norm2 == [0.0, 0.0]

def test_hybrid_rank_min_max():
    bm25_res = [
        {"doc_id": "d1", "score": 10.0},
        {"doc_id": "d2", "score": 5.0}  
    ]
    # For min-max, d1 bm25 norm will be 1.0, d2 will be 0.0
    # doc "d3" implicitly has 0.0 before norm
    # so raw bm25: d1=10, d2=5, d3=0 -> norm bm25: d1=1.0, d2=0.5, d3=0.0
    
    vector_res = [
        {"doc_id": "d2", "score": 0.8},
        {"doc_id": "d3", "score": 0.9}
    ]
    # doc "d1" implicitly has 0.0 before norm
    # raw vector: d1=0.0, d2=0.8, d3=0.9 -> norm vector: d1=0.0, d2=0.888, d3=1.0
    
    # Using alpha = 0.5
    results = hybrid_rank(bm25_res, vector_res, alpha=0.5, normalization="min-max")
    
    assert len(results) == 3
    
    # Build dictionary for easy assert
    res_dict = {r['doc_id']: r for r in results}
    
    # d1 hybrid score = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    assert res_dict["d1"]["bm25_score"] == 1.0
    assert res_dict["d1"]["vector_score"] == 0.0
    assert res_dict["d1"]["hybrid_score"] == 0.5
    
    # d2 hybrid score = 0.5 * 0.5 + 0.5 * (0.8/0.9) = 0.25 + 0.444 = 0.694
    assert res_dict["d2"]["bm25_score"] == 0.5
    assert math.isclose(res_dict["d2"]["vector_score"], 0.8888, rel_tol=1e-3)
    assert math.isclose(res_dict["d2"]["hybrid_score"], 0.6944, rel_tol=1e-3)
    
    # Sort order
    assert results[0]["doc_id"] == "d2" # Highest score
    assert results[1]["doc_id"] == "d1"

def test_hybrid_alpha_weight():
    # If alpha is 1.0, only BM25 matters
    bm25_res = [{"doc_id": "d1", "score": 10.0}, {"doc_id": "d2", "score": 1.0}]
    vector_res = [{"doc_id": "d1", "score": 0.1}, {"doc_id": "d2", "score": 0.9}]
    
    res1 = hybrid_rank(bm25_res, vector_res, alpha=1.0, normalization="min-max")
    assert res1[0]["doc_id"] == "d1"  # d1 wins on bm25
    
    # If alpha is 0.0, only vector matters
    res2 = hybrid_rank(bm25_res, vector_res, alpha=0.0, normalization="min-max")
    assert res2[0]["doc_id"] == "d2"  # d2 wins on vector
