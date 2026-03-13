import numpy as np
from typing import List, Dict, Any

def min_max_normalize(scores: List[float]) -> List[float]:
    """Normalize scores using min-max scaling to range [0, 1]."""
    if not scores:
        return []
    min_val = min(scores)
    max_val = max(scores)
    
    if max_val == min_val:
        # Avoid division by zero if all scores are identical
        return [0.5 for _ in scores]
        
    return [(s - min_val) / (max_val - min_val) for s in scores]

def z_score_normalize(scores: List[float]) -> List[float]:
    """Normalize scores using z-score normalization (standardization)."""
    if not scores:
        return []
    
    mean_val = np.mean(scores)
    std_dev = np.std(scores)
    
    if std_dev == 0:
        return [0.0 for _ in scores]
        
    return [(s - mean_val) / std_dev for s in scores]

def hybrid_rank(
    bm25_results: List[Dict[str, Any]], 
    vector_results: List[Dict[str, Any]], 
    alpha: float = 0.5,
    normalization: str = "min-max"
) -> List[Dict[str, Any]]:
    """
    Combine BM25 and vector search results into a single ranked list.
    hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_vector
    
    Args:
        bm25_results: List of dicts, must contain 'doc_id' and 'score'
        vector_results: List of dicts, must contain 'doc_id' and 'score'
        alpha: Weight for BM25 score (0.0 to 1.0). 1-alpha is used for vector score.
        normalization: 'min-max' or 'z-score'
    """
    if normalization not in ["min-max", "z-score"]:
        raise ValueError("normalization must be 'min-max' or 'z-score'")
        
    # Extract unique doc IDs
    all_doc_ids = set()
    
    bm25_dict = {}
    for res in bm25_results:
        doc_id = res['doc_id']
        bm25_dict[doc_id] = res.get('score', 0.0)
        all_doc_ids.add(doc_id)
        
    vector_dict = {}
    for res in vector_results:
        doc_id = res['doc_id']
        vector_dict[doc_id] = res.get('score', 0.0)
        all_doc_ids.add(doc_id)
        
    # Prepare lists for normalization aligned by doc_id
    doc_id_list = list(all_doc_ids)
    raw_bm25 = [bm25_dict.get(did, 0.0) for did in doc_id_list]
    raw_vector = [vector_dict.get(did, 0.0) for did in doc_id_list]
    
    # Normalize
    if normalization == "min-max":
        norm_bm25 = min_max_normalize(raw_bm25)
        norm_vector = min_max_normalize(raw_vector)
    else:  # z-score
        norm_bm25 = z_score_normalize(raw_bm25)
        norm_vector = z_score_normalize(raw_vector)
        
    # Calculate hybrid scores
    hybrid_results = []
    for i, doc_id in enumerate(doc_id_list):
        b_score = norm_bm25[i]
        v_score = norm_vector[i]
        
        hybrid_score = (alpha * b_score) + ((1 - alpha) * v_score)
        
        hybrid_results.append({
            "doc_id": doc_id,
            "bm25_score": float(b_score),
            "vector_score": float(v_score),
            "hybrid_score": float(hybrid_score)
        })
        
    # Sort descending by hybrid score
    hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return hybrid_results
