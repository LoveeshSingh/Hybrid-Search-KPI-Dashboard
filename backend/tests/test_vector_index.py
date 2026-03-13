import pytest
import numpy as np
from tempfile import TemporaryDirectory
from backend.app.search.vector_index import VectorIndex

def test_vector_index_build_and_query():
    with TemporaryDirectory() as temp_dir:
        index = VectorIndex(index_dir=temp_dir)
        
        # 3 documents, dimension=4
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],  # doc_A
            [0.0, 1.0, 0.0, 0.0],  # doc_B
            [0.0, 0.0, 1.0, 0.0],  # doc_C
        ], dtype=np.float32)
        doc_ids = np.array(["doc_A", "doc_B", "doc_C"])
        
        index.build(embeddings, doc_ids)
        
        # Query matching doc_B perfectly and slightly doc_C
        query_vector = np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32)
        results = index.query(query_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0]['doc_id'] == "doc_B"
        assert results[1]['doc_id'] == "doc_C"
        
        # Closer match should have higher cosine similarity score
        assert results[0]['score'] > results[1]['score']
        # Closer match should have lower distance
        assert results[0]['distance'] < results[1]['distance']

def test_vector_index_save_load():
    with TemporaryDirectory() as temp_dir:
        index1 = VectorIndex(index_dir=temp_dir)
        embeddings = np.array([
            [0.5, 0.5],
            [-0.5, 0.5]
        ], dtype=np.float32)
        doc_ids = np.array(["d1", "d2"])
        
        index1.build(embeddings, doc_ids)
        
        assert index1.index_path.exists()
        assert index1.doc_ids_path.exists()
        
        index2 = VectorIndex(index_dir=temp_dir)
        # load requires knowing dimensions and optionally max_elements
        success = index2.load(dim=2, max_elements=2)
        
        assert success is True
        assert index2.index.element_count == 2
        assert len(index2.doc_ids) == 2
        
        query_vector = np.array([0.4, 0.6], dtype=np.float32)
        results = index2.query(query_vector, top_k=1)
        assert results[0]['doc_id'] == "d1"
