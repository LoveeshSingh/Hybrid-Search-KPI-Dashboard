import pytest
import numpy as np
from tempfile import TemporaryDirectory
from backend.app.search.embeddings import EmbeddingPipeline

@pytest.fixture
def sample_docs():
    return [
        {
            "doc_id": "doc1",
            "title": "Machine Learning",
            "text": "Machine learning is a field of artificial intelligence."
        },
        {
            "doc_id": "doc2",
            "title": "Deep Learning Models",
            "text": "Neural networks and deep learning models use multiple layers."
        }
    ]

def test_embed_documents_and_query(sample_docs):
    with TemporaryDirectory() as temp_dir:
        # Initialize pipeline pointing to temp directory
        pipeline = EmbeddingPipeline(index_dir=temp_dir)
        
        # Generate embeddings and save
        embeddings, doc_ids = pipeline.embed_documents(sample_docs, save=True)
        
        # Verify sizes and shapes
        assert len(doc_ids) == 2
        assert doc_ids[0] == "doc1"
        assert doc_ids[1] == "doc2"
        
        assert embeddings.shape[0] == 2
        
        # Dimension size default for all-MiniLM-L6-v2 is 384
        assert embeddings.shape[1] == 384
        
        # Verify file artifacts were created
        assert (pipeline.embeddings_path).exists()
        assert (pipeline.doc_ids_path).exists()
        
        # Test exact loaded items
        loaded_embeddings, loaded_doc_ids = pipeline.load_embeddings()
        np.testing.assert_array_equal(embeddings, loaded_embeddings)
        np.testing.assert_array_equal(doc_ids, loaded_doc_ids)
        
        # Test query embedding
        query_emb = pipeline.embed_query("artificial intelligence")
        
        # Should return a 1D vector of same dimensions
        assert len(query_emb.shape) == 1
        assert len(query_emb) == 384
