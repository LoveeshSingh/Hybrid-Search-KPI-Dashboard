import pytest
from tempfile import TemporaryDirectory
from backend.app.search.bm25 import BM25Index

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
        },
        {
            "doc_id": "doc3",
            "title": "Python Programming",
            "text": "Python is a popular programming language for artificial intelligence."
        }
    ]

def test_bm25_build_and_query(sample_docs):
    with TemporaryDirectory() as temp_dir:
        # Initialize and build
        index = BM25Index(index_dir=temp_dir)
        index.build(sample_docs)
        
        # Test Query 1 - should strongly match doc1
        results = index.query("machine learning", top_k=3)
        assert len(results) > 0
        assert results[0]['doc_id'] == "doc1"
        assert 'score' in results[0]
        
        # Test Query 2 - should strongly match doc3
        results = index.query("python programming", top_k=3)
        assert len(results) > 0
        assert results[0]['doc_id'] == "doc3"
        
        # Test Query 3 - Artificial Intelligence should appear in doc1 and doc3
        results = index.query("artificial intelligence", top_k=3)
        assert len(results) == 2  # docs 1 and 3 have it
        docs_found = {r['doc_id'] for r in results}
        assert "doc1" in docs_found
        assert "doc3" in docs_found
        assert "doc2" not in docs_found

def test_bm25_save_and_load(sample_docs):
    with TemporaryDirectory() as temp_dir:
        # Build and save
        index1 = BM25Index(index_dir=temp_dir)
        index1.build(sample_docs)
        
        # Verify files were created
        assert (index1.model_path).exists()
        assert (index1.docs_path).exists()
        
        # Load in new instance
        index2 = BM25Index(index_dir=temp_dir)
        success = index2.load()
        
        assert success is True
        assert len(index2.docs) == 3
        
        # Query loaded instance
        results = index2.query("deep learning", top_k=1)
        assert results[0]['doc_id'] == "doc2"
