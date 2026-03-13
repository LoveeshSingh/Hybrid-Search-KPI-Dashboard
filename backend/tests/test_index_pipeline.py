import pytest
import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from backend.app.index import build_indices

def test_build_indices_pipeline():
    with TemporaryDirectory() as temp_dir:
        # 1. Create dummy input JSONL
        input_file = Path(temp_dir) / "test_docs.jsonl"
        docs = [
            {"doc_id": "d1", "title": "T1", "text": "This is doc one", "source": "f1.txt"},
            {"doc_id": "d2", "title": "T2", "text": "This is doc two", "source": "f2.txt"}
        ]
        
        with open(input_file, 'w', encoding='utf-8') as f:
            for d in docs:
                f.write(json.dumps(d) + '\n')
                
        # 2. Patch the artifact directories to write into temp_dir
        # We also need to patch the metadata path so it saves natively here
        bm25_dir = os.path.join(temp_dir, "bm25")
        vec_dir = os.path.join(temp_dir, "vector")
        meta_path = os.path.join(temp_dir, "metadata.json")
        
        with patch('backend.app.index.BM25Index') as mock_bm25_class, \
             patch('backend.app.index.EmbeddingPipeline') as mock_embedder_class, \
             patch('backend.app.index.VectorIndex') as mock_vec_class, \
             patch('backend.app.index.Path') as mock_path:
             
            # Configure path mocking for metadata
            # Only mock the metadata path resolution, leave input_file alone
            def path_side_effect(arg):
                if arg == "data/index/metadata.json":
                    p = Path(meta_path)
                    # mock mkdir so it doesn't try to create parent arbitrarily
                    p.parent.mkdir = lambda parents, exist_ok: None
                    return p
                return Path(arg)
                
            mock_path.side_effect = path_side_effect
            
            # Setup mock instances
            mock_bm25_instance = mock_bm25_class.return_value
            mock_embedder_instance = mock_embedder_class.return_value
            mock_vec_instance = mock_vec_class.return_value
            
            import numpy as np
            # embedder returns dummy embeddings
            mock_embedder_instance.embed_documents.return_value = (np.zeros((2, 384)), np.array(['d1', 'd2']))
             
            # Run pipeline
            build_indices(str(input_file))
            
            # Assert calls
            mock_bm25_instance.build.assert_called_once()
            mock_embedder_instance.embed_documents.assert_called_once()
            mock_vec_instance.build.assert_called_once()
            
            # Verify metadata was created
            assert os.path.exists(meta_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                assert meta["num_documents"] == 2
                assert meta["embedding_model"] == "all-MiniLM-L6-v2"
                assert "corpus_hash" in meta
                assert "build_timestamp" in meta
