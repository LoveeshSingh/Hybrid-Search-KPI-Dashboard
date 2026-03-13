import pytest
import json
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import MagicMock

from backend.app.search.vector_index import VectorIndex


def test_metadata_saved_on_build():
    """build() should write index_metadata.json alongside the index."""
    with TemporaryDirectory() as tmp:
        idx = VectorIndex(index_dir=tmp)
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        ids = np.array(["a", "b"])

        idx.build(emb, doc_ids=ids, embedding_model="test-model", corpus_hash="abc123")

        meta_path = Path(tmp) / "index_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["embedding_model"] == "test-model"
        assert meta["embedding_dimension"] == 2
        assert meta["num_elements"] == 2
        assert meta["corpus_hash"] == "abc123"
        assert "build_timestamp" in meta


def test_validate_metadata_pass():
    """Validation should succeed when model and dim match."""
    with TemporaryDirectory() as tmp:
        idx = VectorIndex(index_dir=tmp)

        # Write matching metadata
        meta = {"embedding_model": "all-MiniLM-L6-v2", "embedding_dimension": 384}
        with open(idx.metadata_path, "w") as f:
            json.dump(meta, f)

        # Should NOT raise
        idx.validate_metadata(expected_model="all-MiniLM-L6-v2", expected_dim=384)


def test_validate_metadata_model_mismatch():
    """Changing the model name after build should raise ValueError."""
    with TemporaryDirectory() as tmp:
        idx = VectorIndex(index_dir=tmp)

        meta = {"embedding_model": "all-MiniLM-L6-v2", "embedding_dimension": 384}
        with open(idx.metadata_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="model mismatch"):
            idx.validate_metadata(
                expected_model="all-mpnet-base-v2",  # different model!
                expected_dim=768,                     # different dimension!
            )


def test_validate_metadata_dimension_mismatch():
    """Dimension-only mismatch should also raise."""
    with TemporaryDirectory() as tmp:
        idx = VectorIndex(index_dir=tmp)

        meta = {"embedding_model": "all-MiniLM-L6-v2", "embedding_dimension": 384}
        with open(idx.metadata_path, "w") as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="dimension mismatch"):
            idx.validate_metadata(
                expected_model="all-MiniLM-L6-v2",
                expected_dim=768,  # wrong dim
            )


def test_hybrid_search_load_detects_mismatch():
    """
    Simulate a failure: index was built with model A (dim=384),
    but HybridSearch now uses model B (dim=768).
    load() should return False with a clear error.
    """
    mock_bm25 = MagicMock()
    mock_bm25.model_path.exists.return_value = True

    mock_embedder = MagicMock()
    mock_embedder.model_name = "all-mpnet-base-v2"
    mock_embedder.get_dimension.return_value = 768

    with TemporaryDirectory() as tmp:
        vi = VectorIndex(index_dir=tmp)

        # Build a real small index with model A metadata
        emb = np.array([[0.5, 0.5]], dtype=np.float32)
        vi.build(emb, doc_ids=np.array(["d1"]), embedding_model="all-MiniLM-L6-v2")

        # Now wrap in HybridSearch with the WRONG embedder
        from backend.app.search.hybrid_search import HybridSearch

        hs = HybridSearch(
            bm25_index=mock_bm25,
            embedding_pipeline=mock_embedder,
            vector_index=vi,
        )

        # load should still physically load the index,
        # but validation should catch the mismatch and return False
        result = hs.load(vector_dim=2)
        assert result is False
