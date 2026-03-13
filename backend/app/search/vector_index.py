import hnswlib
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class VectorIndex:
    def __init__(self, index_dir: str = "data/index/vector", space: str = "cosine"):
        """
        Initialize the hnswlib vector index.
        Typically for sentence-transformers, cosine similarity is best.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "hnswlib.bin"
        self.doc_ids_path = self.index_dir / "index_doc_ids.npy"
        self.metadata_path = self.index_dir / "index_metadata.json"
        
        self.space = space
        self.index = None
        self.doc_ids = None

    def build(self, embeddings: np.ndarray, doc_ids: Optional[np.ndarray] = None,
              embedding_model: str = "all-MiniLM-L6-v2", corpus_hash: str = "") -> None:
        """
        Build the hnswlib similarity index using given embeddings.
        Provide doc_ids to keep a mapping to the original document IDs.
        """
        if embeddings.size == 0:
            logger.warning("No embeddings provided to build the vector index.")
            return

        num_elements, dim = embeddings.shape
        # Initialize hnswlib index
        self.index = hnswlib.Index(space=self.space, dim=dim)
        
        # ef_construction defines a construction time/accuracy trade-off
        # M defines the maximum number of outgoing connections in the graph
        self.index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # hnswlib requires integer labels
        integer_labels = np.arange(num_elements)
        self.index.add_items(embeddings, integer_labels)
        
        # Store the mapping back to original doc_ids
        if doc_ids is not None:
            self.doc_ids = np.array(doc_ids)
        else:
            self.doc_ids = integer_labels
            
        logger.info(f"Built vector index with {num_elements} elements.")
        self._save_metadata(embedding_model, dim, num_elements, corpus_hash)
        self.save()

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector index and return the most similar documents.
        """
        if self.index is None:
            logger.error("Vector index is not initialized.")
            return []

        # query_vector might be 1D, hnswlib expects 2D (batch size, dim)
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        # Ensure k is not greater than the number of elements in the index
        k = min(top_k, self.index.element_count)
        if k == 0:
            return []

        # Set ef for query (should be >= k)
        self.index.set_ef(max(50, k))
        
        labels, distances = self.index.knn_query(query_vector, k=k)
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            doc_id = self.doc_ids[label] if self.doc_ids is not None else str(label)
            
            # For cosine distance, similarity is 1.0 - distance
            similarity = 1.0 - distance if self.space == 'cosine' else distance
            
            results.append({
                "doc_id": doc_id,
                "score": float(similarity),
                "distance": float(distance)
            })
            
        return results

    def save(self) -> None:
        """Save the hnswlib configuration and doc mappings to disk."""
        if self.index is None:
            return
            
        self.index.save_index(str(self.index_path))
        if self.doc_ids is not None:
            np.save(self.doc_ids_path, self.doc_ids)
            
        logger.info(f"Saved vector index to {self.index_dir}")

    def _save_metadata(self, embedding_model: str, dim: int, num_elements: int, corpus_hash: str) -> None:
        """Persist index metadata as JSON."""
        metadata = {
            "embedding_model": embedding_model,
            "embedding_dimension": dim,
            "num_elements": num_elements,
            "corpus_hash": corpus_hash,
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved index metadata to {self.metadata_path}")

    def load_metadata(self) -> Dict[str, Any]:
        """Load and return index metadata, or empty dict if missing."""
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_metadata(self, expected_model: str, expected_dim: int) -> None:
        """
        Validate that the stored index metadata matches the currently
        configured embedding model and dimension.

        Raises:
            ValueError  if there is a mismatch.
        """
        meta = self.load_metadata()
        if not meta:
            logger.warning("No index metadata found — skipping validation.")
            return

        stored_model = meta.get("embedding_model", "unknown")
        stored_dim = meta.get("embedding_dimension", -1)

        errors = []
        if stored_model != expected_model:
            errors.append(
                f"Embedding model mismatch: index was built with '{stored_model}' "
                f"but the current model is '{expected_model}'."
            )
        if stored_dim != expected_dim:
            errors.append(
                f"Embedding dimension mismatch: index has dim={stored_dim} "
                f"but the current model produces dim={expected_dim}."
            )

        if errors:
            msg = (
                "\n".join(errors)
                + "\n\n⚠️  The vector index must be REBUILT with the current "
                "embedding model.\n"
                "   Run:  python -m backend.app.index --input data/processed/docs.jsonl"
            )
            logger.error(msg)
            raise ValueError(msg)

    def load(self, dim: int, max_elements: int = 0) -> bool:
        """
        Load the index. Requires the dimension `dim` of the vectors.
        """
        if not self.index_path.exists():
            logger.error(f"Index file not found at {self.index_path}")
            return False
            
        try:
            self.index = hnswlib.Index(space=self.space, dim=dim)
            self.index.load_index(str(self.index_path), max_elements=max_elements)
            
            if self.doc_ids_path.exists():
                self.doc_ids = np.load(self.doc_ids_path, allow_pickle=True)
                
            logger.info(f"Loaded vector index with {self.index.element_count} elements.")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return False
