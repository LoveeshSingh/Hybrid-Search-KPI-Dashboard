import logging
from typing import List, Dict, Any

from backend.app.search.bm25 import BM25Index
from backend.app.search.embeddings import EmbeddingPipeline
from backend.app.search.vector_index import VectorIndex
from backend.app.search.hybrid import hybrid_rank

logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(self, bm25_index: BM25Index = None, embedding_pipeline: EmbeddingPipeline = None, vector_index: VectorIndex = None):
        """
        Initialize the Hybrid Search service.
        Optionally accept pre-loaded dependencies, otherwise instantiates empty ones to load.
        """
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_pipeline = embedding_pipeline or EmbeddingPipeline()
        self.vector_index = vector_index or VectorIndex()

    def load(self, vector_dim: int = 384) -> bool:
        """
        Load all search indices from disk.
        Returns True if successful, False otherwise.
        """
        try:
            logger.info("Loading BM25 index...")
            if self.bm25_index.model_path.exists():
                self.bm25_index.load()
            else:
                logger.warning("BM25 index path not found.")
                return False

            logger.info("Loading Vector index...")
            if self.vector_index.index_path.exists():
                self.vector_index.load(dim=vector_dim)
            else:
                logger.warning("Vector index path not found.")
                return False

            # Validate that the index was built with the same model
            expected_model = self.embedding_pipeline.model_name
            expected_dim = self.embedding_pipeline.get_dimension()
            self.vector_index.validate_metadata(expected_model, expected_dim)
                
            logger.info("Hybrid Search indices loaded and validated successfully.")
            return True
        except ValueError as ve:
            logger.error(f"Index validation failed: {ve}")
            return False
        except Exception as e:
            logger.error(f"Failed to load Hybrid Search indices: {e}")
            return False

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Execute a hybrid search query.
        
        Args:
            query: The search text
            top_k: Number of final results to return
            alpha: Weight of BM25 (0.0=Vector only, 1.0=BM25 only, 0.5=Equal)
            
        Returns:
            List of dictionaries containing doc_id, bm25_score, vector_score, and hybrid_score.
        """
        if not self.bm25_index.bm25 or not self.vector_index.index:
            logger.error("Indices are not loaded. Call load() first.")
            return []

        logger.debug(f"Executing hybrid search for: '{query}' (top_k={top_k}, alpha={alpha})")
        
        # 1. Lexical retrieval
        # Fetching more than top_k for better candidate overlap before routing
        fetch_k = max(20, top_k * 2)
        bm25_results = self.bm25_index.query(query, top_k=fetch_k)
        
        # 2. Semantic retrieval
        query_vector = self.embedding_pipeline.embed_query(query)
        vector_results = self.vector_index.query(query_vector, top_k=fetch_k)
        
        # 3. Hybrid fusion
        results = hybrid_rank(
            bm25_results=bm25_results,
            vector_results=vector_results,
            alpha=alpha,
            normalization="min-max"
        )
        
        return results[:top_k]
