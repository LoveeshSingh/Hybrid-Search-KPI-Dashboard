import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class BM25Index:
    def __init__(self, index_dir: str = "data/index/bm25"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.index_dir / "bm25_model.pkl"
        self.docs_path = self.index_dir / "docs.pkl"
        
        self.bm25 = None
        self.docs: List[Dict[str, Any]] = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Strict simple tokenization for BM25 (lowercasing and splitting by space)."""
        return text.lower().split()

    def build(self, docs: List[Dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of document dictionaries.
        Indexes the text by combining title + text.
        """
        if not docs:
            logger.warning("No documents provided to build the index.")
            return

        self.docs = docs
        tokenized_corpus = []
        
        for doc in docs:
            # Combine title and text for semantic context
            combined_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokenized_corpus.append(self._tokenize(combined_text))
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Built BM25 index with {len(docs)} documents.")
        
        self.save()

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the BM25 index and return the top_k matching documents with scores.
        """
        if self.bm25 is None:
            logger.error("BM25 model is not initialized. Please build or load the index first.")
            return []

        tokenized_query = self._tokenize(query_text)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top indices sorted by score descending
        # np.argsort could be used, but python's sorted with indices avoids explicitly importing numpy just for this list comprehension.
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            # Only include documents that actually scored above 0 to prevent returning completely irrelevant docs
            if doc_scores[idx] > 0:
                result_doc = self.docs[idx].copy()
                result_doc['score'] = doc_scores[idx]
                results.append(result_doc)
                
        return results

    def save(self) -> None:
        """Save the BM25 model and document store to disk."""
        if self.bm25 is None:
            return
            
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.bm25, f)
            
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.docs, f)
            
        logger.info(f"Saved BM25 index to {self.index_dir}")

    def load(self) -> bool:
        """Load the BM25 model and document store from disk."""
        if not self.model_path.exists() or not self.docs_path.exists():
            logger.error(f"Index files not found in {self.index_dir}")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                self.bm25 = pickle.load(f)
                
            with open(self.docs_path, 'rb') as f:
                self.docs = pickle.load(f)
                
            logger.info(f"Loaded BM25 index with {len(self.docs)} documents.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
