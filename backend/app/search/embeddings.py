import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "data/index/vector"):
        """
        Initialize the embedding pipeline using a lightweight CPU-friendly model.
        Default model 'all-MiniLM-L6-v2' is small, fast, and good for general semantic search.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.doc_ids_path = self.index_dir / "doc_ids.npy"
        
        self.model_name = model_name
        
        # device="cpu" enforces CPU-only processing per requirements
        logger.info(f"Loading sentence-transformer model: {model_name} on CPU")
        self.model = SentenceTransformer(model_name, device='cpu')

    def get_dimension(self) -> int:
        """Return the embedding vector dimension of the loaded model."""
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, docs: List[Dict[str, Any]], save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for a list of documents. 
        Combines title and text for semantic meaning.
        """
        if not docs:
            logger.warning("No documents provided to embed.")
            return np.array([]), np.array([])
            
        doc_ids = []
        texts_to_embed = []
        
        for doc in docs:
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
                
            title = doc.get("title", "")
            text = doc.get("text", "")
            
            # Combine fields
            combined_text = f"{title}. {text}".strip()
            
            doc_ids.append(doc_id)
            texts_to_embed.append(combined_text)
            
        logger.info(f"Encoding {len(texts_to_embed)} documents...")
        # Encode returns a numpy array by default
        embeddings = self.model.encode(texts_to_embed, show_progress_bar=False)
        doc_ids_array = np.array(doc_ids)
        
        if save:
            self.save_embeddings(embeddings, doc_ids_array)
            
        return embeddings, doc_ids_array

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Generate a single embedding vector for a search query.
        """
        return self.model.encode(query_text, show_progress_bar=False)

    def save_embeddings(self, embeddings: np.ndarray, doc_ids: np.ndarray) -> None:
        """Save the generated numpy arrays to disk."""
        np.save(self.embeddings_path, embeddings)
        np.save(self.doc_ids_path, doc_ids)
        logger.info(f"Saved {len(doc_ids)} embeddings to {self.index_dir}")

    def load_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load embeddings and doc IDs from disk."""
        if not self.embeddings_path.exists() or not self.doc_ids_path.exists():
            logger.error(f"Embedding files not found in {self.index_dir}")
            return np.array([]), np.array([])
            
        embeddings = np.load(self.embeddings_path)
        doc_ids = np.load(self.doc_ids_path)
        logger.info(f"Loaded {len(doc_ids)} embeddings from {self.index_dir}")
        
        return embeddings, doc_ids
