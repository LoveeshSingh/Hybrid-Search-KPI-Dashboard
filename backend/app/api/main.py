from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging

from backend.app.search.bm25 import BM25Index
from backend.app.search.embeddings import EmbeddingPipeline
from backend.app.search.vector_index import VectorIndex
from backend.app.search.hybrid import hybrid_rank

logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid Search API")

# Initialize global search components (lazy loaded on startup)
bm25_index = None
embedding_pipeline = None
vector_index = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

class SearchResult(BaseModel):
    doc_id: str
    bm25_score: float
    vector_score: float
    hybrid_score: float

@app.on_event("startup")
async def startup_event():
    """Load indices on startup."""
    global bm25_index, embedding_pipeline, vector_index
    
    logger.info("Loading search indices...")
    try:
        bm25_index = BM25Index()
        # Non-fatal if missing, allows tests to pass and API to start empty
        if bm25_index.model_path.exists():
            bm25_index.load()
            
        embedding_pipeline = EmbeddingPipeline()
        
        vector_index = VectorIndex()
        # dimension 384 for all-MiniLM-L6-v2
        if vector_index.index_path.exists():
            vector_index.load(dim=384)
            
        logger.info("Search indices loaded (if available).")
    except Exception as e:
        logger.error(f"Error loading indices: {e}")

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest):
    """
    Perform a hybrid search combining BM25 and vector search.
    """
    if bm25_index is None or bm25_index.bm25 is None:
        raise HTTPException(status_code=503, detail="BM25 index not initialized or empty.")
        
    if vector_index is None or vector_index.index is None:
        raise HTTPException(status_code=503, detail="Vector index not initialized or empty.")

    query_text = request.query
    
    try:
        # 1. Lexical Search
        bm25_results = bm25_index.query(query_text, top_k=request.top_k * 2) # Fetch more for re-ranking
        
        # 2. Semantic Search
        query_vector = embedding_pipeline.embed_query(query_text)
        vector_results = vector_index.query(query_vector, top_k=request.top_k * 2)
        
        # 3. Hybrid Ranking
        hybrid_results = hybrid_rank(
            bm25_results, 
            vector_results, 
            alpha=request.alpha, 
            normalization="min-max"
        )
        
        # 4. Truncate to requested top_k
        top_results = hybrid_results[:request.top_k]
        
        # Convert to Pydantic models
        return [SearchResult(**res) for res in top_results]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
