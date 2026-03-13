from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
import time

from backend.app.search.bm25 import BM25Index
from backend.app.search.embeddings import EmbeddingPipeline
from backend.app.search.vector_index import VectorIndex
from backend.app.search.hybrid import hybrid_rank
from backend.app.logging.logger import SQLiteLogger

logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid Search API")

# Initialize global search components (lazy loaded on startup)
bm25_index = None
embedding_pipeline = None
vector_index = None
search_logger = None

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
    global bm25_index, embedding_pipeline, vector_index, search_logger
    
    logger.info("Loading search indices and components...")
    try:
        search_logger = SQLiteLogger()
        
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

@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """Return basic request metrics."""
    if search_logger is None:
        raise HTTPException(status_code=503, detail="Logger not initialized")
    return search_logger.get_metrics()

@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest):
    """
    Perform a hybrid search combining BM25 and vector search.
    """
    start_time = time.time()
    
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
        
        # Log request
        latency_ms = (time.time() - start_time) * 1000.0
        if search_logger:
            search_logger.log_search(
                query=query_text,
                latency_ms=latency_ms,
                top_k=request.top_k,
                alpha=request.alpha,
                result_count=len(top_results)
            )
        
        # Convert to Pydantic models
        return [SearchResult(**res) for res in top_results]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
