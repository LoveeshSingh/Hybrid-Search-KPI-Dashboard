from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import logging
import time

from backend.app.search.hybrid_search import HybridSearch
from backend.app.logging.logger import SQLiteLogger
from backend.app.db.query_store import QueryStore
import subprocess

logger = logging.getLogger(__name__)

app = FastAPI(title="Hybrid Search API")

# Initialize global search components (lazy loaded on startup)
hybrid_search = None
search_logger = None
query_store = None

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
    global hybrid_search, search_logger, query_store
    
    logger.info("Loading search indices and components...")
    try:
        search_logger = SQLiteLogger()
        query_store = QueryStore()
        
        hybrid_search = HybridSearch()
        hybrid_search.load()
        
        logger.info("Search indices loaded (if available).")
    except Exception as e:
        logger.error(f"Error loading indices: {e}")

def get_git_commit():
    """Retrieve the current git commit hash if available."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"

@app.get("/health")
def health_check():
    """Service health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "commit": get_git_commit()
    }

@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """Return search system metrics derived from the query log."""
    if query_store is None:
        raise HTTPException(status_code=503, detail="QueryStore not initialized")
    return query_store.get_metrics()

@app.post("/search", response_model=List[SearchResult])
def search(request: SearchRequest):
    """
    Perform a hybrid search combining BM25 and vector search.
    """
    start_time = time.time()
    
    if hybrid_search is None or not hybrid_search.bm25_index.bm25 or not hybrid_search.vector_index.index:
        raise HTTPException(status_code=503, detail="Indices not initialized or empty. Please run the indexing pipeline.")

    query_text = request.query
    
    try:
        top_results = hybrid_search.search(
            query=query_text,
            top_k=request.top_k,
            alpha=request.alpha
        )
        
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
        if query_store:
            query_store.log_query(
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
