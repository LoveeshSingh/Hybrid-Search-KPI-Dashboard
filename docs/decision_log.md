# Decision Log

Records significant technical decisions made during the development of the Hybrid Search + KPI Dashboard.

## Decisions

### 1. Project Structure
- Context: The project required clear separation between backend logic, frontend UI, datasets, and documentation.
- Decision: Adopted a modular directory structure with `backend/`, `frontend/`, `data/`, and `docs/`.
- Reasoning: Improves maintainability and separates ingestion, indexing, API, and evaluation logic.
- Consequences: Easier testing and clearer architecture.

### 2. Hybrid Search Architecture
- Context: The system required combining lexical and semantic retrieval.
- Decision: Implemented hybrid search using BM25 and vector similarity.
- Implementation:

hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_vector

- Reasoning: BM25 captures keyword relevance while embeddings capture semantic similarity.
- Consequences: Improved ranking quality compared to single-method retrieval.

### 3. BM25 Library Choice
- Context: Needed a lightweight lexical search implementation.
- Decision: Used the `rank-bm25` library.
- Reasoning: Simple, CPU-friendly, and easy to integrate with Python.
- Consequences: No external search engine required.

### 4. Embedding Model Selection
- Context: Semantic search requires document embeddings.
- Decision: Used `sentence-transformers` with the `all-MiniLM-L6-v2` model.
- Reasoning: Small model that runs efficiently on CPU while producing good semantic representations.
- Consequences: Fast indexing and query embedding generation.

### 5. Vector Search Engine
- Context: Needed efficient nearest-neighbor search for embeddings.
- Decision: Used `hnswlib`.
- Reasoning: High-performance approximate nearest neighbor search with minimal dependencies.
- Consequences: Fast vector retrieval suitable for CPU environments.

### 6. Hybrid Score Normalization
- Context: BM25 and vector scores exist on different scales.
- Decision: Implemented normalization before combining scores.
- Methods:
- Min-max normalization
- Z-score normalization
- Reasoning: Makes scores comparable for hybrid ranking.

### 7. Hybrid Search Service Layer
- Context: API endpoints initially interacted with BM25 and vector logic directly.
- Decision: Introduced a `HybridSearch` service class.
- Reasoning: Centralizes retrieval logic and simplifies API code.
- Consequences: Cleaner architecture and easier testing.

### 8. Query Logging Strategy
- Context: The dashboard required analytics about search activity.
- Decision: Implemented `QueryStore` using SQLite.
- Stored fields:

request_id
query
latency_ms
top_k
alpha
result_count
timestamp

- Reasoning: SQLite is lightweight and sufficient for local analytics.
- Consequences: Enables metrics endpoint and dashboard visualizations.

### 9. Evaluation Framework
- Context: Search quality needed measurable metrics.
- Decision: Implemented evaluation metrics:
- nDCG@10
- Recall@10
- MRR@10

Results stored in:

data/metrics/experiments.csv

- Reasoning: Enables comparison between experiments and parameter changes.

### 10. Dashboard Technology
- Context: Needed a UI for search testing and metrics visualization.
- Decision: Used Streamlit instead of React.
- Reasoning: Faster development and easy integration with Python backend.
- Consequences: Rapid implementation of dashboard pages.

### 11. Reproducible Startup
- Context: Reviewers must be able to run the system easily.
- Decision: Implemented a startup script `up.sh`.
- Responsibilities:
- create virtual environment
- install dependencies
- run ingestion if needed
- build indexes if missing
- start API and dashboard

- Consequences: Entire system can be launched with a single command.

### 12. Vector Index Metadata Validation
- Context: Changing embedding models can break vector indexes.
- Decision: Store metadata alongside vector index.

embedding_model
embedding_dimension
corpus_hash
build_timestamp

- Reasoning: Ensures index compatibility with embedding configuration.

### 13. Database Schema Migration
- Context: Query logging schema changed during development.
- Decision: Added schema versioning and automatic migration.
- Example change: Added `user_agent` column to query logs.
- Reasoning: Prevents runtime errors when schema evolves.

### 14. Hybrid Score Edge Case Fix
- Context: Min-max normalization caused divide-by-zero when all scores were identical.
- Decision: Return `0.0` when scores are equal instead of dividing by zero.
- Consequences: Prevents NaN scores and ensures stable hybrid ranking.