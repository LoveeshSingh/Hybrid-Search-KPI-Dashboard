Note:
This log records prompts used with a coding assistant during development.
Although the assignment refers to "Codex", the assistant used was Antigravity.
The same granular prompting protocol was followed.

## Prompts

### Step 1: Create Repo Structure
**Prompt:**
```text
You are acting as an engineering assistant helping me build an end-to-end Hybrid Search + KPI Dashboard system for a technical assessment.

The system must follow strict engineering rules.

Project Overview:
We are building a CPU-only hybrid search system using:
- Python 3.11
- FastAPI backend
- rank-bm25 for lexical search
- sentence-transformers for embeddings
- FAISS (CPU) or hnswlib for vector search
- SQLite for logs and metrics
- React + Vite or Streamlit for dashboard
- pytest for testing

The system must support:
1. Data ingestion
2. BM25 index
3. Vector embedding index
4. Hybrid search API
5. Evaluation harness (nDCG, Recall, MRR)
6. KPI dashboard
7. Structured logging
8. Metrics endpoint
9. Unit tests
10. Reproducible startup via ./up.sh

CRITICAL RULES YOU MUST FOLLOW:

1. Work in SMALL INCREMENTS
Each response must implement only ONE small unit of work.

Examples of acceptable units:
- create one file
- implement one class
- implement one endpoint
- write one test
- modify one script

Never generate large multi-module systems in one step.

2. ALWAYS SPECIFY FILE PATHS
Every code output must clearly state:
- file path
- new file vs modification

Example format:
File: backend/app/search/bm25.py

3. AFTER EVERY STEP PROVIDE:
- code
- explanation of what changed
- git commit message
- entry for docs/codex_log.md

4. ALWAYS INCLUDE TESTS
If code logic is added, include pytest tests.

5. ALWAYS MAINTAIN THESE DOCUMENTS
Update when relevant:

docs/codex_log.md
docs/decision_log.md
docs/break_fix_log.md

6. EVERY STEP MUST BE COMMIT-READY

After each task provide:

Git command example:

git add <files>
git commit -m "implement BM25 index with rank-bm25"

7. FOLLOW THIS REPO STRUCTURE

repo-root/

backend/
    app/
        api/
        search/
        ingestion/
        evaluation/
        logging/
        db/
    tests/

frontend/

data/
    raw/
    processed/
    index/
    eval/
    metrics/

docs/
    architecture.md
    codex_log.md
    decision_log.md
    break_fix_log.md

up.sh
requirements.txt
README.md

8. ALL CODE MUST RUN ON CPU
Do not introduce GPU dependencies.

9. NO HARD-CODED PATHS
Everything must work from repo root.

10. ASSUME REVIEWER WILL RUN:

./up.sh

on a clean machine.

11. WHEN IMPLEMENTING FEATURES:
Always include acceptance criteria and how it will be validated.

12. WHEN GENERATING PROMPTS FOR CODEX:
Use granular prompt style and include them in docs/codex_log.md.

13. MAINTAIN CLEAN ENGINEERING PRACTICES
- type hints
- structured logging
- clear modular design
- defensive programming

14. IF A STEP DEPENDS ON A PREVIOUS STEP:
Explain the dependency.

15. WHEN WRITING SCRIPTS:
Ensure they are idempotent (safe to run multiple times).

Workflow Rules:
We will proceed step-by-step in this order:

1. Create repo structure
2. Setup Python environment
3. Data ingestion pipeline
4. BM25 index
5. Embedding pipeline
6. Vector index
7. Hybrid scoring
8. Search API
9. Logging + SQLite
10. Evaluation harness
11. Dashboard
12. Metrics endpoint
13. up.sh automation
14. Tests
15. Failure scenarios

For each step provide:
- file changes
- commit message
- codex log entry
- instructions to test

When unsure, ask before implementing.
```

**Summary of response:**
Created the basic directory structure and placeholder documentation files to organize the project according to requirements.

**Files generated:**
- `docs/architecture.md`
- `docs/codex_log.md`
- `docs/decision_log.md`
- `README.md`
- `up.sh`

### Step 2: Setup Python Environment
**Prompt:**
```text
We're starting the Hybrid Search + KPI Dashboard project and I want to first set up the basic repository structure before writing any logic.

Create the following project layout:

backend/
app/
api/
search/
ingestion/
evaluation/
logging/
db/
tests/

frontend/

data/
raw/
processed/
index/
eval/
metrics/

docs/

Also create these files in the repo root:

requirements.txt
README.md
up.sh

Inside docs/, create:

architecture.md
codex_log.md
decision_log.md
break_fix_log.md

Add `__init__.py` files so the backend folders work as Python packages:

backend/app/**init**.py
backend/app/api/**init**.py
backend/app/search/**init**.py
backend/app/ingestion/**init**.py
backend/app/evaluation/**init**.py
backend/app/logging/**init**.py
backend/app/db/**init**.py

For requirements.txt include:

fastapi
uvicorn
rank-bm25
sentence-transformers
hnswlib
numpy
pandas
pytest
streamlit

Show me:

1. the final folder tree
2. the contents of requirements.txt
3. a minimal README placeholder
4. the git commands I should run for the first commit
```

**Summary of response:**
Created `__init__.py` files to initialize the backend modules and defined project dependencies in `requirements.txt`.

**Files generated:**
- `backend/app/__init__.py`
- `backend/app/api/__init__.py`
- `backend/app/search/__init__.py`
- `backend/app/ingestion/__init__.py`
- `backend/app/evaluation/__init__.py`
- `backend/app/logging/__init__.py`
- `backend/app/db/__init__.py`
- `requirements.txt`

### Step 3: Data Ingestion Pipeline
**Prompt:**
```text
Next I want to implement the document ingestion pipeline.

Create a module that loads documents from a folder and converts them into a normalized JSONL dataset.

File to create:
backend/app/ingestion/ingest.py

Functionality:

The script should read all `.txt` and `.md` files from an input directory and convert them into a JSONL file.

Each document entry should contain:

doc_id
title
text
source
created_at

Requirements:

* doc_id should be generated from the filename
* title can be the first line of the document
* text should contain the remaining content
* source should store the original file path
* created_at should store the current timestamp

Add basic preprocessing:

* strip extra whitespace
* ignore empty files
* handle very long files safely

The script should support CLI usage like:

python -m app.ingestion.ingest --input data/raw --out data/processed/docs.jsonl

Also create a small pytest test:

backend/tests/test_ingest.py

The test should:

* create a temporary folder
* add a few sample `.txt` files
* run the ingestion function
* verify the JSONL output contains valid document entries

Show:

1. the full code for ingest.py
2. the test file
3. example command to run ingestion
4. git commit message
```

**Summary of response:**
Created a script to read `.txt` and `.md` files, extract data, and save it to a JSONL file. Added basic cleaning and testing.

**Files generated:**
- `backend/app/ingestion/ingest.py`
- `backend/tests/test_ingest.py`

### Step 4: BM25 Index
**Prompt:**
```text
Now implement the lexical search index.

Create backend/app/search/bm25.py.

Implement a BM25Index class using the rank-bm25 library.

The class should support:

build(docs)
query(query_text, top_k)

The index should be built using title + text fields.

Save the BM25 index artifacts to:

data/index/bm25/

Also create a pytest test:

backend/tests/test_bm25.py

Use a small 3-document corpus and verify that keyword queries return the correct ranking order.

Show the class implementation, the test file, and a commit message.
```

**Summary of response:**
Implemented a `BM25Index` class that combines document titles and text to build a searchable lexical index using `rank-bm25`. Added pytest coverage verifying index building, querying, saving, and loading.

**Files generated:**
- `backend/app/search/bm25.py`
- `backend/tests/test_bm25.py`

### Step 5: Embedding Pipeline
**Prompt:**
```text
Next implement the semantic embedding pipeline.

Create backend/app/search/embeddings.py.

Use sentence-transformers with a lightweight CPU model.

This module should load documents and generate embeddings for each document.

The output should include:

doc_id
embedding vector

Store embeddings as a numpy array file inside:

data/index/vector/

Add a small helper function to embed a query as well.

Also create a pytest test that verifies embeddings are generated for a small set of sample documents.

Show the implementation and test file.
```

**Summary of response:**
Created the `EmbeddingPipeline` using `sentence-transformers` configured to run strictly on the CPU with the lightweight `all-MiniLM-L6-v2` model. Implemented embedding generation, saving, and loading functionality for documents as numpy arrays, as well as single-query scalar embedding generation. Verified matrix operations and file saving using Pytest.

**Files generated:**
- `backend/app/search/embeddings.py`
- `backend/tests/test_embeddings.py`

### Step 6: Vector Index
**Prompt:**
```text
Now implement the vector search index.

Create backend/app/search/vector_index.py.

Use hnswlib to build a vector similarity index.

The class should support:

build(embeddings)
query(query_vector, top_k)

Persist the vector index to:

data/index/vector/

Add a pytest test that builds a small index and verifies that similar vectors are retrieved correctly.
```

**Summary of response:**
Implemented the `VectorIndex` class using `hnswlib` with `cosine` similarity space. Included `build()`, `query()`, `save()`, and `load()` methods to persist and retrieve the nearest neighbors graph and the identifier mappings. Wrote unit tests to confirm valid clustering of identical vectors and correct IO behavior.

**Files generated:**
- `backend/app/search/vector_index.py`
- `backend/tests/test_vector_index.py`

### Step 7: Hybrid Scoring
**Prompt:**
```text
Now implement hybrid ranking.

Create backend/app/search/hybrid.py.

Combine BM25 results and vector search results.

Hybrid score formula:

hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_vector

Include two normalization strategies:

* min-max
* z-score

Return results containing:

doc_id
bm25_score
vector_score
hybrid_score

Add tests verifying that hybrid ranking combines scores correctly.
```

**Summary of response:**
Created `hybrid_rank` function which mathematically merges results from the lexical component (BM25) and Semantic component (Vector Search) using an alpha weight to penalize and prioritize sources. Implemented and tested both min-max scaling and z-score (standardization) normalizations. Pytests verified proper ranking and arithmetic.

**Files generated:**
- `backend/app/search/hybrid.py`
- `backend/tests/test_hybrid.py`

### Step 8: Search API
**Prompt:**
```text
Now create the search API.

Create backend/app/api/main.py.

Implement a FastAPI application with endpoints:

GET /health

POST /search

Search endpoint input:

query
top_k
alpha

Return ranked results including:

doc_id
bm25_score
vector_score
hybrid_score

Add a FastAPI TestClient test to verify the search endpoint returns valid responses.
```

**Summary of response:**
Created a FastAPI application in `backend/app/api/main.py` containing `/health` and `/search` endpoints. Used Pydantic strictly for typed payloads. The search endpoint dynamically connects the `BM25Index`, `EmbeddingPipeline`, `VectorIndex`, and `hybrid_rank` logic to form the complete end-to-end multi-index hybrid pipeline request. Tested endpoint handling using Starlette's `TestClient` and `unittest.mock`.

**Files generated:**
- `backend/app/api/main.py`
- `backend/tests/test_main.py`

### Step 9: Logging + SQLite
**Prompt:**
```text
Implement structured request logging.

Create backend/app/logging/logger.py.

Every search request should log:

request_id
query
latency_ms
top_k
alpha
result_count

Store logs in SQLite.

Add a metrics endpoint:

GET /metrics

Return basic counters and latency statistics.
```

**Summary of response:**
Implemented `SQLiteLogger` to capture and record runtime metrics asynchronously for every hybrid search query performed via the API. Modded `main.py` to transparently trace request latency, returning results and persisting the telemetry to an auto-created `data/metrics/search_logs.db`. Added a `/metrics` HTTP endpoint to compute and retrieve request volume, average speeds, and yield sizes. Added validation tests mocking API integrations.

**Files generated:**
- `backend/app/logging/logger.py`
- `backend/tests/test_logger.py`
- `backend/app/api/main.py` (Modified)

### Step 9b: Indexing Pipeline Entry Point
**Prompt:**
```text
Now I want to implement the indexing pipeline entry point.

Create a module that reads the processed JSONL dataset and builds the search indexes.

File to create:

backend/app/index.py

This script should:

1. Load documents from data/processed/docs.jsonl
2. Extract title + text for indexing
3. Build the BM25 index
4. Generate embeddings and build the vector index
5. Save index artifacts to:

data/index/bm25/
data/index/vector/

The script should support CLI usage like:

python -m app.index --input data/processed/docs.jsonl

It should:

* verify the input dataset exists
* print basic progress logs
* store metadata about the indexes such as:

  * embedding model name
  * vector dimension
  * corpus hash
  * build timestamp

Also create a pytest test:

backend/tests/test_index_pipeline.py

The test should:

* generate a small sample JSONL dataset
* run the indexing function
* verify that BM25 and vector index artifacts are created.

Show:

1. the full code for backend/app/index.py
2. the test file
3. example command to run indexing
4. git commit message
```

**Summary of response:**
Created the unified indexing script `backend/app/index.py` that processes the parsed JSONL documents into executable BM25 and Vector indices sequentially. Added extensive progress logging, parameter generation, and MD5 corpus hash tracking dumped to `data/index/metadata.json`. Implemented `backend/tests/test_index_pipeline.py` mocking file system actions to securely assert function tracking logic.

**Files generated:**
- `backend/app/index.py`
- `backend/tests/test_index_pipeline.py`

### Step 10: Hybrid Search Service
**Prompt:**
```text
Now implement the hybrid search service that will be used by the API.

Create a module:

backend/app/search/hybrid_search.py

This module should load the BM25 index and the vector index and expose a HybridSearch class.

The class should:

* load index artifacts from data/index/
* accept a query string
* run BM25 retrieval
* generate a query embedding and run vector search
* combine results using hybrid scoring

Use the formula:

hybrid_score = alpha * normalized_bm25 + (1 - alpha) * normalized_vector

Support configurable alpha.

Return results containing:

doc_id
bm25_score
vector_score
hybrid_score

Include helper functions for:

* min-max normalization
* merging result sets

Also add a pytest test:

backend/tests/test_hybrid_search.py

The test should create a small synthetic dataset and verify that hybrid scoring produces a ranked result list.

Show:

* hybrid_search.py
* the test file
* example usage
* git commit message
```

**Summary of response:**
Created `hybrid_search.py` containing the `HybridSearch` service class. This acts as a facade, abstracting the `BM25Index`, `EmbeddingPipeline`, and `VectorIndex` into a cohesive querying environment. Refactored `backend/app/api/main.py` and its tests (`test_main.py`, `test_logger.py`) to utilize this single interface. Implemented unit tests validating the integration points without real files using `unittest.mock`.

**Files generated:**
- `backend/app/search/hybrid_search.py`
- `backend/tests/test_hybrid_search.py`
- `backend/app/api/main.py` (Modified)
- `backend/tests/test_main.py` (Modified)
- `backend/tests/test_logger.py` (Modified)

### Step 10c: FastAPI Backend Service Verification
**Prompt:**
```text
Now create the FastAPI backend service.

Create:

backend/app/api/main.py

This should initialize a FastAPI application and load the HybridSearch service.

Add the following endpoints:

GET /health

Return service status, version, and commit hash.

POST /search

Input JSON:

query
top_k
alpha

The endpoint should call the HybridSearch class and return ranked results including:

doc_id
bm25_score
vector_score
hybrid_score

Add a FastAPI TestClient test:

backend/tests/test_api_search.py

Verify that the /search endpoint returns valid JSON results.

Show:

* main.py
* the test file
* example request using curl
* git commit message
```

**Summary of response:**
Updated `backend/app/api/main.py` so that `/health` resolves the live git commit hash using `subprocess` and returns a `1.0.0` version signature. Implemented `backend/tests/test_api_search.py` leveraging FastAPI's `TestClient` and native `unittest.mock` components. The test thoroughly overrides `HybridSearch` class integration verifying API payloads safely map into endpoint response logic.

**Files generated:**
- `backend/app/api/main.py` (Modified)
- `backend/tests/test_api_search.py`

### Step 11: Query Logging and Persistence
**Prompt:**
```text
Now implement query logging and persistence.

Create:

backend/app/db/query_store.py

Use SQLite to store search requests.

Create a table that records:

request_id
query
latency_ms
top_k
alpha
result_count
timestamp

Add functions:

log_query(...)
get_recent_queries(...)

Modify the /search endpoint so each request is logged.

Add a pytest test verifying that queries are correctly stored in SQLite.

Show:

* query_store.py
* schema
* test file
* commit message
```

**Summary of response:**
Created `QueryStore` class in `backend/app/db/query_store.py` with a dedicated SQLite database (`data/metrics/query_store.db`). The table schema stores `request_id`, `query`, `latency_ms`, `top_k`, `alpha`, `result_count`, and `timestamp`. Integrated `QueryStore` into `main.py` so every `/search` request is automatically persisted. Created `test_query_store.py` validating insert/retrieval ordering and field integrity.

**Files generated:**
- `backend/app/db/query_store.py`
- `backend/tests/test_query_store.py`
- `backend/app/api/main.py` (Modified)

### Step 12: Metrics Endpoint Enhancement
**Prompt:**
```text
Add a metrics endpoint to the FastAPI backend so we can monitor basic search system statistics.

Update the FastAPI application to include:

GET /metrics

The endpoint should return metrics derived from the SQLite query log such as:

total_search_requests
average_latency_ms
p50_latency
p95_latency
zero_result_queries

Metrics can be calculated from the stored query records.

Add a small pytest test that sends requests to the endpoint and verifies the response structure.

Show:

* the updated API code
* the test file
* an example response
* the commit message
```

**Summary of response:**
Added a `get_metrics()` method to `QueryStore` that computes `total_search_requests`, `average_latency_ms`, `p50_latency`, `p95_latency`, and `zero_result_queries` from the SQLite log. Updated the `/metrics` endpoint in `main.py` to source data from `QueryStore` instead of the older `SQLiteLogger`. Created `test_metrics.py` to validate both the mocked endpoint response structure and standalone percentile computation.

**Files generated:**
- `backend/app/db/query_store.py` (Modified)
- `backend/app/api/main.py` (Modified)
- `backend/tests/test_metrics.py`

### Step 13: Evaluation Harness
**Prompt:**
```text
Now implement the evaluation harness for measuring search quality.

Create a module:

backend/app/evaluation/evaluate.py

This script should:

* load evaluation queries from data/eval/queries.jsonl
* load relevance judgments from data/eval/qrels.json
* run the hybrid search system
* compute the following metrics:

nDCG@10
Recall@10
MRR@10

Each run should append a row to:

data/metrics/experiments.csv

Include:

timestamp
alpha value
embedding model name
metric scores

Also add a pytest test that runs the evaluation on a small toy dataset to verify the metrics are computed correctly.

Show:

* evaluate.py
* the test file
* example command to run evaluation
* commit message
```

**Summary of response:**
Created `backend/app/evaluation/evaluate.py` with helper functions for DCG, nDCG@k, Recall@k, and MRR@k. The `run_evaluation` driver accepts a pluggable search function, computes macro-averaged metrics across all queries, and appends results to `data/metrics/experiments.csv` with timestamp, alpha, and model name. CLI entry point loads queries from JSONL and qrels from JSON. Created `test_evaluate.py` verifying individual metric correctness and the end-to-end pipeline with a toy dataset.

**Files generated:**
- `backend/app/evaluation/evaluate.py`
- `backend/tests/test_evaluate.py`

### Step 14: Dashboard UI
**Prompt:**
```text
Create a simple dashboard UI using Streamlit to visualize search activity and evaluation results.

Create:

frontend/dashboard.py

The dashboard should contain four sections:

Search Page
* text input for query
* show search results with bm25_score, vector_score, hybrid_score

KPI Page
* number of search requests
* average latency
* p50 and p95 latency
* most common queries
* zero-result queries

Evaluation Page
* load data/metrics/experiments.csv
* plot nDCG trends across experiments

Debug Page
* display recent query logs from SQLite

The dashboard should communicate with the FastAPI backend.

Show:
* dashboard.py implementation
* example screenshot description
* commit message
```

**Summary of response:**
Created `frontend/dashboard.py` using Streamlit with sidebar navigation across four pages. The Search page sends queries to POST /search and displays ranked results in a DataFrame. The KPI page fetches GET /metrics and renders metric cards (total requests, avg/p50/p95 latency, zero-result queries) plus a bar chart of common queries via GET /recent_queries. The Evaluation page loads experiments.csv and plots nDCG/Recall/MRR trends. The Debug page shows raw query logs and service health. Also added a GET /recent_queries endpoint to main.py.

**Files generated:**
- `frontend/dashboard.py`
- `backend/app/api/main.py` (Modified)

### Step 15: Startup Script (up.sh)
**Prompt:**
```text
Create the startup script that runs the entire system from a clean machine.

Create:

up.sh

The script should:

1. create a Python virtual environment (.venv) if it does not exist
2. activate the environment
3. install dependencies from requirements.txt
4. run the ingestion pipeline if data/processed/docs.jsonl does not exist
5. run the indexing pipeline if index artifacts are missing
6. start the FastAPI backend
7. start the Streamlit dashboard

The script should print the URLs for:

Search API
Dashboard UI

Ensure the script works when executed from the repository root using:

./up.sh

Show:

* the full script
* explanation of each step
* commit message
```

**Summary of response:**
Rewrote `up.sh` as a 7-step automated startup script. Creates `.venv` if missing, installs deps, conditionally runs ingestion and indexing pipelines, then launches FastAPI and Streamlit as background processes. Prints a summary box with URLs and supports Ctrl+C graceful shutdown via `trap`.

**Files generated:**
- `up.sh` (Overwritten)
