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
