# Architecture Overview

This document describes the architecture of the Hybrid Search + KPI Dashboard system.

The system provides a searchable document corpus using a hybrid retrieval pipeline that combines lexical and semantic search.

---

# System Components

The system consists of five main components:

1. Data Ingestion
2. Indexing Pipeline
3. Hybrid Search Service
4. Backend API
5. Dashboard UI

---

# Repository Structure

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

---

# Data Flow

The system processes documents through the following pipeline:

1. Raw documents (.txt / .md) are placed in `data/raw/`.
2. The ingestion pipeline converts them into a normalized dataset (`docs.jsonl`).
3. The indexing pipeline builds:
   - a BM25 lexical index
   - a vector similarity index
4. The API receives search queries.
5. The hybrid search service retrieves results from both indexes.
6. Results are combined using hybrid scoring.
7. Queries and metrics are logged in SQLite.
8. The dashboard visualizes search results and system metrics.

---

# Data Ingestion

Module:
backend/app/ingestion/ingest.py

Responsibilities:

- load raw documents
- extract metadata
- clean whitespace
- generate structured JSONL dataset

Output format:

doc_id  
title  
text  
source  
created_at

Output file:

data/processed/docs.jsonl

---

# Indexing Pipeline

Module:
backend/app/index.py

The indexing pipeline builds two indexes:

## BM25 Index

Library:
rank-bm25

Stored in:

data/index/bm25/

Used for keyword matching.

---

## Vector Index

Libraries:

sentence-transformers  
hnswlib

Embedding model:

all-MiniLM-L6-v2

Stored in:

data/index/vector/

Used for semantic similarity search.

---

# Hybrid Search

Module:
backend/app/search/hybrid_search.py

The system retrieves results from both BM25 and vector indexes and combines them using hybrid scoring.

Formula:

hybrid_score =
alpha * normalized_bm25 +
(1 - alpha) * normalized_vector

Normalization methods:

- Min-max normalization
- Z-score normalization

The alpha parameter controls the balance between lexical and semantic relevance.

---

# Backend API

Framework:
FastAPI

Endpoints:

GET /health  
Returns system status and version.

POST /search  
Executes hybrid search.

GET /metrics  
Returns request and latency statistics.

---

# Query Logging

Module:
backend/app/db/query_store.py

Search requests are stored in SQLite for analytics.

Fields stored:

request_id  
query  
latency_ms  
top_k  
alpha  
result_count  
timestamp

Database file:

data/metrics/query_store.db

---

# Evaluation Framework

Module:
backend/app/evaluation/evaluate.py

The evaluation harness computes retrieval metrics using labeled query data.

Metrics:

nDCG@10  
Recall@10  
MRR@10

Results are stored in:

data/metrics/experiments.csv

---

# Dashboard

Framework:
Streamlit

Location:

frontend/dashboard.py

Pages:

Search
KPI metrics
Evaluation results
Debug logs

The dashboard communicates with the backend API.

---

# Startup Process

The system is started using:

./up.sh

The script performs:

1. environment setup
2. dependency installation
3. ingestion if dataset missing
4. index building if indexes missing
5. backend startup
6. dashboard startup

---

# Key Design Goals

CPU-only execution  
Local reproducibility  
Modular architecture  
Transparent evaluation  
Simple deployment