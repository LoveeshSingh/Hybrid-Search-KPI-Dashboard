# Architecure

## Overview
CPU-only Hybrid Search system.

## Components
1. **FastAPI Backend**: Handles search and ingestion.
2. **Search Engine**: BM25 (rank-bm25) + Vector (FAISS/hnswlib CPU) via sentence-transformers.
3. **Database**: SQLite for logs and metrics.
4. **Dashboard**: React/Streamlit for KPI visualization.
