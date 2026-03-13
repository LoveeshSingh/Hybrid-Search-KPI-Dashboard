#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════
# up.sh — One-command startup for the Hybrid Search + KPI Dashboard
# Usage:  ./up.sh
# ═══════════════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════╗"
echo "║   Hybrid Search + KPI Dashboard — Startup       ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Step 1: Create virtual environment if missing ─────────────────────────
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo ""
    echo "✅ Virtual environment already exists."
fi

# ── Step 2: Activate the virtual environment ──────────────────────────────
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# ── Step 3: Install dependencies ──────────────────────────────────────────
echo "📥 Installing dependencies from requirements.txt..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# ── Step 4: Run ingestion if processed data is missing ────────────────────
if [ ! -f "data/processed/docs.jsonl" ]; then
    echo ""
    echo "📂 Running data ingestion pipeline..."
    if [ -d "data/raw" ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then
        python -m backend.app.ingestion.ingest --input data/raw --out data/processed/docs.jsonl
    else
        echo "⚠️  No files found in data/raw/. Skipping ingestion."
        echo "   Add .txt or .md files to data/raw/ and re-run."
    fi
else
    echo ""
    echo "✅ Processed dataset already exists at data/processed/docs.jsonl"
fi

# ── Step 5: Run indexing if index artifacts are missing ───────────────────
BM25_EXISTS=false
VECTOR_EXISTS=false

[ -d "data/index/bm25" ] && [ "$(ls -A data/index/bm25 2>/dev/null)" ] && BM25_EXISTS=true
[ -d "data/index/vector" ] && [ "$(ls -A data/index/vector 2>/dev/null)" ] && VECTOR_EXISTS=true

if [ "$BM25_EXISTS" = false ] || [ "$VECTOR_EXISTS" = false ]; then
    if [ -f "data/processed/docs.jsonl" ]; then
        echo ""
        echo "🔨 Building search indices..."
        python -m backend.app.index --input data/processed/docs.jsonl
    else
        echo "⚠️  Cannot build indices: data/processed/docs.jsonl not found."
    fi
else
    echo ""
    echo "✅ Search indices already built."
fi

# ── Step 6: Start FastAPI backend ─────────────────────────────────────────
echo ""
echo "🚀 Starting FastAPI backend on http://127.0.0.1:8000 ..."
uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 3

# ── Step 7: Start Streamlit dashboard ─────────────────────────────────────
echo "📊 Starting Streamlit dashboard on http://127.0.0.1:8501 ..."
streamlit run frontend/dashboard.py --server.port 8501 --server.headless true &
DASHBOARD_PID=$!

# ── Print summary ─────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Services Running                              ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║   Search API:    http://127.0.0.1:8000          ║"
echo "║   API Docs:      http://127.0.0.1:8000/docs     ║"
echo "║   Dashboard UI:  http://127.0.0.1:8501          ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║   Press Ctrl+C to stop all services.            ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Trap Ctrl+C to gracefully shut down ───────────────────────────────────
trap "echo ''; echo '🛑 Shutting down...'; kill $BACKEND_PID $DASHBOARD_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for background processes
wait
