"""
Hybrid Search KPI Dashboard
Run with:  streamlit run frontend/dashboard.py
Requires the FastAPI backend running at http://127.0.0.1:8000
"""

import streamlit as st
import requests
import pandas as pd
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Hybrid Search Dashboard", layout="wide")

# ── Sidebar navigation ────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["🔍 Search", "📊 KPI", "📈 Evaluation", "🐛 Debug"])


# ═══════════════════════════════════════════════════════════════════════════
# 1. SEARCH PAGE
# ═══════════════════════════════════════════════════════════════════════════
if page == "🔍 Search":
    st.title("🔍 Hybrid Search")
    st.markdown("Enter a query to search the indexed document corpus.")

    query = st.text_input("Query", placeholder="e.g. machine learning optimization")
    col1, col2 = st.columns(2)
    top_k = col1.slider("Top K", min_value=1, max_value=50, value=5)
    alpha = col2.slider("Alpha (BM25 weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    if st.button("Search") and query:
        try:
            resp = requests.post(
                f"{API_BASE}/search",
                json={"query": query, "top_k": top_k, "alpha": alpha},
                timeout=30,
            )
            if resp.status_code == 200:
                results = resp.json()
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(
                        df.style.format({
                            "bm25_score": "{:.4f}",
                            "vector_score": "{:.4f}",
                            "hybrid_score": "{:.4f}",
                        }),
                        use_container_width=True,
                    )
                else:
                    st.warning("No results found.")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.ConnectionError:
            st.error("Cannot reach the backend API. Is it running?")


# ═══════════════════════════════════════════════════════════════════════════
# 2. KPI PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 KPI":
    st.title("📊 Key Performance Indicators")

    try:
        resp = requests.get(f"{API_BASE}/metrics", timeout=10)
        if resp.status_code == 200:
            m = resp.json()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Requests", m.get("total_search_requests", 0))
            c2.metric("Avg Latency (ms)", f"{m.get('average_latency_ms', 0):.1f}")
            c3.metric("p50 Latency (ms)", f"{m.get('p50_latency', 0):.1f}")
            c4.metric("p95 Latency (ms)", f"{m.get('p95_latency', 0):.1f}")

            st.divider()
            st.subheader("Zero-Result Queries")
            st.metric("Count", m.get("zero_result_queries", 0))
        else:
            st.error(f"Metrics endpoint returned {resp.status_code}")
    except requests.ConnectionError:
        st.error("Cannot reach the backend API. Is it running?")

    # Most common queries from QueryStore via recent queries
    st.divider()
    st.subheader("Most Common Queries (recent)")
    try:
        # We pull recent queries from a dedicated endpoint if available,
        # otherwise fall back to showing a placeholder.
        resp = requests.get(f"{API_BASE}/recent_queries", timeout=10)
        if resp.status_code == 200:
            recent = resp.json()
            if recent:
                query_counts = pd.DataFrame(recent)["query"].value_counts().head(10)
                st.bar_chart(query_counts)
            else:
                st.info("No queries logged yet.")
        else:
            st.info("Recent-queries endpoint not available yet.")
    except requests.ConnectionError:
        st.info("Backend not reachable.")


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVALUATION PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation":
    st.title("📈 Evaluation Results")

    csv_path = Path("data/metrics/experiments.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("nDCG@10 Trend")
        if "ndcg@10" in df.columns:
            st.line_chart(df["ndcg@10"])

        st.subheader("Recall@10 Trend")
        if "recall@10" in df.columns:
            st.line_chart(df["recall@10"])

        st.subheader("MRR@10 Trend")
        if "mrr@10" in df.columns:
            st.line_chart(df["mrr@10"])
    else:
        st.info("No experiments.csv found. Run the evaluation harness first.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. DEBUG PAGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🐛 Debug":
    st.title("🐛 Debug — Recent Query Logs")

    try:
        resp = requests.get(f"{API_BASE}/recent_queries", timeout=10)
        if resp.status_code == 200:
            logs = resp.json()
            if logs:
                df = pd.DataFrame(logs)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No query logs recorded yet.")
        else:
            st.warning(f"Endpoint returned {resp.status_code}. "
                       "You may need to add a /recent_queries route to the API.")
    except requests.ConnectionError:
        st.error("Cannot reach the backend API. Is it running?")

    st.divider()
    st.subheader("Service Health")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            st.json(resp.json())
        else:
            st.error(f"Health check failed: {resp.status_code}")
    except requests.ConnectionError:
        st.error("Backend not reachable.")
