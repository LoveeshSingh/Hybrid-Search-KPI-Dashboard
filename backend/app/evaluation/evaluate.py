"""
Evaluation harness for measuring hybrid search quality.

Metrics computed:
  - nDCG@k   (Normalized Discounted Cumulative Gain)
  - Recall@k (fraction of relevant docs retrieved)
  - MRR@k    (Mean Reciprocal Rank)

Usage:
    python -m backend.app.evaluation.evaluate \
        --queries data/eval/queries.jsonl \
        --qrels   data/eval/qrels.json \
        --alpha   0.5 \
        --top_k   10
"""

import argparse
import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

EXPERIMENTS_CSV = "data/metrics/experiments.csv"

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def dcg(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)  # i+2 because positions are 1-indexed
    return score


def ndcg_at_k(retrieved_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    """Compute nDCG@k for a single query."""
    relevances = [qrel.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg(relevances, k)

    ideal_relevances = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg_val = dcg(ideal_relevances, k)

    if ideal_dcg_val == 0:
        return 0.0
    return actual_dcg / ideal_dcg_val


def recall_at_k(retrieved_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    """Compute Recall@k for a single query."""
    relevant_docs = {doc_id for doc_id, rel in qrel.items() if rel > 0}
    if not relevant_docs:
        return 0.0
    retrieved_relevant = relevant_docs.intersection(set(retrieved_ids[:k]))
    return len(retrieved_relevant) / len(relevant_docs)


def mrr_at_k(retrieved_ids: List[str], qrel: Dict[str, int], k: int = 10) -> float:
    """Compute Reciprocal Rank at k for a single query."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if qrel.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    queries: List[Dict[str, Any]],
    qrels: Dict[str, Dict[str, int]],
    search_fn,
    alpha: float = 0.5,
    top_k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Run the evaluation harness.

    Args:
        queries:    list of {"query_id": str, "text": str}
        qrels:      mapping query_id → {doc_id: relevance_int}
        search_fn:  callable(query_text, top_k, alpha) → list of result dicts
        alpha:      hybrid weight
        top_k:      evaluation depth
        model_name: embedding model identifier for logging
    """
    ndcg_scores, recall_scores, mrr_scores = [], [], []

    for q in queries:
        qid = q["query_id"]
        text = q["text"]

        if qid not in qrels:
            logger.warning(f"No relevance judgments for query '{qid}', skipping.")
            continue

        results = search_fn(text, top_k, alpha)
        retrieved_ids = [r["doc_id"] for r in results]

        qrel = qrels[qid]
        ndcg_scores.append(ndcg_at_k(retrieved_ids, qrel, k=top_k))
        recall_scores.append(recall_at_k(retrieved_ids, qrel, k=top_k))
        mrr_scores.append(mrr_at_k(retrieved_ids, qrel, k=top_k))

    n = len(ndcg_scores) or 1
    metrics = {
        "ndcg@10": round(sum(ndcg_scores) / n, 4),
        "recall@10": round(sum(recall_scores) / n, 4),
        "mrr@10": round(sum(mrr_scores) / n, 4),
        "num_queries": len(ndcg_scores),
    }

    logger.info(f"Evaluation results: {json.dumps(metrics, indent=2)}")

    # Persist to experiments CSV
    _append_experiment(metrics, alpha, model_name)

    return metrics


def _append_experiment(metrics: Dict[str, Any], alpha: float, model_name: str):
    """Append a single experiment row to the experiments CSV."""
    csv_path = Path(EXPERIMENTS_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    fieldnames = [
        "timestamp", "alpha", "embedding_model",
        "ndcg@10", "recall@10", "mrr@10", "num_queries",
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alpha": alpha,
            "embedding_model": model_name,
            **metrics,
        })

    logger.info(f"Experiment logged to {csv_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid search quality.")
    parser.add_argument("--queries", required=True, help="Path to queries JSONL")
    parser.add_argument("--qrels", required=True, help="Path to qrels JSON")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # Load queries
    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    # Load qrels
    with open(args.qrels, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    # Import and wire the search service
    from backend.app.search.hybrid_search import HybridSearch
    service = HybridSearch()
    service.load()

    def search_fn(query_text, top_k, alpha):
        return service.search(query=query_text, top_k=top_k, alpha=alpha)

    run_evaluation(queries, qrels, search_fn, alpha=args.alpha, top_k=args.top_k)


if __name__ == "__main__":
    main()
