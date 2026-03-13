import argparse
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from backend.app.search.bm25 import BM25Index
from backend.app.search.embeddings import EmbeddingPipeline
from backend.app.search.vector_index import VectorIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_hash(docs: list) -> str:
    """Compute an MD5 hash of the corpus for metadata tracking."""
    hasher = hashlib.md5()
    for doc in docs:
        content = f"{doc.get('doc_id','')}{doc.get('title','')}{doc.get('text','')}"
        hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()

def build_indices(input_file: str):
    input_path = Path(input_file)
    if not input_path.exists() or not input_path.is_file():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"Loading documents from {input_file}...")
    docs = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load JSONL dataset: {e}")
        return

    num_docs = len(docs)
    logger.info(f"Loaded {num_docs} documents.")
    if num_docs == 0:
        logger.warning("No documents to index.")
        return

    # 1. Build lexical BM25 index
    logger.info("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(docs)

    # 2. Extract texts for embeddings and build semantic index
    logger.info("Initializing Embedding Pipeline...")
    # hardcoded model choice
    model_name = "all-MiniLM-L6-v2"
    embedder = EmbeddingPipeline(model_name=model_name)
    
    logger.info("Generating embeddings...")
    embeddings, doc_ids = embedder.embed_documents(docs, save=True)
    
    logger.info("Building Vector Index...")
    v_index = VectorIndex()
    v_index.build(embeddings, doc_ids=doc_ids)
    
    # 3. Save metadata
    metadata = {
        "num_documents": num_docs,
        "embedding_model": model_name,
        "vector_dimension": embeddings.shape[1] if num_docs > 0 else 0,
        "corpus_hash": compute_hash(docs),
        "build_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    metadata_path = Path("data/index/metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Indexing complete. Metadata saved to {metadata_path}")
    logger.info(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BM25 and Vector Search indices from a JSONL document dataset.")
    parser.add_argument("--input", required=True, help="Path to input JSONL dataset (e.g. data/processed/docs.jsonl)")
    
    args = parser.parse_args()
    
    build_indices(args.input)
