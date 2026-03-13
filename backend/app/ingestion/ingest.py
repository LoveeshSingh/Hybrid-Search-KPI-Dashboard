import argparse
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_file(file_path: Path) -> dict | None:
    """Read a text or markdown file, extract its title and content, and format as a document dictionary."""
    if not file_path.is_file() or file_path.suffix not in ['.txt', '.md']:
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

    # Filter out empty lines for accurate title extraction and content stripping
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if not non_empty_lines:
        logger.warning(f"File {file_path} is empty or contains only whitespace. Skipping.")
        return None

    title = non_empty_lines[0]
    # Re-join all lines after the first non-empty one for the text body, maintaining some original paragraph structure
    text = '\n'.join([line.strip() for line in lines[len(lines) - len(non_empty_lines) + 1:]]).strip()
    
    # Fallback if the whole document was just a single line
    if not text:
        text = title

    return {
        "doc_id": file_path.stem,
        "title": title,
        "text": text,
        "source": str(file_path),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

def ingest_directory(input_dir: str, output_file: str) -> None:
    """Process all .txt and .md files in input_dir and write them as JSONL to output_file."""
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    
    logger.info(f"Starting ingestion from {input_dir} to {output_file}")
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.txt', '.md']:
                doc = process_file(file_path)
                if doc:
                    out_f.write(json.dumps(doc) + '\n')
                    processed_count += 1
                    
    logger.info(f"Successfully processed {processed_count} files into {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw text/markdown documents into JSONL format.")
    parser.add_argument("--input", required=True, help="Path to input directory containing .txt or .md files")
    parser.add_argument("--out", required=True, help="Path to output JSONL file")
    
    args = parser.parse_args()
    
    ingest_directory(args.input, args.out)
