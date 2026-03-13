import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from backend.app.ingestion.ingest import ingest_directory

def test_ingest_directory():
    with TemporaryDirectory() as temp_input_dir:
        input_path = Path(temp_input_dir)
        
        # Create valid doc 1
        doc1_path = input_path / "doc1.txt"
        doc1_path.write_text("Hello World\nThis is the body of the first document.\n\nIt has multiple lines.\n", encoding="utf-8")
        
        # Create valid doc 2
        doc2_path = input_path / "doc2.md"
        doc2_path.write_text("# Markdown Title\nSome markdown content.\n", encoding="utf-8")
        
        # Create an empty file (should be ignored)
        empty_doc_path = input_path / "empty.txt"
        empty_doc_path.write_text("   \n\n  \n", encoding="utf-8")
        
        # Output file path
        with TemporaryDirectory() as temp_out_dir:
            out_file = Path(temp_out_dir) / "docs.jsonl"
            
            # Run ingestion
            ingest_directory(str(input_path), str(out_file))
            
            # Verify results
            assert out_file.exists()
            
            with open(out_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            assert len(lines) == 2  # The empty file should be skipped
            
            items = [json.loads(line) for line in lines]
            
            # Sort by doc_id to assert safely
            items.sort(key=lambda x: x['doc_id'])
            
            assert items[0]['doc_id'] == "doc1"
            assert items[0]['title'] == "Hello World"
            assert items[0]['text'] == "This is the body of the first document.\n\nIt has multiple lines."
            assert items[0]['source'] == str(doc1_path)
            assert "created_at" in items[0]
            
            assert items[1]['doc_id'] == "doc2"
            assert items[1]['title'] == "# Markdown Title"
            assert items[1]['text'] == "Some markdown content."
            assert items[1]['source'] == str(doc2_path)
            assert "created_at" in items[1]
