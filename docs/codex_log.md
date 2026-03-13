Note:
This log records prompts used with a coding assistant during development.
Although the assignment refers to "Codex", the assistant used was Antigravity.
The same granular prompting protocol was followed.

## Prompts

### Step 1: Create Repo Structure
**Prompt Intent:** Initialize repository structure and required documentation files.
**Context:** First step of the project setup.
**Output Expectations:** Directory structure created, placeholder markdown logs (`codex_log.md`, `decision_log.md`, `break_fix_log.md`), `README.md`, and `up.sh` created.

### Step 2: Setup Python Environment
**Prompt Intent:** Prepare Python package structure and define dependencies.
**Context:** Second step to allow writing backend code in subsequent tasks.
**Output Expectations:** `__init__.py` files created in backend modular directories, and `requirements.txt` populated with basic data science, search, and FastAPI dependencies.

### Step 3: Data Ingestion Pipeline
**Prompt Intent:** Implement the document ingestion pipeline.
**Context:** Third step to process raw documents into a normalized JSONL dataset.
**Output Expectations:** `backend/app/ingestion/ingest.py` handling file reading and JSONL creation. Small pytest test in `backend/tests/test_ingest.py` to verify functionality.
