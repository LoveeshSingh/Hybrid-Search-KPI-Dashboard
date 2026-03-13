# Break / Fix Log

Documents intentional failures introduced during development and the steps taken to diagnose and fix them.

## Scenario 1: Vector Index Metadata Mismatch

Problem:
Changing the embedding model used for queries can produce vectors with a different dimension than the stored vector index.

Failure Simulation:
The embedding model was modified so the generated query embedding dimension no longer matched the stored index metadata.

Observed Failure:
Vector search failed during query execution because the vector dimensions were incompatible.

Fix Implemented:
Added metadata validation when loading the vector index. The system now checks:

embedding_model
embedding_dimension
corpus_hash
build_timestamp

If the metadata does not match the current configuration, the system returns a clear error instructing the user to rebuild the index.

Result:
The system now prevents invalid vector searches caused by incompatible embeddings.


## Scenario 2: Database Schema Migration Failure

Problem:
The query logging database schema changed during development.

Failure Simulation:
A new column `user_agent` was added to the query logging table. Restarting the application without migrating the database caused insert operations to fail.

Observed Failure:
SQLite raised an error because the existing table schema did not contain the required column.

Fix Implemented:
Added a schema version system with automatic migration.

Migration:
v1 → v2  
Added column `user_agent`.

The application now checks the schema version on startup and automatically upgrades the database if required.

Result:
The system continues operating even after schema changes.


## Scenario 3: Hybrid Score Normalization Bug

Problem:
Min-max normalization caused a divide-by-zero error when all scores were identical.

Failure Simulation:
Test cases were created where all BM25 scores had the same value.

Observed Failure:
Normalization produced NaN values due to division by zero.

Fix Implemented:
The normalization function now detects equal score ranges and returns `0.0` for all normalized values instead of dividing by zero.

Result:
Hybrid scoring now works correctly even when score ranges are identical.