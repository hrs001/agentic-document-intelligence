# Project Examples Scan

Scan date: 2026-03-16

Reference scanned:

- `project_examples/universal-knowledge-copilot`

## 1. What the example app is trying to achieve

The reference project is an "agentic" multi-source RAG demo app. Its goal is to let a user upload documents and tabular data, ingest them into several retrieval systems, and then answer questions using a coordinated reasoning pipeline.

The app combines:

- document RAG over chunked text,
- GraphRAG over extracted entities and relationships,
- text-to-SQL over CSV/XLSX data,
- optional web fallback,
- UI telemetry so the user can see the internal pipeline.

The bundled demo corpus is already SEC-oriented:

- `Micreosoft 10-K 2025.pdf`
- `microsoft_financials_data.csv`

## 2. Main product surfaces found in the example

### Frontend surfaces

- Chat panel
  - asks questions,
  - polls async backend jobs,
  - renders markdown answers,
  - shows citation hover cards,
  - shows deferred evaluation.
- Data Vault
  - upload widget,
  - indicates whether a session has data.
- Graph visualizer
  - loads nodes/edges from backend,
  - lays out graph around central hubs,
  - supports search and reveal interactions.
- Brain Monitor
  - shows step-by-step telemetry emitted by the backend.

### Backend surfaces

- Session status API
- Async chat jobs API
- Async demo hydrate API
- Async upload jobs API
- Graph API
- Security / preprocessing API
- Lambda-readiness and latency diagnostics endpoints

## 3. Core backend components discovered

### `backend/app/main.py`

Acts as the FastAPI entrypoint and orchestrator. Important behaviors:

- uses async job polling instead of long-lived streaming as the active path,
- keeps session-scoped state,
- has a demo hydration path that loads the bundled SEC demo files,
- exposes session status and graph endpoints used by the UI.

### `backend/app/services/ingestion_service.py`

This is the expensive ingestion engine. It performs:

- PDF and DOCX extraction,
- denoising,
- PII scrubbing,
- optional image parsing with a vision model,
- parent/child chunking,
- document chunk indexing,
- GraphRAG extraction on selected parent chunks.

Key chunking design:

- structure-aware split by markdown headers,
- parent chunks around 1500,
- child chunks around 400,
- child chunks are retrieved,
- parent chunks are used as richer answer context.

### `backend/app/services/hybrid_search.py`

Document retrieval stack:

- dense embeddings,
- sparse BM25 encoding,
- Pinecone hybrid search,
- Pinecone reranking.

This is the main document RAG path.

### `backend/app/services/master_query_engine.py`

Document retrieval orchestrator. It:

- expands the user query,
- runs multiple searches,
- fuses results with RRF,
- reranks the final candidates.

This is meant to improve retrieval quality on complex questions.

### `backend/app/services/semantic_router.py`

Intent router that decides whether a subquery should go to:

- `structured_data`,
- `relationships`,
- `general_knowledge`,
- `web_search`.

It uses embedding similarity against anchor descriptions.

### `backend/app/services/text_to_sql_agent.py`

Structured-data reasoning path:

- reads the SQLite schema,
- asks an LLM to generate a safe `SELECT`,
- executes the query,
- returns formatted table results for answer synthesis.

### `backend/app/services/sqlite_s3_engine.py`

Structured data storage:

- ingests CSV/XLSX into session-scoped SQLite,
- optionally syncs to S3,
- exposes schema and query execution.

### `backend/app/services/graphrag_extractor.py`

Graph construction path:

- uses an LLM to extract entities and relationships from selected text,
- writes those nodes/edges into Kuzu.

### `backend/app/core/database.py`

Database manager for:

- Pinecone index,
- Kuzu graph DB per session,
- optional S3 sync for graph files.

### `backend/app/services/agentic_brain.py`

This is the main reasoning engine. It uses LangGraph and coordinates:

- query decomposition,
- routing,
- retrieval,
- relevance grading,
- query rewrite,
- bounded fallback,
- compression,
- final answer generation,
- optional refinement.

Important observation:

- this is the main "brain" of the app,
- it is also the biggest source of complexity and runtime cost.

## 4. Data and storage model used by the example

The reference app is session-scoped almost everywhere.

Per session it may create or use:

- Pinecone namespace for document chunks,
- Kuzu graph database files,
- SQLite database,
- JSON session state,
- async job state,
- telemetry/job event logs.

This design supports uploads, but it is exactly why cost and operational complexity increase when every user can bring their own corpus.

## 5. UI components worth reusing conceptually

These look useful to recreate in the new demo:

- `frontend/src/components/ChatInterface.tsx`
  - core chat UX,
  - citation cards,
  - telemetry integration,
  - async job polling pattern.
- `frontend/src/components/GraphVisualizer.tsx`
  - graph exploration experience.
- `frontend/src/components/BrainMonitor.tsx`
  - excellent demo feature for showing pipeline steps.
- `frontend/src/components/DataVault.tsx`
  - concept is useful, but for the new app it should probably become a read-only "Demo Corpus" panel instead of upload UI.

## 6. What is expensive in the old approach

The costly parts are mostly on ingestion and per-user corpus setup:

- PDF/DOCX parsing,
- vision parsing of document images,
- chunk embedding generation,
- Pinecone upserts,
- GraphRAG entity and relationship extraction,
- Kuzu graph creation,
- CSV/XLSX ingestion and schema creation for each uploaded dataset,
- session-scoped persistence for many users.

This cost happens because new user uploads create new retrieval assets instead of reusing a fixed prepared corpus.

## 7. Why the SEC-only precomputed demo is a good idea

Yes, this is a good and normal approach for a demo app.

For a demo, it is better to:

- lock the corpus to your chosen SEC 10-K documents and related datasets,
- run ingestion, embeddings, graph extraction, and database prep once offline,
- ship only the retrieval and answer-time experience to end users,
- keep the same UI feel while removing the most expensive per-user setup.

Benefits:

- much cheaper hosting,
- much faster first-use experience,
- easier QA because everyone queries the same corpus,
- easier caching,
- easier graph and SQL debugging,
- better demo reliability.

## 8. Recommended rebuild direction

For the new app, keep these components:

- chat UI,
- citation UI,
- telemetry / brain monitor,
- graph explorer,
- document RAG,
- GraphRAG retrieval,
- structured-data querying if the SEC dataset includes useful tables.

For the new app, remove or change these components:

- remove user uploads,
- remove per-user ingestion jobs,
- replace "Data Vault" with a read-only "Demo Corpus" panel,
- replace session-scoped data creation with prebuilt shared demo indices/data,
- keep session IDs only for chat history, telemetry, and lightweight state.

## 9. Suggested lower-cost architecture

### Offline precompute step

Run once before deployment:

- parse SEC document set,
- create parent/child chunks,
- generate embeddings,
- build vector index,
- extract entities and relationships,
- build graph store,
- ingest selected financial datasets into SQLite,
- generate corpus metadata used by the UI.

### Online runtime step

Run per user query:

- load shared demo corpus metadata,
- retrieve from prebuilt vector index,
- retrieve graph evidence from prebuilt graph store,
- run SQL query only when needed,
- synthesize answer with citations,
- emit telemetry to UI.

This preserves the "AI components" feel while moving the expensive work out of the live user path.

## 10. Minimal rebuild plan

Recommended first version:

1. Build a new backend with only precomputed-corpus query endpoints.
2. Build a new frontend that recreates:
   - chat,
   - corpus panel,
   - graph panel,
   - telemetry panel.
3. Use the bundled SEC files as the fixed demo corpus.
4. Delay advanced features until the basic flow works:
   - no user upload,
   - no per-user indexing,
   - no web fallback initially.

## 11. Components to reproduce first

Highest-value first:

- session bootstrap
- session status
- chat jobs + polling
- citation normalization
- graph endpoint
- chat panel
- graph panel
- telemetry panel
- read-only corpus panel

Second wave:

- retrieval orchestration
- SQL route
- graph route
- evaluator card

Later only if needed:

- FLARE
- web fallback
- advanced rewrite loops
- Lambda-specific hardening

## 12. Important rebuild caution

The reference app is ambitious and feature-rich, but it may be too heavy to copy 1:1 for the new demo.

For the new version, the better goal is:

- preserve the best visible demo components,
- preserve the multi-source RAG idea,
- simplify the runtime,
- precompute everything expensive.

That should produce a much more practical and hostable SEC demo.
