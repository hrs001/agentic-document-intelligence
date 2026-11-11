# Component 5D: Graph Normalization and Dedup

This component cleans the raw graph extraction output into a reusable normalized graph artifact.

What it does:
- merges obvious aliases and duplicate node names
- merges duplicate edges after node canonicalization
- preserves provenance from the raw extraction:
  - graph input ids
  - parent ids
  - child ids
  - section titles
  - page ranges
- keeps alias lists and evidence snippets for later auditability

Why:
- the raw extraction output is intentionally high recall and somewhat noisy
- later schema validation and Kuzu loading should start from a cleaner graph artifact

Main script:
- `agentic_document_intelligence/scripts/normalize_graph_extraction.py`

Output:
- `artifacts/experiments/<run_id>/graph_normalized/<document_id>_graph_normalized.json`

Important note:
- this step is deterministic and does not call an LLM
- it is meant to reduce duplicate noise cheaply before later validation and graph storage
