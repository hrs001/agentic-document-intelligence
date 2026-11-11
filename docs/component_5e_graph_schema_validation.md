# Component 5E: Graph Schema / Ontology Validation

This component validates the normalized graph against a practical schema before graph storage.

What it does:
- keeps high-value node types by default
- applies stricter rules to lower-value abstract node types
- filters weak edges whose endpoints are invalid or whose support is too weak
- records explicit rejection reasons for auditability

Why:
- the normalized graph is cleaner than raw extraction, but still too noisy for direct graph loading
- Kuzu should be loaded only with graph-usable nodes and edges

Main script:
- `agentic_document_intelligence/scripts/validate_graph_schema.py`

Output:
- `artifacts/experiments/<run_id>/graph_validated/<document_id>_graph_validated.json`

Important note:
- this step is deterministic and cost-free at runtime
- it should be treated as the final filter before Kuzu graph storage build
