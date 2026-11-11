# Component 5H: Graph Evidence Packaging

This component converts graph retrieval output into an answer-ready graph evidence bundle.

What it does:
- preserves matched nodes and matched edges
- keeps provenance and source chunk references
- provides a compact assembled text form for later orchestration and prompting

Main script:
- `agentic_document_intelligence/scripts/package_graph_evidence.py`

Output:
- `artifacts/experiments/<run_id>/graph_evidence/graph_evidence_bundle.json`
