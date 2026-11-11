# Component 7B: Multi-Source Orchestration Execution

This component executes the routed source plan and packages evidence from the selected retrieval systems.

What it does:
- consumes the sub-query routing plan
- selectively runs document RAG, GraphRAG, and SQL retrieval only where needed
- packages each source output into answer-ready evidence bundles
- keeps per-sub-query provenance so later answer generation can cite the right source

Why:
- routing alone is not enough; the system needs a real execution layer that fans out to the right retrievers
- this is the bridge from source planning to later evidence fusion and answer synthesis

Main script:
- `agentic_document_intelligence/scripts/execute_multi_source_orchestration.py`

Evaluation:
- `agentic_document_intelligence/scripts/evaluate_multi_source_orchestration.py`

Outputs:
- `artifacts/experiments/<run_id>/orchestration/multi_source_orchestration_report.json`
- `artifacts/experiments/<run_id>/orchestration/multi_source_orchestration_eval_report.json`
