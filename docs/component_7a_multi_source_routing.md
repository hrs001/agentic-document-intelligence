# Component 7A: Multi-Source Routing

This component decides which retrieval source or sources should handle each sub-query.

What it does:
- reads the transformed sub-queries
- considers SQL capabilities, GraphRAG capabilities, and document RAG capabilities
- uses an LLM to choose among `vector_document`, `graph_relationships`, and `sql_structured`
- validates the returned route set with lightweight signal-aware post checks

Why:
- real user questions vary too much for pure hard-coded routing rules
- routing needs to support mixed questions that may require multiple sources
- later orchestration should execute only the sources most likely to produce useful evidence

Main script:
- `agentic_document_intelligence/scripts/multi_source_routing.py`

Evaluation:
- `agentic_document_intelligence/scripts/evaluate_multi_source_routing.py`

Outputs:
- `artifacts/experiments/<run_id>/routing/multi_source_routing_plan.json`
- `artifacts/experiments/<run_id>/routing/multi_source_routing_eval_report.json`
