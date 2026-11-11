# Component 5G: Graph Retrieval

This component retrieves graph evidence from the saved local Kuzu database.

What it does:
- loads the reusable local Kuzu graph database
- scores candidate nodes lexically against the user query
- selects top graph nodes
- expands a one-hop neighborhood in Kuzu
- scores and ranks nearby edges
- returns graph evidence for later orchestration

Why:
- this uses the saved Kuzu graph instead of rerunning extraction
- it is lightweight enough for local development and Lambda-style deployment on a small graph

Main scripts:
- `agentic_document_intelligence/scripts/graph_retrieval.py`
- `agentic_document_intelligence/scripts/evaluate_graph_retrieval.py`

Outputs:
- retrieval report at `artifacts/experiments/<run_id>/graph_retrieval/graph_retrieval_report.json`
- evaluation report at `artifacts/experiments/<run_id>/graph_retrieval/graph_retrieval_eval_report.json`
