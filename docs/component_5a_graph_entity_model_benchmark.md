# Component 5A: Graph Entity Extraction Model Benchmark

This benchmark exists to choose a practical LLM for GraphRAG entity and relationship extraction before the full graph pipeline is built.

What it does:
- uses real text from the first substantive pages of `Microsoft_FY2025_10K_Summary.pdf`
- runs the same extraction schema across multiple candidate models
- scores outputs against a deterministic gold set
- estimates token-based cost from official OpenAI pricing inputs configured in the benchmark script

What is evaluated:
- expected entity recall
- expected relationship recall
- precision against the benchmark gold set
- grounding quality via exact evidence snippets copied from the source text
- estimated token cost

Main script:
- `agentic_document_intelligence/scripts/benchmark_graph_entity_extraction.py`

Eval set:
- `agentic_document_intelligence/evals/graph_extraction_benchmark_cases.json`

Important note:
- this benchmark is meant to choose a sensible starting model for GraphRAG extraction
- it is not the full graph extraction pipeline yet
