# Component 7D: Parallel-Safe Orchestration Evaluation

This component evaluates the safe parallel execution path for independent SQL and GraphRAG calls.

What it does:
- uses atomic mixed-source subqueries that should not decompose further
- verifies the policy marks them as `parallel_safe`
- executes the latency-optimized orchestrator
- checks that both SQL and graph evidence are returned correctly

Why:
- the main end-to-end benchmark mostly contains dependent multi-step queries
- those are correctly kept sequential, so they do not prove whether safe parallelism works
- this benchmark isolates the scenario where SQL and graph can run concurrently

Main script:
- `agentic_document_intelligence/scripts/evaluate_parallel_safe_orchestration.py`

Output:
- `artifacts/experiments/<run_id>/orchestration/parallel_safe_orchestration_eval_report.json`
