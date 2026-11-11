# Component 7C: Latency-Optimized Orchestration Policy

This component adds an execution-policy layer on top of routing so the system does not always run the most expensive retrieval path.

What it does:
- plans per-sub-query execution depth after routing
- uses an LLM to select vector execution profiles such as `fast`, `balanced`, or `full`
- keeps SQL and graph execution direct where possible
- allows runtime escalation when the first-pass vector retrieval is weak

Why:
- the full orchestration path is robust but too slow to run unchanged for every query
- production systems need bounded cost and latency while still preserving quality on difficult questions

Main scripts:
- `agentic_document_intelligence/scripts/latency_optimized_orchestration_policy.py`
- `agentic_document_intelligence/scripts/execute_latency_optimized_orchestration.py`
- `agentic_document_intelligence/scripts/evaluate_latency_optimized_orchestration.py`

Outputs:
- `artifacts/experiments/<run_id>/orchestration/latency_optimized_policy_plan.json`
- `artifacts/experiments/<run_id>/orchestration/latency_optimized_orchestration_report.json`
- `artifacts/experiments/<run_id>/orchestration/latency_optimized_orchestration_eval_report.json`
