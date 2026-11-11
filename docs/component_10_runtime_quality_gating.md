# Component 10: Runtime Quality Gating and Escalation Policy

This component is the bounded runtime controller for final-answer quality.

What it does:
- runs the final answer path for a live query
- reads the self-reflective critique result
- reads the RAGAS-style judge result
- chooses the next action under a hard retry budget
- records action history and terminates safely

Actions supported now:
- `stop_accept`
- `stop_best_effort`
- `answer_regeneration_only`
- `targeted_answer_repair`
- `citation_strict_repair`
- `full_pipeline_rerun_once`

Retry safety:
- hard cap on total rounds
- hard cap on full pipeline reruns
- hard cap on answer regenerations
- hard cap on repairs
- stops if quality does not improve after a retry

Why this matters:
- without a runtime controller, self-reflection and judge scores are only observations
- this layer turns them into actual execution decisions
- it also prevents infinite rerun loops and uncontrolled cost growth

Current honest limitation:
- the controller currently supports one full pipeline rerun, not a deeply targeted sub-query/source rerun planner
- that means the escalation behavior is real and bounded, but still not as surgical as a mature production router could become later

Main files:
- `agentic_document_intelligence/scripts/runtime_quality_gating.py`
- `agentic_document_intelligence/tests/test_runtime_quality_gating.py`

Output artifact:
- `artifacts/experiments/<run_id>/answers/runtime_quality_gating_report.json`
