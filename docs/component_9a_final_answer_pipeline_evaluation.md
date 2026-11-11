# Component 9A: Final Answer Pipeline Evaluation

This component evaluates the full final answer path end to end:

1. grounded answer generation
2. self-reflective critique
3. corrective answer repair
4. deterministic final answer scoring

What it measures:
- initial pass rate before repair
- final pass rate after repair
- how often repair was applied
- how often repair actually recovered a failed answer
- runtime per case

Why this matters:
- good answer generation alone is not enough
- we need to know whether critique and corrective loops really improve the final returned answer
- this also surfaces when the remaining problem is upstream evidence quality rather than answer-text repair

Current honest limitation:
- a repair can be applied even when the initial answer already passes deterministic evaluation
- that means the critique layer is still somewhat conservative and may add cost without always improving final pass rate
- the hardest remaining benchmark miss appears to be an upstream evidence issue, not just answer phrasing

Main files:
- `agentic_document_intelligence/scripts/evaluate_final_answer_pipeline.py`
- `agentic_document_intelligence/tests/test_evaluate_final_answer_pipeline.py`

Output artifact:
- `artifacts/experiments/<run_id>/answers/final_answer_pipeline_eval_report.json`
