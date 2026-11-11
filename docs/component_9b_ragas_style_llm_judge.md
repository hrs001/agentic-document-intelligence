# Component 9B: RAGAS-Style LLM Judge

This component adds an LLM-as-Judge layer using RAGAS-style evaluation metrics.

Metrics implemented:
- `faithfulness`
  whether answer claims are supported by the available evidence
- `answer_relevancy`
  whether the answer addresses the user query directly
- `context_precision`
  whether the cited / used context is focused and relevant instead of noisy
- `citation_grounding`
  whether major claims are properly tied to citations

Why this is not raw RAGAS library:
- this project must remain practical for AWS Lambda serverless deployment
- a lighter custom judge is easier to control and integrate with our fused-evidence format
- the metric concepts are aligned to RAGAS-style industry metrics even though the implementation is custom

How it is used:
- it can run as a standalone judge on a final answer
- it is also integrated into the final answer pipeline evaluation so the judge runs for every evaluated query

Current honest limitation:
- the LLM judge and deterministic evaluator can disagree
- that is expected and useful, but it means the judge should be treated as an additional quality signal, not a perfect oracle

Main files:
- `agentic_document_intelligence/scripts/ragas_style_llm_judge.py`
- `agentic_document_intelligence/tests/test_ragas_style_llm_judge.py`

Output artifact:
- `artifacts/experiments/<run_id>/answers/ragas_style_llm_judge.json`
