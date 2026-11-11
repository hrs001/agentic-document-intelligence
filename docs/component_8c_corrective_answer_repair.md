# Component 8C: Corrective Answer Repair

This component is the answer-stage corrective RAG loop.

What it does:
- consumes the fused evidence bundle
- consumes the current grounded answer
- consumes the self-reflective critique result
- chooses a targeted repair strategy
- regenerates a stricter repaired answer only when correction is needed
- re-checks the repaired answer with deterministic validation

Repair strategies:
- `no_op`
  use the original answer when critique says no correction is needed
- `targeted_coverage_repair`
  repair missing or unsupported sub-query coverage
- `conflict_aware_regeneration`
  repair answers that use conflicting evidence without proper uncertainty
- `citation_strict_regeneration`
  repair citation mismatches or missing inline citations
- `general_grounded_regeneration`
  fallback when correction is needed but no narrower repair strategy dominates

Why this design:
- corrective RAG should not blindly rerun the whole pipeline
- many answer failures can be repaired from the already fused evidence bundle
- this keeps cost and latency lower than full retrieval retries

Current limitation:
- this component currently repairs from the available fused evidence
- if the real problem is missing upstream retrieval evidence, a later retrieval-stage corrective loop should escalate back to targeted retrieval

Main files:
- `agentic_document_intelligence/scripts/corrective_answer_repair.py`
- `agentic_document_intelligence/tests/test_corrective_answer_repair.py`

Output artifact:
- `artifacts/experiments/<run_id>/answers/corrective_answer_repair.json`
