# Component 8B: Self-Reflective Answer Critique

This component critiques a grounded answer before corrective repair is attempted.

What it does:
- runs deterministic checks over the generated answer
- runs an LLM-based critique pass over the answer and fused evidence
- merges both into one final critique object
- flags whether correction is needed before the answer is accepted

Deterministic checks:
- used facts missing inline citations
- inline fact ids not declared in the answer payload
- unknown fact ids not present in the fused evidence
- unanswered or uncovered sub-queries
- conflict-linked facts used without acknowledging uncertainty or conflict

LLM critique checks:
- whether the answer is grounded in the provided fused evidence
- whether the answer completely addresses the original user query
- whether the answer ignores conflicts or caveats
- what targeted repair action should be taken if correction is needed

Why this design:
- real-world answer failures are not only factual hallucinations
- many failures are partial coverage, weak caveats, bad citation behavior, or ignored evidence conflicts
- deterministic checks catch obvious structural problems cheaply
- the LLM critique catches more semantic problems than brittle rules alone

Main files:
- `agentic_document_intelligence/scripts/self_reflective_answer_critique.py`
- `agentic_document_intelligence/tests/test_self_reflective_answer_critique.py`

Output artifact:
- `artifacts/experiments/<run_id>/answers/self_reflective_answer_critique.json`
