# Component 8A: Cross-Source Evidence Fusion and Conflict Handling

This component converts raw multi-source orchestration output into one answer-ready fused evidence bundle.

What it does:
- normalizes vector, graph, and SQL outputs into a shared fact schema
- preserves provenance such as citations, parent ids, section titles, SQL tables, and graph relations
- detects overlap signals across sources
- detects conflict signals that later answer verification and corrective loops can react to
- assembles a fused per-sub-query context block for answer generation

What counts as overlap right now:
- the same entity appears across multiple sources
- the same parent context id appears across multiple sources
- the same section title appears across multiple sources

What counts as conflict right now:
- SQL value mismatch:
  the same SQL identifier group and metric column appear with different values
- graph relation mismatch:
  the same source-target entity pair appears with different relation types

Why this design:
- answer generation should not consume raw orchestration outputs directly
- SQL, graph, and vector evidence have different shapes and confidence profiles
- this layer makes later answer generation, self-reflection, and corrective retrieval more disciplined

Current limitation:
- this component detects explicit structural conflicts, not every semantic contradiction
- deeper semantic conflict judgment should be handled later by the self-reflective and corrective answer loop

Main files:
- `agentic_document_intelligence/scripts/cross_source_evidence_fusion.py`
- `agentic_document_intelligence/tests/test_cross_source_evidence_fusion.py`

Output artifact:
- `artifacts/experiments/<run_id>/answer_context/cross_source_fused_evidence.json`
