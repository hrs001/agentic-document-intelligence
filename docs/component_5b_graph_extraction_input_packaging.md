# Component 5B: Graph Extraction Input Packaging

This component packages stable graph-extraction input records from the existing chunk artifact.

What it does:
- uses the already-built chunk artifact as the source of truth
- creates one graph input record per unique text parent chunk
- preserves child-chunk provenance for later citation and traceability
- excludes table chunks from the active graph extraction path for now

Why:
- later graph extraction should run once on stable reusable input records
- parent-sized text units provide more context than child chunks
- provenance must be preserved so extracted graph facts can be traced back to the source corpus

Main script:
- `agentic_document_intelligence/scripts/package_graph_extraction_inputs.py`

Output:
- `artifacts/experiments/<run_id>/graph_inputs/<document_id>_graph_inputs.json`

Important note:
- the active extraction model target in this packaging step is `gpt-5.4-mini`
- table chunks are excluded from the current graph path because this first graph pass is focused on narrative entity/relationship extraction
