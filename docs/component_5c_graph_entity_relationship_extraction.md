# Component 5C: Entity and Relationship Extraction

This component runs the chosen extraction model over the packaged graph inputs and stores reusable extraction artifacts.

Active extraction model:
- `gpt-5.4-mini`

What it does:
- reads the packaged graph input records
- runs entity and relationship extraction once per graph input
- preserves provenance:
  - source graph input id
  - source parent id
  - source child ids
  - section title
  - page range
- writes a reusable artifact for later graph normalization and Kuzu loading

Why:
- graph extraction is an offline preprocessing step
- it should not be rerun during ordinary retrieval or app testing
- later graph loading into Kuzu should consume this saved artifact

Main script:
- `agentic_document_intelligence/scripts/extract_graph_entities.py`

Output:
- `artifacts/experiments/<run_id>/graph_extraction/<document_id>_graph_extraction.json`
