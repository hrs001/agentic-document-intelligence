# Component 5F: Kuzu Graph Storage Build

This component loads the validated graph artifact into a local Kuzu database.

What it does:
- creates a fresh local Kuzu database on disk
- creates the `Entity` node table
- creates the `GraphEdge` relationship table
- loads validated nodes and edges once
- stores provenance fields as JSON strings for later retrieval hydration
- writes a build report with verification counts

Why:
- graph extraction should be an offline preprocessing step
- later graph retrieval should query Kuzu directly instead of rerunning extraction

Main script:
- `agentic_document_intelligence/scripts/build_kuzu_graph.py`

Outputs:
- Kuzu database directory under `artifacts/experiments/<run_id>/kuzu_db/<document_id>/`
- build report at `artifacts/experiments/<run_id>/graph_storage/kuzu_graph_build_report.json`

Important note:
- this is the first step that creates the actual reusable local graph database
- later Lambda deployment should upload the built Kuzu database files to S3 for reuse
