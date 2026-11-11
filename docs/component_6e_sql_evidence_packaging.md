# Component 6E: SQL Evidence Packaging

This component packages validated SQL execution output into a compact artifact for later routing and answer generation.

What it does:
- reads the SQL execution report from the previous step
- keeps the validated SQL, target tables, and confidence label
- keeps a bounded preview of returned rows
- builds one assembled evidence text block for downstream answer synthesis

Why:
- later orchestration should consume a stable SQL evidence bundle instead of raw execution output
- this keeps result handling lightweight for Lambda and avoids re-running SQL formatting logic downstream

Main script:
- `agentic_document_intelligence/scripts/package_sql_evidence.py`

Output:
- `artifacts/experiments/<run_id>/text_to_sql/sql_evidence_bundle.json`
