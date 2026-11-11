# Component 6B: SQL Schema Packaging

This component packages the reusable SQLite schema for later text-to-SQL prompting.

What it does:
- reads the built SQLite database
- captures table names, column names, and SQLite types
- captures row counts and a few sample rows
- builds a compact schema text artifact for later prompting

Why:
- later text-to-SQL generation should use exact schema context
- this avoids table/column hallucination and keeps prompts deterministic

Main script:
- `agentic_document_intelligence/scripts/package_sql_schema.py`

Output:
- `artifacts/experiments/<run_id>/sql_schema/sql_schema_package.json`
