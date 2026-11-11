# Component 6C: Text-to-SQL Generation

This component generates one read-only SQLite query from a user question using the packaged schema context.

What it does:
- reads the packaged SQL schema artifact
- uses an LLM to generate one `SELECT` query
- keeps target table names and a confidence label
- rejects obviously unsafe non-read-only SQL during result normalization

Main script:
- `agentic_document_intelligence/scripts/generate_text_to_sql.py`

Output:
- `artifacts/experiments/<run_id>/text_to_sql/text_to_sql_generation_report.json`

Important note:
- this step only generates SQL
- SQL safety validation and execution remain in the next sub-component
