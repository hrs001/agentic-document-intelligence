# Component 6A: SQLite Database Build

This component builds the reusable local SQLite database for the synthetic analyst dataset.

What it does:
- reads the fixed CSV tables from `corpus/datasets/`
- creates a local SQLite database file
- loads the three demo tables
- writes a schema/build report for later text-to-SQL steps

Why:
- this is the cheapest SQL backend for the demo
- the database can be built once offline and reused later
- for Lambda deployment, the `.sqlite` file can later be stored in S3 and copied into `/tmp`

Main script:
- `agentic_document_intelligence/scripts/build_sqlite_demo_db.py`

Outputs:
- SQLite database file under `artifacts/experiments/<run_id>/sql_db/`
- build report under `artifacts/experiments/<run_id>/sql_db/sqlite_build_report.json`
