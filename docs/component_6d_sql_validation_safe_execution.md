# Component 6D: SQL Validation and Safe Execution

This component validates generated SQL and executes it in read-only mode against the SQLite database.

What it does:
- enforces a single read-only `SELECT` statement
- blocks destructive SQL keywords
- opens SQLite in read-only mode
- adds a SQLite authorizer so write operations are denied at runtime
- executes the validated query and serializes the result rows

Main script:
- `agentic_document_intelligence/scripts/validate_and_execute_sql.py`

Output:
- `artifacts/experiments/<run_id>/text_to_sql/sql_execution_report.json`
