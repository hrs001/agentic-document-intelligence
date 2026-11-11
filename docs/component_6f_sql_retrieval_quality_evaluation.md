# Component 6F: SQL Retrieval Quality Evaluation

This component evaluates text-to-SQL quality end to end using database-grounded benchmark queries.

What it does:
- runs real user-style questions through text-to-SQL generation
- validates and executes the generated SQL safely
- runs a trusted reference SQL for each benchmark case
- compares the actual returned rows against the expected rows from the database

Why:
- SQL quality should be judged by answer correctness, not just by whether a query executes
- this gives a realistic signal before later source routing and orchestration use SQL as one retrieval source

Main script:
- `agentic_document_intelligence/scripts/evaluate_sql_retrieval_quality.py`

Output:
- `artifacts/experiments/<run_id>/text_to_sql/sql_retrieval_quality_report.json`
