import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_NAME = "gpt-5-mini"

SYSTEM_PROMPT = """You are a text-to-SQL generation module for a SQLite analytics database.
Return only valid JSON.

Rules:
- Generate exactly one read-only SQLite SELECT query.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA, or transaction statements.
- Use only the tables and columns present in the provided schema.
- Prefer simple, readable SQL.
- Return only the columns needed to answer the user question.
- Do not include extra descriptive columns unless the question explicitly asks for them.
- When the question asks which entity leads on, exceeds, or is filtered by a specific metric, return both the entity column and that metric column.
- For ranking requests, order the rows correctly but do not add a synthetic rank column unless the user explicitly asks for rank numbers.
- If aggregation is needed, use explicit aliases.
- If the question specifies FY2025 or 2025, apply the right fiscal_year filter.
- If the question cannot be answered from the schema, return a safe fallback SELECT with low confidence and explain why in rationale.
- Do not invent tables, columns, joins, or metrics.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "user_query": {"type": "string"},
        "target_tables": {
            "type": "array",
            "items": {"type": "string"},
        },
        "sql_query": {"type": "string"},
        "rationale": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["user_query", "target_tables", "sql_query", "rationale", "confidence"],
}

FORBIDDEN_SQL_PATTERNS = [
    r"\binsert\b",
    r"\bupdate\b",
    r"\bdelete\b",
    r"\bdrop\b",
    r"\balter\b",
    r"\bcreate\b",
    r"\battach\b",
    r"\bpragma\b",
    r"\bbegin\b",
    r"\bcommit\b",
    r"\brollback\b",
]


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def load_schema_package(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def normalize_sql(sql: str) -> str:
    cleaned = sql.strip()
    cleaned = re.sub(r";+\s*$", "", cleaned)
    return cleaned


def sanitize_generation_result(user_query: str, payload: dict[str, Any], schema_package: dict[str, Any]) -> dict[str, Any]:
    sql_query = normalize_sql(str(payload.get("sql_query", "")).strip())
    if not sql_query:
        raise ValueError("Generated SQL is empty")

    normalized_sql = sql_query.lower()
    if not normalized_sql.startswith("select"):
        raise ValueError("Generated SQL is not a SELECT statement")
    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, normalized_sql):
            raise ValueError(f"Generated SQL contains forbidden pattern: {pattern}")

    known_tables = {table["table_name"] for table in schema_package["tables"]}
    target_tables = [str(item).strip() for item in payload.get("target_tables", []) if str(item).strip()]
    target_tables = [table for table in target_tables if table in known_tables]

    return {
        "user_query": user_query,
        "target_tables": target_tables,
        "sql_query": sql_query,
        "rationale": str(payload.get("rationale", "")).strip(),
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
    }


def generate_text_to_sql(
    user_query: str,
    schema_package: dict[str, Any],
    model: str = MODEL_NAME,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    request_kwargs = {
        "model": model,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "text_to_sql_generation",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_query": user_query,
                        "schema_package": {
                            "table_count": schema_package["table_count"],
                            "tables": schema_package["tables"],
                            "prompt_schema_text": schema_package["prompt_schema_text"],
                        },
                    }
                ),
            },
        ],
    }
    try:
        completion = openai_client.chat.completions.create(
            temperature=0,
            **request_kwargs,
        )
    except BadRequestError as exc:
        message = str(exc)
        if "temperature" not in message or "default (1)" not in message:
            raise
        completion = openai_client.chat.completions.create(**request_kwargs)

    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    return sanitize_generation_result(user_query, payload, schema_package)


def write_report(project_root: Path, run_id: str, result: dict[str, Any], model: str) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "text_to_sql"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "text_to_sql_generation_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a read-only SQLite query from a user question.")
    parser.add_argument(
        "--schema-path",
        default="artifacts/experiments/component6_sql_schema_packaging_live/sql_schema/sql_schema_package.json",
    )
    parser.add_argument(
        "--query",
        default="Which segment had the highest revenue in FY2025?",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component6_text_to_sql_generation")
    args = parser.parse_args()

    schema_package = load_schema_package(PROJECT_ROOT / "agentic_document_intelligence" / args.schema_path)
    result = generate_text_to_sql(args.query, schema_package, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result, args.model)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "target_tables": result["target_tables"],
                "confidence": result["confidence"],
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
