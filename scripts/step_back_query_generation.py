import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_NAME = "gpt-5-mini"

SYSTEM_PROMPT = """You generate one step-back retrieval query for a RAG system.
Return only valid JSON.

Rules:
- Preserve the user's meaning.
- Produce exactly one broader, higher-level query that helps retrieve context for answering the original question.
- The step-back query should be more conceptual or explanatory than the original query.
- Do not broaden so far that the subject changes.
- Do not add new entities, time periods, or claims not implied by the original query.
- Keep it concise and retrieval-ready.
- Preserve important retrieval terms from the original query when they are likely useful keywords, such as product names, governance terms, and finance terms like capex.
- If the original query is already broad and conceptual, keep the step-back query close to it.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "original_query": {"type": "string"},
        "step_back_query": {"type": "string"},
        "broadening_strategy": {
            "type": "string",
            "enum": [
                "conceptualize_metric",
                "broaden_to_context",
                "broaden_to_explanation",
                "keep_close",
                "broaden_to_governance",
                "broaden_to_segment_context",
            ],
        },
    },
    "required": ["original_query", "step_back_query", "broadening_strategy"],
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_result(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    step_back_query = str(payload.get("step_back_query", "")).strip() or query.strip()
    return {
        "original_query": query,
        "step_back_query": step_back_query,
        "broadening_strategy": str(payload.get("broadening_strategy", "keep_close")).strip() or "keep_close",
    }


def generate_step_back_query(query: str, model: str = MODEL_NAME, client: OpenAI | None = None) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "step_back_query_generation",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"query": query})},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    return normalize_result(query, payload)


def write_report(project_root: Path, run_id: str, result: dict[str, Any], model: str) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "query_transforms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "step_back_query_generation_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a step-back query.")
    parser.add_argument("--query", default="What happened with Azure growth and what did management say about AI demand?")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_step_back_query_generation")
    args = parser.parse_args()

    result = generate_step_back_query(args.query, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result, args.model)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
