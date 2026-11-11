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
MAX_REWRITES = 3

SYSTEM_PROMPT = """You generate a small set of alternative retrieval queries for a RAG system.
Return only valid JSON.

Rules:
- Preserve the user's meaning exactly.
- Generate at most 3 rewrites.
- Each rewrite must be short and retrieval-ready.
- Rewrites must be meaningfully different from each other in phrasing or retrieval angle.
- Do not broaden the scope of the query.
- Do not add new facts, entities, or time periods not implied by the original query.
- Avoid placeholder phrases and vague wording.
- Keep comparison queries as comparisons.
- For finance questions, one rewrite may use disclosure/report wording if still faithful.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "original_query": {"type": "string"},
        "rewrites": {
            "type": "array",
            "minItems": 2,
            "maxItems": MAX_REWRITES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "angle": {
                        "type": "string",
                        "enum": [
                            "direct",
                            "financial_disclosure",
                            "management_commentary",
                            "segment_product",
                            "governance_risk",
                            "comparison",
                        ],
                    },
                },
                "required": ["query", "angle"],
            },
        },
    },
    "required": ["original_query", "rewrites"],
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_result(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    rewrites = []
    seen = set()
    for item in payload.get("rewrites", []):
        normalized_query = str(item.get("query", "")).strip()
        angle = str(item.get("angle", "")).strip() or "direct"
        if not normalized_query:
            continue
        dedup_key = normalized_query.lower()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rewrites.append({"query": normalized_query, "angle": angle})
        if len(rewrites) >= MAX_REWRITES:
            break

    return {
        "original_query": query,
        "rewrite_count": len(rewrites),
        "rewrites": rewrites,
    }


def generate_multi_queries(query: str, model: str = MODEL_NAME, client: OpenAI | None = None) -> dict[str, Any]:
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
                "name": "multi_query_generation",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"query": query, "max_rewrites": MAX_REWRITES})},
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
    report_path = output_dir / "multi_query_generation_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multiple retrieval rewrites.")
    parser.add_argument("--query", default="What happened with Azure growth and what did management say about AI demand?")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_multi_query_generation")
    args = parser.parse_args()

    result = generate_multi_queries(args.query, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result, args.model)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "rewrite_count": result["rewrite_count"],
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
