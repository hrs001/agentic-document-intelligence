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

SYSTEM_PROMPT = """You generate a short hypothetical answer passage for HyDE-style retrieval.
Return only valid JSON.

Rules:
- Preserve the user's subject and intent.
- Write a short, retrieval-friendly hypothetical passage, not a final answer.
- Keep it concise: 2 to 4 sentences only.
- Include important entities or keywords from the query when relevant.
- Do not fabricate specific numbers, dates, or claims unless already implied by the query.
- Prefer general factual scaffolding over detailed invented facts.
- Do not add unrelated topics.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "original_query": {"type": "string"},
        "hypothetical_passage": {"type": "string"},
        "generation_style": {
            "type": "string",
            "enum": [
                "factual_stub",
                "contextual_stub",
                "comparison_stub",
                "governance_stub",
                "financial_stub",
            ],
        },
    },
    "required": ["original_query", "hypothetical_passage", "generation_style"],
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_result(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    passage = str(payload.get("hypothetical_passage", "")).strip() or query.strip()
    return {
        "original_query": query,
        "hypothetical_passage": passage,
        "generation_style": str(payload.get("generation_style", "contextual_stub")).strip() or "contextual_stub",
    }


def generate_hyde_passage(query: str, model: str = MODEL_NAME, client: OpenAI | None = None) -> dict[str, Any]:
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
                "name": "hyde_query_generation",
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
    report_path = output_dir / "hyde_query_generation_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a HyDE hypothetical passage.")
    parser.add_argument("--query", default="What happened with Azure growth and what did management say about AI demand?")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_hyde_query_generation")
    args = parser.parse_args()

    result = generate_hyde_passage(args.query, model=args.model)
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
