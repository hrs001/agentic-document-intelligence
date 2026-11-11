import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.query_decomposition import decompose_query


MODEL_NAME = "gemini-3-flash-preview"

SYSTEM_PROMPT = """You are a query decomposition module for a RAG system.
Return only valid JSON.

Rules:
- Preserve the user's meaning exactly.
- Split only when the user query has multiple distinct information needs.
- Keep comparison requests atomic.
- Keep a simple factual query atomic.
- For list-style finance queries, split into one sub-query per requested metric when that improves retrieval.
- Do not invent new asks that the user did not request.
- Output concise sub-queries that can be sent to retrieval systems.
- Prefer explicit question form for each sub-query.
- If the query clearly refers to Microsoft implicitly, make Microsoft explicit in each sub-query.
- Normalize vague request phrasing like "tell me the CEO" into canonical retrieval-friendly form like "Who is the CEO of Microsoft?"
- Normalize metric-only requests like "Revenue, operating income, and cash flow for FY2025" into explicit finance questions.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "required": [
        "needs_decomposition",
        "sub_queries",
        "reasoning_type",
        "decomposition_strategy",
    ],
    "properties": {
        "needs_decomposition": {"type": "boolean"},
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reasoning_type": {
            "type": "string",
            "enum": [
                "single_intent",
                "multi_question",
                "multi_intent",
                "list_intent",
                "comparison",
                "ambiguous",
            ],
        },
        "decomposition_strategy": {
            "type": "string",
            "enum": [
                "none",
                "question_boundary_split",
                "conjunction_split",
                "list_item_split",
                "comparison_atomic",
                "semantic_split",
            ],
        },
    },
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_result(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    sub_queries = [str(item).strip() for item in payload.get("sub_queries", []) if str(item).strip()]
    if not sub_queries:
        sub_queries = [query.strip()]
    return {
        "sanitized_query": query.strip(),
        "needs_decomposition": bool(payload.get("needs_decomposition", len(sub_queries) > 1)),
        "sub_queries": sub_queries,
        "reasoning_type": payload.get("reasoning_type", "ambiguous"),
        "decomposition_strategy": payload.get("decomposition_strategy", "semantic_split"),
    }


def decompose_query_with_gemini(query: str, model: str = MODEL_NAME, client: genai.Client | None = None) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in agentic_document_intelligence/.env")

    gemini_client = client or genai.Client(api_key=api_key)
    response = gemini_client.models.generate_content(
        model=model,
        contents=json.dumps(
            {
                "query": query,
                "deterministic_baseline": decompose_query(query),
            }
        ),
        config=types.GenerateContentConfig(
            systemInstruction=SYSTEM_PROMPT,
            responseMimeType="application/json",
            responseSchema=RESPONSE_SCHEMA,
            temperature=0,
        ),
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    payload = json.loads(text)
    return normalize_result(query, payload)


def write_report(project_root: Path, run_id: str, query: str, result: dict[str, Any], model: str) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "query_transforms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gemini_query_decomposition_report.json"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "query": query,
        "result": result,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini-assisted query decomposition.")
    parser.add_argument(
        "--query",
        default="Explain Microsoft's FY2025 revenue growth and identify who the CEO is.",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_gemini_query_decomposition")
    args = parser.parse_args()

    result = decompose_query_with_gemini(args.query, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, args.query, result, args.model)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "needs_decomposition": result["needs_decomposition"],
                "sub_query_count": len(result["sub_queries"]),
                "model": args.model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
