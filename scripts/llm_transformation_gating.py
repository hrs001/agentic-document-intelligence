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

from agentic_document_intelligence.scripts.transformation_gating import (
    MAX_SUB_QUERIES,
    select_transformations,
)


MODEL_NAME = "gpt-5-mini"

SYSTEM_PROMPT = """You select which query-transformation tools should run for a RAG system.
Return only valid JSON.

Rules:
- You are only called for ambiguous queries after a deterministic policy flags uncertainty.
- Preserve cost discipline. Do not turn on every transformation unless clearly justified.
- Respect the hard cap of 3 sub-queries.
- Use decomposition only when the user query truly contains multiple information needs.
- Use multi-query when lexical variation is likely useful.
- Use step-back when broader explanatory context would help.
- Use HyDE only when the query is sparse or retrieval is likely difficult.
- Output concise reasons.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "run_decomposition": {"type": "boolean"},
        "run_multi_query": {"type": "boolean"},
        "run_step_back": {"type": "boolean"},
        "run_hyde": {"type": "boolean"},
        "estimated_sub_queries": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_SUB_QUERIES,
        },
        "selector_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "run_decomposition",
        "run_multi_query",
        "run_step_back",
        "run_hyde",
        "estimated_sub_queries",
        "selector_notes",
    ],
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_result(query: str, baseline: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": query,
        "max_sub_queries": MAX_SUB_QUERIES,
        "estimated_sub_queries": min(
            MAX_SUB_QUERIES,
            max(1, int(payload.get("estimated_sub_queries", baseline["estimated_sub_queries"]))),
        ),
        "run_decomposition": bool(payload.get("run_decomposition", baseline["run_decomposition"])),
        "run_multi_query": bool(payload.get("run_multi_query", baseline["run_multi_query"])),
        "run_step_back": bool(payload.get("run_step_back", baseline["run_step_back"])),
        "run_hyde": bool(payload.get("run_hyde", baseline["run_hyde"])),
        "is_ambiguous_case": baseline["is_ambiguous_case"],
        "ambiguity_signals": baseline["ambiguity_signals"],
        "reasons": baseline["reasons"],
        "selector_notes": [str(item).strip() for item in payload.get("selector_notes", []) if str(item).strip()],
        "llm_selector_applied": True,
    }


def select_transformations_with_llm(query: str, model: str = MODEL_NAME, client: OpenAI | None = None) -> dict[str, Any]:
    baseline = select_transformations(query)
    if not baseline["is_ambiguous_case"]:
        return {
            **baseline,
            "selector_notes": [],
            "llm_selector_applied": False,
        }

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
                "name": "llm_transformation_gating",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "deterministic_baseline": baseline,
                        "max_sub_queries": MAX_SUB_QUERIES,
                    }
                ),
            },
        ],
    )
    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    return normalize_result(query, baseline, payload)


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
    report_path = output_dir / "llm_transformation_gating_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-assisted transformation gating for ambiguous cases.")
    parser.add_argument("--query", default="What happened with Azure growth and what did management say about AI demand?")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_llm_transformation_gating")
    args = parser.parse_args()

    result = select_transformations_with_llm(args.query, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result, args.model)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "llm_selector_applied": result["llm_selector_applied"],
                "estimated_sub_queries": result["estimated_sub_queries"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
