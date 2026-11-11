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


DEFAULT_INPUT = (
    "artifacts/experiments/component8_cross_source_evidence_fusion_sample/"
    "answer_context/cross_source_fused_evidence.json"
)
DEFAULT_MODEL = "gpt-5-mini"

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer_markdown": {"type": "string"},
        "used_fact_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "fact_id": {"type": "string"},
                    "source_type": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["fact_id", "source_type", "reason"],
            },
        },
        "unanswered_sub_queries": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["answer_markdown", "used_fact_ids", "citations", "unanswered_sub_queries", "confidence"],
}

SYSTEM_PROMPT = """You are a grounded answer generation module for a production RAG system.
Return only valid JSON.

Rules:
- Answer only from the provided fused evidence facts.
- When compressed_units are provided, treat them as the primary query-focused context, but cite only the original supported_fact_ids.
- Do not invent facts, citations, page numbers, or relationships.
- Every substantial claim in the answer_markdown must include inline fact-id citations like [sql::sq_1::1] or [graph_edge::e1].
- Prefer SQL facts for numeric values, graph facts for entity relationships, and vector facts for narrative statements.
- If evidence is incomplete, say so clearly.
- used_fact_ids must be a subset of the provided fact ids.
- citations must explain why each cited fact was used.
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def load_fused_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"] if "result" in payload else payload


def build_answer_input(fused_bundle: dict[str, Any], max_facts_per_sub_query: int = 8) -> dict[str, Any]:
    sub_queries = []
    allowed_fact_ids = []
    fact_catalog = []
    for sub_query in fused_bundle.get("sub_query_fusions", []):
        selected_facts = sub_query.get("facts", [])[:max_facts_per_sub_query]
        compressed_units = sub_query.get("compressed_units", [])
        sub_queries.append(
            {
                "sub_query_id": sub_query["sub_query_id"],
                "original_sub_query": sub_query["original_sub_query"],
                "resolved_sub_query": sub_query.get("resolved_sub_query", sub_query["original_sub_query"]),
                "facts": (
                    [
                        {
                            "fact_id": fact["fact_id"],
                            "source_type": fact["source_type"],
                            "fact_type": fact["fact_type"],
                            "summary": fact["summary"],
                        }
                        for fact in selected_facts
                    ]
                    if compressed_units
                    else selected_facts
                ),
                "compressed_units": compressed_units,
                "conflict_signals": sub_query.get("conflict_signals", []),
                "overlap_signals": sub_query.get("overlap_signals", []),
            }
        )
        for fact in selected_facts:
            allowed_fact_ids.append(fact["fact_id"])
            fact_catalog.append(
                {
                    "fact_id": fact["fact_id"],
                    "source_type": fact["source_type"],
                    "fact_type": fact["fact_type"],
                    "summary": fact["summary"],
                }
            )

    return {
        "original_query": fused_bundle["original_query"],
        "sub_queries": sub_queries,
        "allowed_fact_ids": allowed_fact_ids,
        "fact_catalog": fact_catalog,
        "bundle_summary": fused_bundle.get("bundle_summary", {}),
        "compression_context": fused_bundle.get("compression_context", {}),
    }


def sanitize_answer_payload(payload: dict[str, Any], allowed_fact_ids: set[str], fact_source_map: dict[str, str]) -> dict[str, Any]:
    used_fact_ids = []
    for fact_id in payload.get("used_fact_ids", []):
        cleaned = str(fact_id).strip()
        if cleaned in allowed_fact_ids and cleaned not in used_fact_ids:
            used_fact_ids.append(cleaned)

    citations = []
    for citation in payload.get("citations", []):
        fact_id = str(citation.get("fact_id", "")).strip()
        if fact_id not in allowed_fact_ids:
            continue
        citations.append(
            {
                "fact_id": fact_id,
                "source_type": fact_source_map.get(fact_id, str(citation.get("source_type", "")).strip()),
                "reason": str(citation.get("reason", "")).strip(),
            }
        )

    return {
        "answer_markdown": str(payload.get("answer_markdown", "")).strip(),
        "used_fact_ids": used_fact_ids,
        "citations": citations,
        "unanswered_sub_queries": [str(item).strip() for item in payload.get("unanswered_sub_queries", []) if str(item).strip()],
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
    }


def generate_grounded_answer(
    fused_bundle: dict[str, Any],
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    answer_input = build_answer_input(fused_bundle)
    fact_source_map = {item["fact_id"]: item["source_type"] for item in answer_input["fact_catalog"]}
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer_generation",
                "schema": ANSWER_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(answer_input)},
        ],
    )
    payload = json.loads(completion.choices[0].message.content or "{}")
    sanitized = sanitize_answer_payload(payload, set(answer_input["allowed_fact_ids"]), fact_source_map)
    return {
        "original_query": fused_bundle["original_query"],
        "model": model,
        "answer_input_summary": {
            "sub_query_count": len(answer_input["sub_queries"]),
            "allowed_fact_count": len(answer_input["allowed_fact_ids"]),
            "compression_applied": bool(answer_input.get("compression_context", {}).get("applied", False)),
        },
        "answer_markdown": sanitized["answer_markdown"],
        "used_fact_ids": sanitized["used_fact_ids"],
        "citations": sanitized["citations"],
        "unanswered_sub_queries": sanitized["unanswered_sub_queries"],
        "confidence": sanitized["confidence"],
        "usage": getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else {},
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "grounded_answer_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a grounded final answer from a fused evidence bundle.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="component8_grounded_answer_generation")
    args = parser.parse_args()

    fused_bundle = load_fused_bundle(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    result = generate_grounded_answer(fused_bundle, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "model": result["model"],
                "used_fact_count": len(result["used_fact_ids"]),
                "confidence": result["confidence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
