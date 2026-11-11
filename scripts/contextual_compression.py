import argparse
import copy
import json
import os
import sys
import time
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
DEFAULT_MIN_FACTS_FOR_COMPRESSION = 30
DEFAULT_MIN_SUB_QUERIES_FOR_COMPRESSION = 2
DEFAULT_MIN_VECTOR_FACTS_FOR_COMPRESSION = 8
DEFAULT_MAX_FACTS_PER_SUB_QUERY = 8
DEFAULT_MAX_UNITS_PER_SUB_QUERY = 4

MODEL_PRICING_PER_M_TOKEN = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

COMPRESSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sub_query_id": {"type": "string"},
                    "compressed_units": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "unit_id": {"type": "string"},
                                "summary_text": {"type": "string"},
                                "supported_fact_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                },
                                "compression_type": {"type": "string"},
                            },
                            "required": ["unit_id", "summary_text", "supported_fact_ids", "compression_type"],
                        },
                    },
                },
                "required": ["sub_query_id", "compressed_units"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "compression_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sub_queries", "confidence", "compression_notes"],
}

SYSTEM_PROMPT = """You are a citation-preserving contextual compression module for a production RAG system.
Return only valid JSON.

Rules:
- Rewrite the provided evidence into compact, query-specific summary units.
- Do not invent claims.
- Each compressed unit must cite only the original supported_fact_ids provided in the input.
- Prefer one unit per distinct answerable point rather than paraphrasing the same evidence repeatedly.
- Preserve conflicts or uncertainty when the evidence is mixed.
- Keep each summary_text concise and useful for final answer generation.
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"] if "result" in payload else payload


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_M_TOKEN.get(model)
    if not pricing:
        return 0.0
    return ((prompt_tokens / 1_000_000) * pricing["input"]) + ((completion_tokens / 1_000_000) * pricing["output"])


def build_compression_input(
    fused_bundle: dict[str, Any],
    max_facts_per_sub_query: int = DEFAULT_MAX_FACTS_PER_SUB_QUERY,
) -> dict[str, Any]:
    sub_queries = []
    allowed_fact_ids: list[str] = []
    for sub_query in fused_bundle.get("sub_query_fusions", []):
        selected_facts = sub_query.get("facts", [])[:max_facts_per_sub_query]
        for fact in selected_facts:
            allowed_fact_ids.append(fact["fact_id"])
        sub_queries.append(
            {
                "sub_query_id": sub_query["sub_query_id"],
                "original_sub_query": sub_query["original_sub_query"],
                "resolved_sub_query": sub_query.get("resolved_sub_query", sub_query["original_sub_query"]),
                "facts": [
                    {
                        "fact_id": fact["fact_id"],
                        "source_type": fact["source_type"],
                        "fact_type": fact["fact_type"],
                        "summary": fact["summary"],
                        "section_title": fact.get("section_title", ""),
                        "entities": fact.get("entities", [])[:6],
                    }
                    for fact in selected_facts
                ],
                "conflict_signals": sub_query.get("conflict_signals", []),
                "overlap_signals": sub_query.get("overlap_signals", []),
            }
        )
    return {
        "original_query": fused_bundle["original_query"],
        "bundle_summary": fused_bundle.get("bundle_summary", {}),
        "sub_queries": sub_queries,
        "allowed_fact_ids": allowed_fact_ids,
    }


def should_apply_contextual_compression(
    fused_bundle: dict[str, Any],
    min_facts_for_compression: int = DEFAULT_MIN_FACTS_FOR_COMPRESSION,
    min_sub_queries_for_compression: int = DEFAULT_MIN_SUB_QUERIES_FOR_COMPRESSION,
    min_vector_facts_for_compression: int = DEFAULT_MIN_VECTOR_FACTS_FOR_COMPRESSION,
) -> bool:
    bundle_summary = fused_bundle.get("bundle_summary", {})
    fact_count = int(bundle_summary.get("fact_count", 0))
    sub_query_count = int(bundle_summary.get("sub_query_count", 0))
    vector_fact_count = sum(
        1 for fact in fused_bundle.get("normalized_facts", []) if fact.get("source_type") == "vector_document"
    )
    return (
        fact_count >= min_facts_for_compression
        and sub_query_count >= min_sub_queries_for_compression
        and vector_fact_count >= min_vector_facts_for_compression
    )


def sanitize_compression_payload(
    payload: dict[str, Any],
    compression_input: dict[str, Any],
    max_units_per_sub_query: int = DEFAULT_MAX_UNITS_PER_SUB_QUERY,
) -> dict[str, Any]:
    allowed_fact_ids = set(compression_input["allowed_fact_ids"])
    by_sub_query = {item["sub_query_id"]: item for item in compression_input["sub_queries"]}
    sanitized_sub_queries = []

    for sub_query in compression_input["sub_queries"]:
        payload_sub_query = next(
            (item for item in payload.get("sub_queries", []) if str(item.get("sub_query_id", "")).strip() == sub_query["sub_query_id"]),
            None,
        )
        compressed_units = []
        if payload_sub_query:
            for index, unit in enumerate(payload_sub_query.get("compressed_units", []), start=1):
                supported_fact_ids = []
                for fact_id in unit.get("supported_fact_ids", []):
                    cleaned = str(fact_id).strip()
                    if cleaned in allowed_fact_ids and cleaned not in supported_fact_ids:
                        supported_fact_ids.append(cleaned)
                if not supported_fact_ids:
                    continue
                summary_text = str(unit.get("summary_text", "")).strip()
                if not summary_text:
                    continue
                compressed_units.append(
                    {
                        "unit_id": str(unit.get("unit_id", f"{sub_query['sub_query_id']}_cu_{index}")).strip() or f"{sub_query['sub_query_id']}_cu_{index}",
                        "summary_text": summary_text,
                        "supported_fact_ids": supported_fact_ids,
                        "compression_type": str(unit.get("compression_type", "query_focused_summary")).strip() or "query_focused_summary",
                    }
                )
                if len(compressed_units) >= max_units_per_sub_query:
                    break

        if not compressed_units:
            fallback_facts = by_sub_query[sub_query["sub_query_id"]]["facts"][:max_units_per_sub_query]
            for index, fact in enumerate(fallback_facts, start=1):
                compressed_units.append(
                    {
                        "unit_id": f"{sub_query['sub_query_id']}_fallback_{index}",
                        "summary_text": str(fact["summary"]).strip(),
                        "supported_fact_ids": [fact["fact_id"]],
                        "compression_type": "extractive_fallback",
                    }
                )

        sanitized_sub_queries.append(
            {
                "sub_query_id": sub_query["sub_query_id"],
                "compressed_units": compressed_units,
            }
        )

    return {
        "sub_queries": sanitized_sub_queries,
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
        "compression_notes": [str(item).strip() for item in payload.get("compression_notes", []) if str(item).strip()],
    }


def attach_contextual_compression(
    fused_bundle: dict[str, Any],
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
    min_facts_for_compression: int = DEFAULT_MIN_FACTS_FOR_COMPRESSION,
    min_sub_queries_for_compression: int = DEFAULT_MIN_SUB_QUERIES_FOR_COMPRESSION,
    min_vector_facts_for_compression: int = DEFAULT_MIN_VECTOR_FACTS_FOR_COMPRESSION,
) -> dict[str, Any]:
    result = copy.deepcopy(fused_bundle)
    if not should_apply_contextual_compression(
        fused_bundle,
        min_facts_for_compression=min_facts_for_compression,
        min_sub_queries_for_compression=min_sub_queries_for_compression,
        min_vector_facts_for_compression=min_vector_facts_for_compression,
    ):
        result["compression_context"] = {
            "applied": False,
            "reason": "bundle_below_threshold",
            "compression_model": None,
            "sub_queries": [],
        }
        return result

    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    compression_input = build_compression_input(fused_bundle)
    started_at = time.perf_counter()
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "contextual_compression",
                "schema": COMPRESSION_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(compression_input)},
        ],
    )
    latency_seconds = round(time.perf_counter() - started_at, 3)
    payload = json.loads(completion.choices[0].message.content or "{}")
    sanitized = sanitize_compression_payload(payload, compression_input)
    usage = getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    sub_query_map = {item["sub_query_id"]: item for item in sanitized["sub_queries"]}
    for sub_query in result.get("sub_query_fusions", []):
        sub_query["compressed_units"] = sub_query_map.get(sub_query["sub_query_id"], {}).get("compressed_units", [])

    result["compression_context"] = {
        "applied": True,
        "compression_model": model,
        "confidence": sanitized["confidence"],
        "compression_notes": sanitized["compression_notes"],
        "sub_queries": sanitized["sub_queries"],
        "usage": {
            **usage,
            "latency_seconds": latency_seconds,
            "estimated_cost_usd": round(estimate_cost_usd(model, prompt_tokens, completion_tokens), 6),
        },
    }
    return result


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answer_context"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "contextual_compression_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply citation-preserving contextual compression to fused evidence.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="component8d_contextual_compression")
    args = parser.parse_args()

    fused_bundle = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    result = attach_contextual_compression(fused_bundle, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "compression_applied": result.get("compression_context", {}).get("applied", False),
                "compression_model": result.get("compression_context", {}).get("compression_model"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
