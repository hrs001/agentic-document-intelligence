import argparse
import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.multi_source_routing import (
    MODEL_NAME,
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.transformed_query_bundle_orchestrator import build_transformed_query_bundle


DEFAULT_SQL_SCHEMA_PATH = "artifacts/experiments/component6_sql_schema_packaging_live/sql_schema/sql_schema_package.json"
VECTOR_PROFILES = ["skip", "fast", "balanced", "full"]

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sub_query": {"type": "string"},
        "active_sources": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "vector_profile": {"type": "string", "enum": VECTOR_PROFILES},
        "parallel_safe": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["sub_query", "active_sources", "vector_profile", "parallel_safe", "reasoning", "confidence"],
}

REFERENCE_PATTERN = re.compile(
    r"\b(it|they|them|their|that|those|same|former|latter|that segment|that product|that geography|fastest-growing segment)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You are a latency-optimized execution policy planner for a multi-source RAG system.
Return only valid JSON.

Goal:
- Minimize latency without breaking answer quality on real-world queries.
- The router has already chosen which sources are needed.
- Your task is only to decide the vector retrieval depth and whether multi-source execution is parallel-safe.

Vector profiles:
- skip: no vector route selected.
- fast: use the cheapest acceptable vector path for narrow factual/narrative lookups.
- balanced: use a medium-cost vector path for broader or moderately ambiguous narrative retrieval.
- full: use the most expensive vector path for difficult, multi-hop, under-specified, or high-risk narrative retrieval.

Policy guidance:
- Prefer fast when the vector need is narrow, direct, and citation-oriented.
- Prefer balanced when narrative context is needed but the question is still reasonably specific.
- Prefer full only when the vector portion is complex, ambiguous, multi-hop, or likely to fail with a shallow pass.
- Mark parallel_safe false if the sub-query appears dependent on prior results or the selected sources likely need sequential grounding.
- Mark parallel_safe true only when the selected sources can be executed independently for the same standalone sub-query.
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def extract_policy_signals(sub_query: str, selected_sources: list[str]) -> dict[str, Any]:
    normalized = sub_query.strip().lower()
    return {
        "word_count": len([part for part in normalized.split() if part]),
        "has_reference": bool(REFERENCE_PATTERN.search(sub_query)),
        "selected_source_count": len(selected_sources),
        "has_vector": "vector_document" in selected_sources,
        "has_sql": "sql_structured" in selected_sources,
        "has_graph": "graph_relationships" in selected_sources,
        "asks_for_explanation": any(term in normalized for term in ["explain", "why", "how does", "discuss", "strategy"]),
        "asks_for_quote_like_narrative": any(
            term in normalized for term in ["what did", "what does", "say about", "mentioned", "commentary"]
        ),
        "asks_for_executive_role": any(
            term in normalized
            for term in [
                " ceo",
                " cfo",
                " cto",
                " coo",
                "chief executive officer",
                "chief financial officer",
                "chief technology officer",
                "chief operating officer",
                "executive officer",
                "senior officer",
                "chairman",
                "chairperson",
                "president",
                "founder",
            ]
        ),
        "asks_segment_membership": "segment" in normalized and any(term in normalized for term in ["include", "included", "part of", "belongs"]),
        "is_multi_clause": len([part for part in re.split(r"\b(?:and|or)\b|[?;,]", normalized) if part.strip()]) > 1,
    }


def sanitize_policy_decision(
    sub_query: str,
    selected_sources: list[str],
    payload: dict[str, Any],
    signals: dict[str, Any],
) -> dict[str, Any]:
    active_sources = []
    for item in payload.get("active_sources", []):
        source = str(item).strip()
        if source in selected_sources and source not in active_sources:
            active_sources.append(source)
    if not active_sources:
        active_sources = list(selected_sources)

    vector_profile = str(payload.get("vector_profile", "skip")).strip()
    if vector_profile not in VECTOR_PROFILES:
        vector_profile = "skip"

    has_vector = "vector_document" in selected_sources
    if (
        "vector_document" in active_sources
        and len(active_sources) > 1
        and not signals["asks_for_explanation"]
        and not signals["asks_for_quote_like_narrative"]
        and not signals["asks_segment_membership"]
    ):
        active_sources = [source for source in active_sources if source != "vector_document"]
    if "narrative driver" in sub_query.lower() and "sql_structured" in active_sources:
        active_sources = [source for source in active_sources if source != "vector_document"]

    if not has_vector:
        vector_profile = "skip"
    elif signals["asks_for_executive_role"]:
        if "vector_document" not in active_sources:
            active_sources.append("vector_document")
        vector_profile = "balanced" if vector_profile in {"skip", "fast"} else vector_profile
    elif (signals["asks_for_explanation"] or signals["asks_for_quote_like_narrative"] or signals["asks_segment_membership"]) and "vector_document" not in active_sources:
        active_sources.append("vector_document")
    elif vector_profile == "skip":
        if signals["asks_for_explanation"] or signals["asks_for_quote_like_narrative"] or signals["asks_segment_membership"] or signals["is_multi_clause"]:
            vector_profile = "balanced"
        else:
            vector_profile = "fast"
    if signals["asks_for_quote_like_narrative"] and "vector_document" in active_sources and "sql_structured" in active_sources:
        vector_profile = "balanced" if vector_profile == "fast" else vector_profile
    if "narrative driver" in sub_query.lower() and "vector_document" in active_sources:
        vector_profile = "full"

    parallel_safe = bool(payload.get("parallel_safe", False))
    if signals["has_reference"]:
        parallel_safe = False
    if len(selected_sources) <= 1:
        parallel_safe = False
    if (
        len(active_sources) > 1
        and set(active_sources).issubset({"sql_structured", "graph_relationships"})
        and not signals["has_reference"]
        and not signals["asks_for_quote_like_narrative"]
    ):
        parallel_safe = True
    if (
        len(active_sources) > 1
        and set(active_sources).issubset({"vector_document", "graph_relationships"})
        and not signals["has_reference"]
    ):
        parallel_safe = True

    return {
        "sub_query": sub_query,
        "selected_sources": selected_sources,
        "active_sources": active_sources,
        "vector_profile": vector_profile,
        "parallel_safe": parallel_safe,
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
        "signals": signals,
    }


def plan_sub_query_execution(
    sub_query: str,
    selected_sources: list[str],
    model: str = MODEL_NAME,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    signals = extract_policy_signals(sub_query, selected_sources)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "latency_optimized_orchestration_policy",
                "schema": PLAN_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sub_query": sub_query,
                        "selected_sources": selected_sources,
                        "signals": signals,
                    }
                ),
            },
        ],
    )
    payload = json.loads(completion.choices[0].message.content or "{}")
    return sanitize_policy_decision(sub_query, selected_sources, payload, signals)


def build_latency_optimized_policy(
    query: str,
    transformed_bundle: dict[str, Any],
    routing_plan: dict[str, Any],
    model: str = MODEL_NAME,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    routing_sub_queries = list(routing_plan["sub_query_plans"])

    def plan_for_sub_query(sub_query_plan: dict[str, Any]) -> dict[str, Any]:
        selected_sources = list(sub_query_plan["routing_decision"]["selected_sources"])
        policy = plan_sub_query_execution(
            sub_query_plan["original_sub_query"],
            selected_sources,
            model=model,
            client=client,
        )
        return {
            "sub_query_id": sub_query_plan["sub_query_id"],
            "original_sub_query": sub_query_plan["original_sub_query"],
            "routing_decision": sub_query_plan["routing_decision"],
            "execution_policy": policy,
        }

    if len(routing_sub_queries) <= 1:
        sub_query_execution_plans = [plan_for_sub_query(item) for item in routing_sub_queries]
    else:
        max_workers = min(4, len(routing_sub_queries))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            sub_query_execution_plans = list(executor.map(plan_for_sub_query, routing_sub_queries))

    profile_counts = {profile: 0 for profile in VECTOR_PROFILES}
    parallel_safe_count = 0
    for plan in sub_query_execution_plans:
        profile_counts[plan["execution_policy"]["vector_profile"]] += 1
        if plan["execution_policy"]["parallel_safe"]:
            parallel_safe_count += 1

    return {
        "original_query": query,
        "policy": {
            "policy_model": model,
            "policy_method": "llm_latency_optimized_execution_planner_v1",
        },
        "sub_query_execution_plans": sub_query_execution_plans,
        "policy_summary": {
            "sub_query_count": len(sub_query_execution_plans),
            "vector_profile_counts": profile_counts,
            "parallel_safe_count": parallel_safe_count,
        },
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "orchestration"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "latency_optimized_policy_plan.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a latency-optimized orchestration policy plan.")
    parser.add_argument(
        "--query",
        default="Who is the CEO of Microsoft and which segment had the highest revenue in FY2025?",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--sql-schema-path", default=DEFAULT_SQL_SCHEMA_PATH)
    parser.add_argument("--run-id", default="component7_latency_optimized_policy")
    args = parser.parse_args()

    transformed_bundle = build_transformed_query_bundle(args.query, model=args.model)
    schema_package = load_json_result(PROJECT_ROOT / "agentic_document_intelligence" / args.sql_schema_path)
    sql_capability_summary = build_sql_capability_summary(schema_package)
    graph_capability_summary = build_graph_capability_summary()
    routing_plan = build_multi_source_routing_plan(
        transformed_bundle,
        sql_capability_summary,
        graph_capability_summary,
        model=args.model,
    )
    result = build_latency_optimized_policy(args.query, transformed_bundle, routing_plan, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "policy_summary": result["policy_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
