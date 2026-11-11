import argparse
import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.transformed_query_bundle_orchestrator import (
    MODEL_NAME,
    build_transformed_query_bundle,
)


SUPPORTED_SOURCES = ["vector_document", "graph_relationships", "sql_structured"]

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sub_query": {"type": "string"},
        "primary_source": {"type": "string", "enum": SUPPORTED_SOURCES},
        "selected_sources": {
            "type": "array",
            "items": {"type": "string", "enum": SUPPORTED_SOURCES},
            "minItems": 1,
            "maxItems": 3,
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["sub_query", "primary_source", "selected_sources", "reasoning", "confidence"],
}

SYSTEM_PROMPT = """You are a multi-source router for a fixed Microsoft FY2025 demo corpus.
Return only valid JSON.

Your job:
- Choose which retrieval source or sources should answer one sub-query.
- Available sources are:
  1. vector_document: document RAG over Microsoft FY2025 10-K style text chunks. Best for narrative disclosures, management commentary, risk factors, strategy, qualitative explanations, quotations, and general filing facts.
  2. graph_relationships: GraphRAG over entities and typed relationships. Best for questions about relationships among segments, products, technologies, organizations, geographies, and strategic links.
  3. sql_structured: SQLite financial tables. Best for precise numeric lookups, ranking, aggregation, comparisons, percentages, growth, margins, and tabular filtering where the schema supports it.

Routing rules:
- Use sql_structured when the question is fundamentally asking for a metric, ranking, aggregation, comparison, or filtered table answer supported by the available SQL schema.
- Use graph_relationships when the question is fundamentally about entity-to-entity relationships, membership, inclusion, dependencies, organizational links, or product/segment structure.
- Use vector_document when the question needs narrative explanation, qualitative evidence, management language, risk discussion, strategy, or facts not stored in SQL tables.
- Use multiple sources when the question truly spans multiple evidence types.
- Do not choose sql_structured for facts that are not present in the SQL schema.
- Do not choose graph_relationships for pure numeric aggregation unless relationship context is central.
- Prefer the smallest useful source set, but do not omit a source that is clearly needed.
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def load_json_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["result"]


def build_sql_capability_summary(schema_package: dict[str, Any]) -> dict[str, Any]:
    tables = []
    for table in schema_package["tables"]:
        tables.append(
            {
                "table_name": table["table_name"],
                "columns": [column["name"] for column in table["columns"]],
            }
        )
    return {
        "database_path": schema_package["database_path"],
        "table_count": schema_package["table_count"],
        "tables": tables,
    }


def build_graph_capability_summary() -> dict[str, Any]:
    return {
        "entity_types": [
            "organization",
            "segment",
            "product_or_service",
            "technology_or_platform",
            "regulatory_or_external_body",
            "geography",
            "initiative_or_strategy",
            "risk_or_issue",
            "financial_or_business_concept",
        ],
        "relation_types": [
            "includes",
            "integrates",
            "invests_in",
            "supports",
            "faces",
            "uses",
            "positions_as",
            "depends_on",
        ],
        "best_for": [
            "product to segment relationships",
            "technology and strategy relationships",
            "organizational and business structure questions",
            "relationship-centric fact retrieval",
        ],
    }


def extract_routing_signals(query: str) -> dict[str, Any]:
    normalized = query.strip().lower()
    executive_role_patterns = [
        r"\bceo\b",
        r"\bcfo\b",
        r"\bcto\b",
        r"\bcoo\b",
        r"\bchief executive officer\b",
        r"\bchief financial officer\b",
        r"\bchief technology officer\b",
        r"\bchief operating officer\b",
        r"\bexecutive officer\b",
        r"\bsenior officer\b",
        r"\bpresident\b",
        r"\bchair(man|woman|person)?\b",
        r"\bfounder\b",
    ]
    numeric_patterns = [
        r"\bhighest\b",
        r"\blowest\b",
        r"\baverage\b",
        r"\bavg\b",
        r"\btotal\b",
        r"\bsum\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\brank\b",
        r"\bordered\b",
        r"\btop\b",
        r"\bpercentage\b",
        r"\bpct\b",
        r"\bgrowth\b",
        r"\brevenue\b",
        r"\bmargin\b",
        r"\bincome\b",
        r"\bhow many\b",
        r"\bwhat was\b",
    ]
    relationship_patterns = [
        r"\binclude[s]?\b",
        r"\bincluded under\b",
        r"\bpart of\b",
        r"\bbelong[s]?\b",
        r"\brelate to\b",
        r"\brelationship\b",
        r"\bsupport[s]?\b",
        r"\bdepend[s]?\b",
        r"\buse[s]?\b",
        r"\bintegrate[s]?\b",
        r"\bwhich segment\b.*\binclude",
    ]
    narrative_patterns = [
        r"\bhow does\b",
        r"\bwhy\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bdiscuss\b",
        r"\bwhat did\b",
        r"\bwhat does\b",
        r"\bhow did management\b",
        r"\brisk\b",
        r"\bstrategy\b",
        r"\boutlook\b",
        r"\bcommentary\b",
        r"\bbusiness structure\b",
        r"\bnarrative driver\b",
        r"\bmentioned\b",
        r"\bproducts and services\b",
        r"\bsay about\b",
    ]
    metric_term_patterns = [
        r"\brevenue\b",
        r"\bgrowth\b",
        r"\bmargin\b",
        r"\bincome\b",
        r"\bpercentage\b",
        r"\bscore\b",
        r"\bmix\b",
    ]

    numeric_hits = sum(bool(re.search(pattern, normalized)) for pattern in numeric_patterns)
    relationship_hits = sum(bool(re.search(pattern, normalized)) for pattern in relationship_patterns)
    narrative_hits = sum(bool(re.search(pattern, normalized)) for pattern in narrative_patterns)
    metric_hits = sum(bool(re.search(pattern, normalized)) for pattern in metric_term_patterns)
    executive_role_hits = sum(bool(re.search(pattern, normalized)) for pattern in executive_role_patterns)
    clause_count = len([part for part in re.split(r"\b(?:and|or)\b|[?;,]", normalized) if part.strip()])

    return {
        "word_count": len([part for part in normalized.split() if part]),
        "numeric_signal_count": numeric_hits,
        "relationship_signal_count": relationship_hits,
        "narrative_signal_count": narrative_hits,
        "metric_signal_count": metric_hits,
        "executive_role_signal_count": executive_role_hits,
        "clause_count": clause_count,
        "segment_membership_signal": bool(re.search(r"\b(segment|reporting segment)\b", normalized)) and relationship_hits > 0,
        "executive_role_signal": executive_role_hits > 0,
        "looks_multi_source": sum([numeric_hits > 0, relationship_hits > 0, narrative_hits > 0]) >= 2,
    }


def sanitize_routing_decision(sub_query: str, payload: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    selected_sources = []
    for item in payload.get("selected_sources", []):
        source = str(item).strip()
        if source in SUPPORTED_SOURCES and source not in selected_sources:
            selected_sources.append(source)

    primary_source = str(payload.get("primary_source", "")).strip()
    if primary_source not in SUPPORTED_SOURCES:
        primary_source = ""

    if signals["executive_role_signal"]:
        primary_source = "vector_document"
        selected_sources = [source for source in selected_sources if source != "graph_relationships"]
        if "vector_document" not in selected_sources:
            selected_sources.insert(0, "vector_document")

    if "graph_relationships" in selected_sources and "vector_document" not in selected_sources:
        selected_sources.append("vector_document")

    if signals["narrative_signal_count"] > 0 and "vector_document" not in selected_sources:
        selected_sources.append("vector_document")
    if signals["segment_membership_signal"] and "vector_document" not in selected_sources:
        selected_sources.append("vector_document")
    if signals["relationship_signal_count"] > 0 and signals["narrative_signal_count"] > 0:
        if "vector_document" not in selected_sources:
            selected_sources.append("vector_document")
    if signals["looks_multi_source"] and primary_source == "sql_structured" and "vector_document" not in selected_sources:
        selected_sources.append("vector_document")
    if not selected_sources and primary_source:
        selected_sources.append(primary_source)
    if not selected_sources:
        selected_sources = ["vector_document"]

    if primary_source not in selected_sources:
        primary_source = selected_sources[0]

    ordered_sources = [primary_source] + [source for source in selected_sources if source != primary_source]
    return {
        "sub_query": sub_query,
        "primary_source": primary_source,
        "selected_sources": ordered_sources[:3],
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
    }


def route_sub_query(
    sub_query: str,
    sql_capability_summary: dict[str, Any],
    graph_capability_summary: dict[str, Any],
    model: str = MODEL_NAME,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    signals = extract_routing_signals(sub_query)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "multi_source_routing",
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
                        "sub_query": sub_query,
                        "signals": signals,
                        "sql_capability_summary": sql_capability_summary,
                        "graph_capability_summary": graph_capability_summary,
                    }
                ),
            },
        ],
    )

    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    decision = sanitize_routing_decision(sub_query, payload, signals)
    decision["signals"] = signals
    return decision


def build_multi_source_routing_plan(
    transformed_bundle: dict[str, Any],
    sql_capability_summary: dict[str, Any],
    graph_capability_summary: dict[str, Any],
    model: str = MODEL_NAME,
    route_fn: Callable[[str, dict[str, Any], dict[str, Any], str], dict[str, Any]] | None = None,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    router = route_fn or route_sub_query
    bundles = list(transformed_bundle["sub_query_bundles"])

    def run_route(bundle: dict[str, Any]) -> dict[str, Any]:
        try:
            route = router(
                bundle["original_sub_query"],
                sql_capability_summary,
                graph_capability_summary,
                model,
                client,
            )
        except TypeError:
            route = router(
                bundle["original_sub_query"],
                sql_capability_summary,
                graph_capability_summary,
                model,
            )
        return {
            "sub_query_id": bundle["sub_query_id"],
            "original_sub_query": bundle["original_sub_query"],
            "routing_decision": route,
        }

    if len(bundles) <= 1:
        sub_query_plans = [run_route(bundle) for bundle in bundles]
    else:
        max_workers = min(4, len(bundles))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            sub_query_plans = list(executor.map(run_route, bundles))

    source_usage = {source: 0 for source in SUPPORTED_SOURCES}
    for item in sub_query_plans:
        for source in item["routing_decision"]["selected_sources"]:
            source_usage[source] += 1

    return {
        "original_query": transformed_bundle["original_query"],
        "policy": {
            "routing_model": model,
            "routing_method": "llm_with_capability_summaries_and_signal_validation_v1",
        },
        "sub_query_plans": sub_query_plans,
        "routing_summary": {
            "sub_query_count": len(sub_query_plans),
            "source_usage": source_usage,
        },
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "routing"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "multi_source_routing_plan.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-source routing plan across vector, graph, and SQL.")
    parser.add_argument(
        "--query",
        default="Who is the CEO of Microsoft and which segment had the highest revenue in FY2025?",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--sql-schema-path",
        default="artifacts/experiments/component6_sql_schema_packaging_live/sql_schema/sql_schema_package.json",
    )
    parser.add_argument("--run-id", default="component7_multi_source_routing")
    args = parser.parse_args()

    transformed_bundle = build_transformed_query_bundle(args.query, model=args.model)
    schema_package = load_json_result(PROJECT_ROOT / "agentic_document_intelligence" / args.sql_schema_path)
    sql_capability_summary = build_sql_capability_summary(schema_package)
    graph_capability_summary = build_graph_capability_summary()
    result = build_multi_source_routing_plan(
        transformed_bundle,
        sql_capability_summary,
        graph_capability_summary,
        model=args.model,
    )
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": result["routing_summary"]["sub_query_count"],
                "source_usage": result["routing_summary"]["source_usage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
