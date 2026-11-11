import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.corrective_hyde_retry import run_corrective_hyde_retry
from agentic_document_intelligence.scripts.final_evidence_bundle_assembly import assemble_final_evidence_bundle
from agentic_document_intelligence.scripts.graph_retrieval import retrieve_graph_evidence
from agentic_document_intelligence.scripts.mmr_diversification import (
    diversify_sub_query_results,
    load_embedding_records,
)
from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.package_graph_evidence import assemble_graph_evidence_bundle
from agentic_document_intelligence.scripts.package_sql_evidence import package_sql_evidence
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import (
    build_chunk_indexes,
    load_chunk_artifact,
)
from agentic_document_intelligence.scripts.rerank_sub_query_candidates import rerank_sub_query_results
from agentic_document_intelligence.scripts.retrieval_merge_dedup import merge_variant_results
from agentic_document_intelligence.scripts.sub_query_coverage_scoring import score_coverage
from agentic_document_intelligence.scripts.transformed_query_bundle_orchestrator import (
    MODEL_NAME,
    build_transformed_query_bundle,
)
from agentic_document_intelligence.scripts.transformed_retrieval_executor import (
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_INPUT,
    DEFAULT_NAMESPACE,
    DEFAULT_TOP_K,
    execute_transformed_retrieval,
)
from agentic_document_intelligence.scripts.validate_and_execute_sql import execute_read_only_sql
from agentic_document_intelligence.scripts.generate_text_to_sql import generate_text_to_sql


DEFAULT_SQL_SCHEMA_PATH = "artifacts/experiments/component6_sql_schema_packaging_live/sql_schema/sql_schema_package.json"
DEFAULT_SQL_DATABASE_PATH = (
    "artifacts/experiments/component6_sqlite_database_build_live/sql_db/microsoft_fy2025_analyst_demo.sqlite"
)
DEFAULT_GRAPH_DATABASE_PATH = (
    "artifacts/experiments/component5_kuzu_graph_build_live/kuzu_db/microsoft_fy2025_10k_summary.kuzu"
)
DEFAULT_EMBEDDING_INPUT = (
    "artifacts/experiments/component2_embedding_ready_records/embeddings/"
    "microsoft_fy2025_10k_summary_embedding_records.json"
)
DEFAULT_VECTOR_RERANK_TOP_N = 6
DEFAULT_VECTOR_MMR_TOP_M = 4
DEFAULT_VECTOR_MMR_LAMBDA = 0.75
REFERENCE_PATTERN = re.compile(
    r"\b(it|they|them|their|that|those|same|former|latter|fastest-growing segment|that segment|that product|that geography|the segment that)\b",
    re.IGNORECASE,
)
RESOLUTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resolved_sub_query": {"type": "string"},
        "used_context": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["resolved_sub_query", "used_context", "reasoning"],
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def build_single_sub_query_bundle(transformed_bundle: dict[str, Any], sub_query_id: str) -> dict[str, Any]:
    matching = [item for item in transformed_bundle["sub_query_bundles"] if item["sub_query_id"] == sub_query_id]
    if not matching:
        raise ValueError(f"Sub-query bundle not found for {sub_query_id}")
    return {
        "original_query": transformed_bundle["original_query"],
        "policy": transformed_bundle["policy"],
        "decomposition_result": transformed_bundle["decomposition_result"],
        "sub_query_bundles": matching,
        "bundle_summary": {
            "sub_query_count": 1,
            "total_multi_query_rewrites": matching[0]["multi_query_result"]["rewrite_count"],
            "step_back_count": 1,
            "hyde_candidate_count": int(matching[0]["hyde_recommendation"]["should_consider_hyde_after_weak_retrieval"]),
        },
    }


def build_single_query_execution_bundle(query: str, model: str) -> dict[str, Any]:
    rebuilt = build_transformed_query_bundle(query, model=model)
    return build_single_sub_query_bundle(rebuilt, rebuilt["sub_query_bundles"][0]["sub_query_id"])


def should_resolve_with_context(sub_query: str) -> bool:
    return bool(REFERENCE_PATTERN.search(sub_query))


def summarize_source_output(source_output: dict[str, Any]) -> str:
    source = source_output["source"]
    bundle = source_output.get("evidence_bundle", {})
    if source == "sql_structured":
        preview = bundle.get("preview_rows", [])[:3]
        return f"SQL preview rows: {json.dumps(preview)}"
    if source == "graph_relationships":
        node_summaries = []
        for node in bundle.get("matched_nodes", [])[:5]:
            snippets = [str(item).strip() for item in node.get("evidence_snippets", [])[:1] if str(item).strip()]
            section_titles = [str(item).strip() for item in node.get("section_titles", [])[:1] if str(item).strip()]
            parts = [node.get("canonical_name", "")]
            if section_titles:
                parts.append(f"section={section_titles[0]}")
            if snippets:
                parts.append(f"evidence={snippets[0]}")
            node_summaries.append(" | ".join(part for part in parts if part))
        edges = [
            f"{edge.get('source_canonical_name', '')} -> {edge.get('relation_type', '')} -> {edge.get('target_canonical_name', '')}"
            for edge in bundle.get("matched_edges", [])[:5]
        ]
        return f"Graph nodes: {node_summaries}; Graph edges: {edges}"
    return (bundle.get("assembled_evidence_text", "") or "")[:800]


def build_prior_context_summary(sub_query_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for item in sub_query_results:
        summaries.append(
            {
                "sub_query": item["original_sub_query"],
                "source_summaries": [summarize_source_output(source_output) for source_output in item["source_outputs"]],
            }
        )
    return summaries


def resolve_sub_query_with_context(
    sub_query: str,
    prior_sub_query_results: list[dict[str, Any]],
    model: str,
    openai_client: OpenAI,
) -> dict[str, Any]:
    if not prior_sub_query_results or not should_resolve_with_context(sub_query):
        return {
            "resolved_sub_query": sub_query,
            "used_context": False,
            "reasoning": "No prior context resolution needed.",
        }

    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "context_resolution",
                "schema": RESOLUTION_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the sub-query into a standalone question using only facts supported by prior evidence summaries. "
                    "Do not invent facts. If the query is already standalone, return it unchanged."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sub_query": sub_query,
                        "prior_context": build_prior_context_summary(prior_sub_query_results),
                    }
                ),
            },
        ],
    )
    payload = json.loads(completion.choices[0].message.content or "{}")
    resolved = str(payload.get("resolved_sub_query", "")).strip() or sub_query
    return {
        "resolved_sub_query": resolved,
        "used_context": bool(payload.get("used_context", False)),
        "reasoning": str(payload.get("reasoning", "")).strip(),
    }


def execute_vector_source(
    mini_bundle: dict[str, Any],
    index: Any,
    namespace: str,
    alpha: float,
    top_k: int,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
    child_index: dict[str, dict[str, Any]],
    parent_index: dict[str, str],
    chunk_to_record_id: dict[str, str],
    rerank_top_n: int,
    mmr_top_m: int,
    mmr_lambda: float,
) -> dict[str, Any]:
    retrieval = execute_transformed_retrieval(
        mini_bundle,
        index,
        namespace,
        alpha,
        top_k,
        openai_client,
        pinecone_client,
        child_index,
        parent_index,
    )
    merged = merge_variant_results(retrieval)
    coverage = score_coverage(merged)
    corrected = run_corrective_hyde_retry(
        coverage,
        merged,
        index,
        namespace,
        alpha,
        top_k,
        openai_client,
        pinecone_client,
        child_index,
        parent_index,
    )
    reranked = rerank_sub_query_results(corrected, pinecone_client, rerank_top_n)
    diversified = diversify_sub_query_results(
        reranked,
        index,
        namespace,
        chunk_to_record_id,
        top_m=mmr_top_m,
        lambda_weight=mmr_lambda,
    )
    evidence_bundle = assemble_final_evidence_bundle(diversified)
    return {
        "source": "vector_document",
        "coverage_summary": coverage["coverage_summary"],
        "retry_summary": corrected["retry_summary"],
        "bundle_summary": evidence_bundle["bundle_summary"],
        "evidence_bundle": evidence_bundle,
    }


def execute_graph_source(database_path: Path, query: str, top_node_k: int = 5, top_edge_k: int = 8) -> dict[str, Any]:
    retrieval = retrieve_graph_evidence(database_path, query, top_node_k=top_node_k, top_edge_k=top_edge_k)
    evidence_bundle = assemble_graph_evidence_bundle(
        {
            "database_path": str(database_path),
            **retrieval,
        }
    )
    return {
        "source": "graph_relationships",
        "bundle_summary": evidence_bundle["bundle_summary"],
        "evidence_bundle": evidence_bundle,
    }


def execute_sql_source(
    query: str,
    schema_package: dict[str, Any],
    database_path: Path,
    model: str,
    openai_client: OpenAI,
) -> dict[str, Any]:
    generation = generate_text_to_sql(query, schema_package, model=model, client=openai_client)
    execution = execute_read_only_sql(database_path, generation["sql_query"], row_limit=20)
    report = {
        "database_path": str(database_path),
        "user_query": generation["user_query"],
        "target_tables": generation["target_tables"],
        "generated_sql": generation["sql_query"],
        "validated_sql": execution["validated_sql"],
        "row_limit": execution["row_limit"],
        "row_count": execution["row_count"],
        "columns": execution["columns"],
        "rows": execution["rows"],
        "confidence": generation["confidence"],
    }
    evidence_bundle = package_sql_evidence(report, preview_limit=10)
    return {
        "source": "sql_structured",
        "bundle_summary": evidence_bundle["bundle_summary"],
        "generated_sql": generation["sql_query"],
        "executed_query": query,
        "evidence_bundle": evidence_bundle,
    }


def execute_routed_orchestration(
    query: str,
    transformed_bundle: dict[str, Any],
    routing_plan: dict[str, Any],
    schema_package: dict[str, Any],
    sql_database_path: Path,
    graph_database_path: Path,
    index: Any,
    namespace: str,
    alpha: float,
    top_k: int,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
    child_index: dict[str, dict[str, Any]],
    parent_index: dict[str, str],
    chunk_to_record_id: dict[str, str],
    model: str,
    rerank_top_n: int,
    mmr_top_m: int,
    mmr_lambda: float,
    graph_executor: Callable[..., dict[str, Any]] | None = None,
    sql_executor: Callable[..., dict[str, Any]] | None = None,
    vector_executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    graph_runner = graph_executor or execute_graph_source
    sql_runner = sql_executor or execute_sql_source
    vector_runner = vector_executor or execute_vector_source

    sub_query_results = []
    source_usage = {"vector_document": 0, "graph_relationships": 0, "sql_structured": 0}

    for sub_query_plan in routing_plan["sub_query_plans"]:
        sub_query_id = sub_query_plan["sub_query_id"]
        sub_query = sub_query_plan["original_sub_query"]
        selected_sources = sub_query_plan["routing_decision"]["selected_sources"]
        resolution = resolve_sub_query_with_context(sub_query, sub_query_results, model, openai_client)
        effective_sub_query = resolution["resolved_sub_query"]
        source_outputs = []

        for source in selected_sources:
            source_usage[source] += 1
            if source == "vector_document":
                if effective_sub_query != sub_query:
                    mini_bundle = build_single_query_execution_bundle(effective_sub_query, model)
                else:
                    mini_bundle = build_single_sub_query_bundle(transformed_bundle, sub_query_id)
                source_outputs.append(
                    vector_runner(
                        mini_bundle,
                        index,
                        namespace,
                        alpha,
                        top_k,
                        openai_client,
                        pinecone_client,
                        child_index,
                        parent_index,
                        chunk_to_record_id,
                        rerank_top_n,
                        mmr_top_m,
                        mmr_lambda,
                    )
                )
            elif source == "graph_relationships":
                graph_output = graph_runner(graph_database_path, effective_sub_query)
                graph_output["executed_query"] = effective_sub_query
                source_outputs.append(graph_output)
            elif source == "sql_structured":
                source_outputs.append(
                    sql_runner(effective_sub_query, schema_package, sql_database_path, model, openai_client)
                )

        sub_query_results.append(
            {
                "sub_query_id": sub_query_id,
                "original_sub_query": sub_query,
                "resolved_sub_query": effective_sub_query,
                "context_resolution": resolution,
                "routing_decision": sub_query_plan["routing_decision"],
                "source_outputs": source_outputs,
            }
        )

    return {
        "original_query": query,
        "policy": {
            "routing_policy": routing_plan["policy"],
            "vector_execution": {
                "namespace": namespace,
                "alpha": alpha,
                "top_k": top_k,
                "rerank_top_n": rerank_top_n,
                "mmr_top_m": mmr_top_m,
                "mmr_lambda": mmr_lambda,
            },
        },
        "routing_summary": routing_plan["routing_summary"],
        "execution_summary": {
            "sub_query_count": len(sub_query_results),
            "source_usage": source_usage,
        },
        "sub_query_results": sub_query_results,
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
    report_path = output_dir / "multi_source_orchestration_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the routed multi-source orchestration plan.")
    parser.add_argument(
        "--query",
        default="Who is the CEO of Microsoft and which segment had the highest revenue in FY2025?",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--embedding-input", default=DEFAULT_EMBEDDING_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--rerank-top-n", type=int, default=DEFAULT_VECTOR_RERANK_TOP_N)
    parser.add_argument("--mmr-top-m", type=int, default=DEFAULT_VECTOR_MMR_TOP_M)
    parser.add_argument("--mmr-lambda", type=float, default=DEFAULT_VECTOR_MMR_LAMBDA)
    parser.add_argument("--sql-schema-path", default=DEFAULT_SQL_SCHEMA_PATH)
    parser.add_argument("--sql-database-path", default=DEFAULT_SQL_DATABASE_PATH)
    parser.add_argument("--graph-database-path", default=DEFAULT_GRAPH_DATABASE_PATH)
    parser.add_argument("--run-id", default="component7_multi_source_orchestration")
    args = parser.parse_args()

    load_project_env()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

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

    chunk_artifact = load_chunk_artifact(PROJECT_ROOT / "agentic_document_intelligence" / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)
    chunk_to_record_id = load_embedding_records(PROJECT_ROOT / "agentic_document_intelligence" / args.embedding_input)
    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index(index_name)

    result = execute_routed_orchestration(
        query=args.query,
        transformed_bundle=transformed_bundle,
        routing_plan=routing_plan,
        schema_package=schema_package,
        sql_database_path=PROJECT_ROOT / "agentic_document_intelligence" / args.sql_database_path,
        graph_database_path=PROJECT_ROOT / "agentic_document_intelligence" / args.graph_database_path,
        index=index,
        namespace=args.namespace,
        alpha=args.alpha,
        top_k=args.top_k,
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        child_index=child_index,
        parent_index=parent_index,
        chunk_to_record_id=chunk_to_record_id,
        model=args.model,
        rerank_top_n=args.rerank_top_n,
        mmr_top_m=args.mmr_top_m,
        mmr_lambda=args.mmr_lambda,
    )
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": result["execution_summary"]["sub_query_count"],
                "source_usage": result["execution_summary"]["source_usage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
