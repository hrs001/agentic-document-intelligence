import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.execute_multi_source_orchestration import (
    DEFAULT_EMBEDDING_INPUT,
    DEFAULT_GRAPH_DATABASE_PATH,
    DEFAULT_SQL_DATABASE_PATH,
    DEFAULT_SQL_SCHEMA_PATH,
    build_single_query_execution_bundle,
    execute_graph_source,
    execute_sql_source,
    execute_vector_source,
    load_project_env,
    resolve_sub_query_with_context,
)
from agentic_document_intelligence.scripts.llm_query_decomposition import decompose_query_with_llm
from agentic_document_intelligence.scripts.final_evidence_bundle_assembly import build_citation
from agentic_document_intelligence.scripts.latency_optimized_orchestration_policy import (
    MODEL_NAME,
    build_latency_optimized_policy,
)
from agentic_document_intelligence.scripts.mmr_diversification import load_embedding_records
from agentic_document_intelligence.scripts.multi_query_generation import generate_multi_queries
from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import build_chunk_indexes, load_chunk_artifact
from agentic_document_intelligence.scripts.retrieval_merge_dedup import merge_variant_results
from agentic_document_intelligence.scripts.sub_query_coverage_scoring import score_coverage
from agentic_document_intelligence.scripts.transformed_retrieval_executor import (
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_INPUT,
    DEFAULT_NAMESPACE,
    DEFAULT_TOP_K,
    execute_transformed_retrieval,
    execute_variant_retrieval,
)


DEFAULT_VECTOR_FAST_TOP_K = 4
DEFAULT_VECTOR_FAST_EVIDENCE = 3
MAX_SUB_QUERIES = 3


def _cap_sub_queries(sub_queries: list[str]) -> list[str]:
    seen = set()
    capped = []
    for item in sub_queries:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        capped.append(cleaned)
        if len(capped) >= MAX_SUB_QUERIES:
            break
    return capped


def build_minimal_orchestration_bundle(query: str, model: str, openai_client: OpenAI | None = None) -> dict[str, Any]:
    decomposition_result = decompose_query_with_llm(query, model=model, client=openai_client)
    raw_sub_queries = decomposition_result.get("sub_queries", [query])
    sub_queries = _cap_sub_queries(raw_sub_queries)
    truncated_subqueries = len([item for item in raw_sub_queries if str(item).strip()]) > len(sub_queries)
    sub_query_bundles = [
        {
            "sub_query_id": f"sq_{index}",
            "original_sub_query": sub_query,
        }
        for index, sub_query in enumerate(sub_queries, start=1)
    ]
    return {
        "original_query": query,
        "policy": {"decomposition_model": model, "minimal_bundle": True},
        "decomposition_result": {
            **decomposition_result,
            "sub_queries": sub_queries,
            "raw_sub_query_count": len([item for item in raw_sub_queries if str(item).strip()]),
            "truncated_subqueries": truncated_subqueries,
            "subquery_cap": MAX_SUB_QUERIES,
        },
        "sub_query_bundles": sub_query_bundles,
        "bundle_summary": {"sub_query_count": len(sub_query_bundles)},
    }


def assemble_fast_vector_evidence(merged: dict[str, Any], top_n: int = DEFAULT_VECTOR_FAST_EVIDENCE) -> dict[str, Any]:
    sub_query_bundles = []
    total_evidence_items = 0
    context_blocks: list[str] = []
    for sub_query_result in merged["sub_query_results"]:
        evidence_items = []
        for index, match in enumerate(sub_query_result["merged_matches"][:top_n], start=1):
            citation = build_citation(match)
            evidence_items.append(
                {
                    "source_chunk_id": match["source_chunk_id"],
                    "parent_id": citation["parent_id"],
                    "child_text": match.get("child_text", ""),
                    "parent_text": match.get("parent_text", ""),
                    "citation": citation,
                    "best_score": match.get("best_score", 0.0),
                    "match_count": match.get("match_count", 0),
                    "variant_types": match.get("variant_types", []),
                    "query_angles": match.get("query_angles", []),
                }
            )
            context_blocks.append(
                "\n".join(
                    [
                        f"[Fast Evidence {index}] Sub-query: {sub_query_result['original_sub_query']}",
                        f"Section: {citation['section_title']}",
                        f"Pages: {citation['page']} - {citation['page_end']}",
                        f"Parent ID: {citation['parent_id']}",
                        f"Best Score: {match.get('best_score', 0.0)}",
                        "Child Evidence:",
                        match.get("child_text", ""),
                        "Parent Context:",
                        match.get("parent_text", ""),
                    ]
                )
            )
        total_evidence_items += len(evidence_items)
        sub_query_bundles.append(
            {
                "sub_query_id": sub_query_result["sub_query_id"],
                "original_sub_query": sub_query_result["original_sub_query"],
                "evidence_count": len(evidence_items),
                "evidence_items": evidence_items,
            }
        )

    return {
        "original_query": merged["original_query"],
        "policy": merged["policy"],
        "bundle_summary": {
            "sub_query_count": len(sub_query_bundles),
            "total_evidence_items": total_evidence_items,
        },
        "sub_query_bundles": sub_query_bundles,
        "assembled_evidence_text": "\n\n---\n\n".join(context_blocks),
    }


def execute_vector_source_optimized(
    query: str,
    vector_profile: str,
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
) -> dict[str, Any]:
    if vector_profile == "full":
        full_bundle = build_single_query_execution_bundle(query, model)
        full_result = execute_vector_source(
            full_bundle,
            index,
            namespace,
            alpha,
            top_k,
            openai_client,
            pinecone_client,
            child_index,
            parent_index,
            chunk_to_record_id,
            rerank_top_n=6,
            mmr_top_m=4,
            mmr_lambda=0.75,
        )
        full_result["execution_profile"] = "full"
        full_result["escalated_to_full"] = False
        return full_result

    variants = [
        {
            "sub_query_id": "sq_1",
            "variant_id": "sq_1_original",
            "variant_type": "original_sub_query",
            "query_text": query,
            "query_angle": "original",
        }
    ]
    if vector_profile == "balanced":
        multi_query_result = generate_multi_queries(query, model=model, client=openai_client)
        for rewrite_index, rewrite in enumerate(multi_query_result["rewrites"], start=1):
            variants.append(
                {
                    "sub_query_id": "sq_1",
                    "variant_id": f"sq_1_mq_{rewrite_index}",
                    "variant_type": "multi_query",
                    "query_text": rewrite["query"],
                    "query_angle": rewrite["angle"],
                }
            )

    variant_results = [
        execute_variant_retrieval(
            variant,
            index,
            namespace,
            alpha,
            DEFAULT_VECTOR_FAST_TOP_K if vector_profile == "fast" else top_k,
            openai_client,
            pinecone_client,
            child_index,
            parent_index,
        )
        for variant in variants
    ]
    retrieval = {
        "original_query": query,
        "policy": {"vector_profile": vector_profile},
        "retrieval_summary": {
            "sub_query_count": 1,
            "variant_count": len(variants),
            "retrieval_call_count": len(variants),
            "total_deduped_matches": sum(item["deduped_match_count"] for item in variant_results),
        },
        "variant_results": variant_results,
    }
    merged = merge_variant_results(retrieval)
    coverage = score_coverage(merged)
    average_score = coverage["coverage_summary"]["average_coverage_score"]
    weak_count = coverage["coverage_summary"]["weak_sub_query_count"]

    if vector_profile == "fast" and average_score >= 0.62 and weak_count == 0:
        evidence_bundle = assemble_fast_vector_evidence(merged, top_n=3)
        return {
            "source": "vector_document",
            "execution_profile": "fast",
            "coverage_summary": coverage["coverage_summary"],
            "bundle_summary": evidence_bundle["bundle_summary"],
            "evidence_bundle": evidence_bundle,
            "escalated_to_full": False,
        }

    if vector_profile == "balanced" and average_score >= 0.7 and weak_count == 0:
        evidence_bundle = assemble_fast_vector_evidence(merged, top_n=4)
        return {
            "source": "vector_document",
            "execution_profile": "balanced",
            "coverage_summary": coverage["coverage_summary"],
            "bundle_summary": evidence_bundle["bundle_summary"],
            "evidence_bundle": evidence_bundle,
            "escalated_to_full": False,
        }

    full_bundle = build_single_query_execution_bundle(query, model)
    full_result = execute_vector_source(
        full_bundle,
        index,
        namespace,
        alpha,
        top_k,
        openai_client,
        pinecone_client,
        child_index,
        parent_index,
        chunk_to_record_id,
        rerank_top_n=6,
        mmr_top_m=4,
        mmr_lambda=0.75,
    )
    full_result["execution_profile"] = "full" if vector_profile == "full" else f"{vector_profile}_escalated"
    full_result["escalated_to_full"] = vector_profile != "full"
    return full_result


def _execute_sub_query_sources(
    effective_sub_query: str,
    execution_policy: dict[str, Any],
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
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    selected_sources = list(execution_policy["active_sources"])
    source_outputs: list[dict[str, Any]] = []
    source_usage = {"vector_document": 0, "graph_relationships": 0, "sql_structured": 0}
    profile_usage = {"skip": 0, "fast": 0, "balanced": 0, "full": 0, "fast_escalated": 0, "balanced_escalated": 0}

    can_parallelize_sources = execution_policy["parallel_safe"] and len(selected_sources) > 1

    if can_parallelize_sources:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sources)) as executor:
            future_map: dict[str, concurrent.futures.Future] = {}
            for source in selected_sources:
                source_usage[source] += 1
                if source == "vector_document":
                    future_map[source] = executor.submit(
                        execute_vector_source_optimized,
                        effective_sub_query,
                        execution_policy["vector_profile"],
                        index,
                        namespace,
                        alpha,
                        top_k,
                        openai_client,
                        pinecone_client,
                        child_index,
                        parent_index,
                        chunk_to_record_id,
                        model,
                    )
                if source == "graph_relationships":
                    future_map[source] = executor.submit(execute_graph_source, graph_database_path, effective_sub_query)
                elif source == "sql_structured":
                    future_map[source] = executor.submit(
                        execute_sql_source,
                        effective_sub_query,
                        schema_package,
                        sql_database_path,
                        model,
                        openai_client,
                    )

            for source in selected_sources:
                output = future_map[source].result()
                if source == "vector_document":
                    profile_usage[output["execution_profile"]] = profile_usage.get(output["execution_profile"], 0) + 1
                if source == "graph_relationships":
                    output["executed_query"] = effective_sub_query
                source_outputs.append(output)
        return source_outputs, source_usage, profile_usage

    for source in selected_sources:
        source_usage[source] += 1
        if source == "vector_document":
            vector_output = execute_vector_source_optimized(
                effective_sub_query,
                execution_policy["vector_profile"],
                index,
                namespace,
                alpha,
                top_k,
                openai_client,
                pinecone_client,
                child_index,
                parent_index,
                chunk_to_record_id,
                model,
            )
            profile_usage[vector_output["execution_profile"]] = profile_usage.get(vector_output["execution_profile"], 0) + 1
            source_outputs.append(vector_output)
        elif source == "graph_relationships":
            graph_output = execute_graph_source(graph_database_path, effective_sub_query)
            graph_output["executed_query"] = effective_sub_query
            source_outputs.append(graph_output)
        elif source == "sql_structured":
            source_outputs.append(execute_sql_source(effective_sub_query, schema_package, sql_database_path, model, openai_client))

    return source_outputs, source_usage, profile_usage


def _execute_independent_sub_query(
    item: tuple[str, dict[str, Any]],
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
) -> dict[str, Any]:
    sub_query_id, sub_query_plan = item
    sub_query = sub_query_plan["original_sub_query"]
    execution_policy = sub_query_plan["execution_policy"]
    source_outputs, source_usage, profile_usage = _execute_sub_query_sources(
        sub_query,
        execution_policy,
        schema_package,
        sql_database_path,
        graph_database_path,
        index,
        namespace,
        alpha,
        top_k,
        openai_client,
        pinecone_client,
        child_index,
        parent_index,
        chunk_to_record_id,
        model,
    )
    return {
        "sub_query_id": sub_query_id,
        "original_sub_query": sub_query,
        "resolved_sub_query": sub_query,
        "context_resolution": {
            "resolved_sub_query": sub_query,
            "used_context": False,
            "reasoning": "No prior context resolution needed.",
        },
        "routing_decision": sub_query_plan["routing_decision"],
        "execution_policy": execution_policy,
        "source_outputs": source_outputs,
        "source_usage": source_usage,
        "profile_usage": profile_usage,
    }


def execute_latency_optimized_orchestration(
    query: str,
    transformed_bundle: dict[str, Any],
    routing_plan: dict[str, Any],
    policy_plan: dict[str, Any],
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
) -> dict[str, Any]:
    policy_by_sub_query = {
        item["sub_query_id"]: item["execution_policy"] for item in policy_plan["sub_query_execution_plans"]
    }

    sub_query_results = []
    source_usage = {"vector_document": 0, "graph_relationships": 0, "sql_structured": 0}
    profile_usage = {"skip": 0, "fast": 0, "balanced": 0, "full": 0, "fast_escalated": 0, "balanced_escalated": 0}

    independent_batch: list[tuple[str, dict[str, Any]]] = []

    def flush_independent_batch() -> None:
        nonlocal independent_batch
        if not independent_batch:
            return

        if len(independent_batch) == 1:
            completed = [
                _execute_independent_sub_query(
                    independent_batch[0],
                    schema_package,
                    sql_database_path,
                    graph_database_path,
                    index,
                    namespace,
                    alpha,
                    top_k,
                    openai_client,
                    pinecone_client,
                    child_index,
                    parent_index,
                    chunk_to_record_id,
                    model,
                )
            ]
        else:
            max_workers = min(4, len(independent_batch))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _execute_independent_sub_query,
                        item,
                        schema_package,
                        sql_database_path,
                        graph_database_path,
                        index,
                        namespace,
                        alpha,
                        top_k,
                        openai_client,
                        pinecone_client,
                        child_index,
                        parent_index,
                        chunk_to_record_id,
                        model,
                    )
                    for item in independent_batch
                ]
                completed = [future.result() for future in futures]

        for result in completed:
            for source, count in result.pop("source_usage").items():
                source_usage[source] += count
            for profile, count in result.pop("profile_usage").items():
                profile_usage[profile] = profile_usage.get(profile, 0) + count
            sub_query_results.append(result)

        independent_batch = []

    for sub_query_plan in routing_plan["sub_query_plans"]:
        sub_query_id = sub_query_plan["sub_query_id"]
        sub_query = sub_query_plan["original_sub_query"]
        execution_policy = policy_by_sub_query[sub_query_id]
        if (
            not execution_policy.get("signals", {}).get("has_reference", False)
            and "vector_document" not in execution_policy.get("active_sources", [])
        ):
            independent_batch.append(
                (
                    sub_query_id,
                    {
                        "original_sub_query": sub_query,
                        "routing_decision": sub_query_plan["routing_decision"],
                        "execution_policy": execution_policy,
                    },
                )
            )
            continue

        flush_independent_batch()

        selected_sources = execution_policy["active_sources"]
        resolution = resolve_sub_query_with_context(sub_query, sub_query_results, model, openai_client)
        effective_sub_query = resolution["resolved_sub_query"]
        source_outputs, sub_source_usage, sub_profile_usage = _execute_sub_query_sources(
            effective_sub_query,
            {**execution_policy, "parallel_safe": execution_policy["parallel_safe"] and not resolution["used_context"]},
            schema_package,
            sql_database_path,
            graph_database_path,
            index,
            namespace,
            alpha,
            top_k,
            openai_client,
            pinecone_client,
            child_index,
            parent_index,
            chunk_to_record_id,
            model,
        )
        for source, count in sub_source_usage.items():
            source_usage[source] += count
        for profile, count in sub_profile_usage.items():
            profile_usage[profile] = profile_usage.get(profile, 0) + count

        sub_query_results.append(
            {
                "sub_query_id": sub_query_id,
                "original_sub_query": sub_query,
                "resolved_sub_query": effective_sub_query,
                "context_resolution": resolution,
                "routing_decision": sub_query_plan["routing_decision"],
                "execution_policy": execution_policy,
                "source_outputs": source_outputs,
            }
        )

    flush_independent_batch()

    return {
        "original_query": query,
        "policy": {
            "routing_policy": routing_plan["policy"],
            "latency_policy": policy_plan["policy"],
        },
        "routing_summary": routing_plan["routing_summary"],
        "policy_summary": policy_plan["policy_summary"],
        "execution_summary": {
            "sub_query_count": len(sub_query_results),
            "source_usage": source_usage,
            "vector_profile_usage": profile_usage,
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
    report_path = output_dir / "latency_optimized_orchestration_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute latency-optimized multi-source orchestration.")
    parser.add_argument("--query", default="Which segment grew the fastest in FY2025 and what narrative driver was mentioned for it?")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--embedding-input", default=DEFAULT_EMBEDDING_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--sql-schema-path", default=DEFAULT_SQL_SCHEMA_PATH)
    parser.add_argument("--sql-database-path", default=DEFAULT_SQL_DATABASE_PATH)
    parser.add_argument("--graph-database-path", default=DEFAULT_GRAPH_DATABASE_PATH)
    parser.add_argument("--run-id", default="component7_latency_optimized_orchestration")
    args = parser.parse_args()

    load_project_env()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    start = time.perf_counter()
    transformed_bundle = build_minimal_orchestration_bundle(args.query, model=args.model, openai_client=OpenAI(api_key=openai_api_key))
    schema_package = load_json_result(PROJECT_ROOT / "agentic_document_intelligence" / args.sql_schema_path)
    sql_capability_summary = build_sql_capability_summary(schema_package)
    graph_capability_summary = build_graph_capability_summary()
    routing_plan = build_multi_source_routing_plan(
        transformed_bundle,
        sql_capability_summary,
        graph_capability_summary,
        model=args.model,
    )
    policy_plan = build_latency_optimized_policy(
        args.query,
        transformed_bundle,
        routing_plan,
        model=args.model,
    )

    chunk_artifact = load_chunk_artifact(PROJECT_ROOT / "agentic_document_intelligence" / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)
    chunk_to_record_id = load_embedding_records(PROJECT_ROOT / "agentic_document_intelligence" / args.embedding_input)
    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index(index_name)

    result = execute_latency_optimized_orchestration(
        query=args.query,
        transformed_bundle=transformed_bundle,
        routing_plan=routing_plan,
        policy_plan=policy_plan,
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
    )
    result["runtime_seconds"] = round(time.perf_counter() - start, 3)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "runtime_seconds": result["runtime_seconds"],
                "vector_profile_usage": result["execution_summary"]["vector_profile_usage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
