import argparse
import json
import os
import re
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


from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.execute_latency_optimized_orchestration import (
    DEFAULT_EMBEDDING_INPUT,
    DEFAULT_GRAPH_DATABASE_PATH,
    DEFAULT_SQL_DATABASE_PATH,
    DEFAULT_SQL_SCHEMA_PATH,
    build_minimal_orchestration_bundle,
    execute_latency_optimized_orchestration,
)
from agentic_document_intelligence.scripts.execute_multi_source_orchestration import (
    DEFAULT_CHUNK_INPUT,
    DEFAULT_NAMESPACE,
    load_project_env,
)
from agentic_document_intelligence.scripts.generate_grounded_answer import generate_grounded_answer
from agentic_document_intelligence.scripts.latency_optimized_orchestration_policy import build_latency_optimized_policy
from agentic_document_intelligence.scripts.mmr_diversification import load_embedding_records
from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import build_chunk_indexes, load_chunk_artifact


DEFAULT_CASES_PATH = "evals/answer_generation_benchmark_cases.json"
DEFAULT_PIPELINE_MODEL = "gpt-5-mini"
DEFAULT_ANSWER_MODELS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.2", "gpt-5.1", "gpt-5-mini"]


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def phrase_present(answer_text: str, phrase: str) -> bool:
    answer_lower = answer_text.lower()
    phrase_lower = phrase.lower()
    if phrase_lower in answer_lower:
        return True

    normalized_answer = re.sub(r"[\s,$%]+", "", answer_lower)
    normalized_phrase = re.sub(r"[\s,$%]+", "", phrase_lower)
    return bool(normalized_phrase) and normalized_phrase in normalized_answer


def build_case_cache_dir(run_id: str) -> Path:
    path = (
        PROJECT_ROOT
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answer_benchmark_cache"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_fused_bundle_for_case(
    case: dict[str, Any],
    cache_dir: Path,
    schema_package: dict[str, Any],
    sql_capability_summary: str,
    graph_capability_summary: str,
    sql_database_path: Path,
    graph_database_path: Path,
    index: Any,
    namespace: str,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
    child_index: dict[str, dict[str, Any]],
    parent_index: dict[str, str],
    chunk_to_record_id: dict[str, str],
    pipeline_model: str,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{slugify(case['case_id'])}_fused_bundle.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload["result"]

    transformed_bundle = build_minimal_orchestration_bundle(case["query"], model=pipeline_model, openai_client=openai_client)
    routing_plan = build_multi_source_routing_plan(
        transformed_bundle,
        sql_capability_summary,
        graph_capability_summary,
        model=pipeline_model,
    )
    policy_plan = build_latency_optimized_policy(
        case["query"],
        transformed_bundle,
        routing_plan,
        model=pipeline_model,
        client=openai_client,
    )
    orchestration_result = execute_latency_optimized_orchestration(
        query=case["query"],
        transformed_bundle=transformed_bundle,
        routing_plan=routing_plan,
        policy_plan=policy_plan,
        schema_package=schema_package,
        sql_database_path=sql_database_path,
        graph_database_path=graph_database_path,
        index=index,
        namespace=namespace,
        alpha=0.6,
        top_k=6,
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        child_index=child_index,
        parent_index=parent_index,
        chunk_to_record_id=chunk_to_record_id,
        model=pipeline_model,
    )
    fused_bundle = fuse_cross_source_evidence(orchestration_result)
    write_json(
        cache_path,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_id": case["case_id"],
            "pipeline_model": pipeline_model,
            "result": fused_bundle,
        },
    )
    return fused_bundle


def evaluate_answer(answer_result: dict[str, Any], fused_bundle: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    answer_text = answer_result.get("answer_markdown", "")
    required_phrases = case.get("required_phrases", [])
    missing_phrases = [phrase for phrase in required_phrases if not phrase_present(answer_text, phrase)]

    fact_source_map = {fact["fact_id"]: fact["source_type"] for fact in fused_bundle.get("normalized_facts", [])}
    cited_sources = {fact_source_map[fact_id] for fact_id in answer_result.get("used_fact_ids", []) if fact_id in fact_source_map}
    missing_source_types = [source for source in case.get("required_source_types", []) if source not in cited_sources]

    min_citation_count = int(case.get("min_citation_count", 1))
    citation_count = len(answer_result.get("used_fact_ids", []))
    missing_inline_citations = [fact_id for fact_id in answer_result.get("used_fact_ids", []) if f"[{fact_id}]" not in answer_text]

    passed = (
        not missing_phrases
        and not missing_source_types
        and citation_count >= min_citation_count
        and not missing_inline_citations
        and not answer_result.get("unanswered_sub_queries", [])
    )
    return {
        "passed": passed,
        "missing_phrases": missing_phrases,
        "missing_source_types": missing_source_types,
        "citation_count": citation_count,
        "missing_inline_citations": missing_inline_citations,
        "unanswered_sub_queries": answer_result.get("unanswered_sub_queries", []),
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answers"
        / "answer_generation_model_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark answer generation quality across candidate OpenAI models.")
    parser.add_argument("--cases-path", default=DEFAULT_CASES_PATH)
    parser.add_argument("--pipeline-model", default=DEFAULT_PIPELINE_MODEL)
    parser.add_argument("--answer-models", nargs="+", default=DEFAULT_ANSWER_MODELS)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--embedding-input", default=DEFAULT_EMBEDDING_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--sql-schema-path", default=DEFAULT_SQL_SCHEMA_PATH)
    parser.add_argument("--sql-database-path", default=DEFAULT_SQL_DATABASE_PATH)
    parser.add_argument("--graph-database-path", default=DEFAULT_GRAPH_DATABASE_PATH)
    parser.add_argument("--run-id", default="component8_answer_generation_model_benchmark")
    args = parser.parse_args()

    load_project_env()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key or not pinecone_api_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set in .env")

    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index(index_name)

    schema_package = load_json_result(PROJECT_ROOT / "agentic_document_intelligence" / args.sql_schema_path)
    sql_capability_summary = build_sql_capability_summary(schema_package)
    graph_capability_summary = build_graph_capability_summary()
    chunk_artifact = load_chunk_artifact(PROJECT_ROOT / "agentic_document_intelligence" / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)
    chunk_to_record_id = load_embedding_records(PROJECT_ROOT / "agentic_document_intelligence" / args.embedding_input)
    sql_database_path = PROJECT_ROOT / "agentic_document_intelligence" / args.sql_database_path
    graph_database_path = PROJECT_ROOT / "agentic_document_intelligence" / args.graph_database_path
    cases = load_cases(PROJECT_ROOT / "agentic_document_intelligence" / args.cases_path)
    cache_dir = build_case_cache_dir(args.run_id)

    fused_case_bundles = {}
    for case in cases:
        fused_case_bundles[case["case_id"]] = prepare_fused_bundle_for_case(
            case,
            cache_dir,
            schema_package,
            sql_capability_summary,
            graph_capability_summary,
            sql_database_path,
            graph_database_path,
            index,
            args.namespace,
            openai_client,
            pinecone_client,
            child_index,
            parent_index,
            chunk_to_record_id,
            args.pipeline_model,
        )

    model_results = []
    for model_name in args.answer_models:
        case_results = []
        model_start = time.perf_counter()
        for case in cases:
            fused_bundle = fused_case_bundles[case["case_id"]]
            answer_start = time.perf_counter()
            answer_result = generate_grounded_answer(fused_bundle, model=model_name, client=openai_client)
            runtime_seconds = round(time.perf_counter() - answer_start, 3)
            evaluation = evaluate_answer(answer_result, fused_bundle, case)
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "runtime_seconds": runtime_seconds,
                    "evaluation": evaluation,
                    "answer_result": answer_result,
                }
            )

        passed_count = sum(1 for item in case_results if item["evaluation"]["passed"])
        total_runtime = round(time.perf_counter() - model_start, 3)
        model_results.append(
            {
                "model": model_name,
                "case_count": len(case_results),
                "passed_count": passed_count,
                "pass_rate": round(passed_count / len(case_results), 4) if case_results else 0.0,
                "total_runtime_seconds": total_runtime,
                "average_runtime_seconds": round(total_runtime / len(case_results), 3) if case_results else 0.0,
                "case_results": case_results,
            }
        )

    best_model = max(model_results, key=lambda item: (item["pass_rate"], -item["average_runtime_seconds"])) if model_results else {}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_model": args.pipeline_model,
        "answer_models": args.answer_models,
        "case_count": len(cases),
        "best_model": best_model.get("model"),
        "model_results": model_results,
    }
    report_path = write_report(PROJECT_ROOT, args.run_id, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(cases),
                "best_model": report["best_model"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
