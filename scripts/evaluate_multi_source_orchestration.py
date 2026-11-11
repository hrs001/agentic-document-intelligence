import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.execute_multi_source_orchestration import (
    DEFAULT_EMBEDDING_INPUT,
    DEFAULT_GRAPH_DATABASE_PATH,
    DEFAULT_SQL_DATABASE_PATH,
    DEFAULT_SQL_SCHEMA_PATH,
    execute_routed_orchestration,
    load_project_env,
    write_report as write_orchestration_report,
)
from agentic_document_intelligence.scripts.multi_source_routing import (
    MODEL_NAME,
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import build_chunk_indexes, load_chunk_artifact
from agentic_document_intelligence.scripts.transformed_query_bundle_orchestrator import build_transformed_query_bundle
from agentic_document_intelligence.scripts.mmr_diversification import load_embedding_records

from openai import OpenAI
from pinecone import Pinecone
import os


DEFAULT_CHUNK_INPUT = "artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json"
DEFAULT_NAMESPACE = "microsoft_fy2025_fixed_corpus"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_source_text(source_output: dict[str, Any]) -> str:
    bundle = source_output.get("evidence_bundle", {})
    for key in ("assembled_evidence_text", "assembled_graph_evidence_text", "assembled_sql_evidence_text"):
        if key in bundle:
            return bundle[key]
    return json.dumps(bundle)


def evaluate_case(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    source_outputs = []
    for sub_query_result in result["sub_query_results"]:
        source_outputs.extend(sub_query_result["source_outputs"])

    source_map: dict[str, list[dict[str, Any]]] = {}
    for item in source_outputs:
        source_map.setdefault(item["source"], []).append(item)

    missing_sources = [source for source in case.get("required_sources", []) if source not in source_map]
    missing_keywords = []
    for expectation in case.get("keyword_expectations", []):
        source = expectation["source"]
        source_text = "\n".join(extract_source_text(item) for item in source_map.get(source, [])).lower()
        for keyword in expectation["keywords"]:
            if keyword.lower() not in source_text:
                missing_keywords.append({"source": source, "keyword": keyword})
    passed = not missing_sources and not missing_keywords
    return {
        "missing_sources": missing_sources,
        "missing_keywords": missing_keywords,
        "passed": passed,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "orchestration"
        / "multi_source_orchestration_eval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate routed multi-source orchestration end to end.")
    parser.add_argument(
        "--cases-path",
        default="evals/multi_source_orchestration_cases.json",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--embedding-input", default=DEFAULT_EMBEDDING_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--sql-schema-path", default=DEFAULT_SQL_SCHEMA_PATH)
    parser.add_argument("--sql-database-path", default=DEFAULT_SQL_DATABASE_PATH)
    parser.add_argument("--graph-database-path", default=DEFAULT_GRAPH_DATABASE_PATH)
    parser.add_argument("--run-id", default="component7_multi_source_orchestration_eval")
    args = parser.parse_args()

    load_project_env()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key or not pinecone_api_key:
        raise RuntimeError("OPENAI_API_KEY and PINECONE_API_KEY must be set in .env")

    schema_package = load_json_result(PROJECT_ROOT / "agentic_document_intelligence" / args.sql_schema_path)
    sql_capability_summary = build_sql_capability_summary(schema_package)
    graph_capability_summary = build_graph_capability_summary()
    chunk_artifact = load_chunk_artifact(PROJECT_ROOT / "agentic_document_intelligence" / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)
    chunk_to_record_id = load_embedding_records(PROJECT_ROOT / "agentic_document_intelligence" / args.embedding_input)
    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index(index_name)

    cases = load_cases(PROJECT_ROOT / "agentic_document_intelligence" / args.cases_path)
    case_results = []
    for case in cases:
        transformed_bundle = build_transformed_query_bundle(case["query"], model=args.model)
        routing_plan = build_multi_source_routing_plan(
            transformed_bundle,
            sql_capability_summary,
            graph_capability_summary,
            model=args.model,
        )
        result = execute_routed_orchestration(
            query=case["query"],
            transformed_bundle=transformed_bundle,
            routing_plan=routing_plan,
            schema_package=schema_package,
            sql_database_path=PROJECT_ROOT / "agentic_document_intelligence" / args.sql_database_path,
            graph_database_path=PROJECT_ROOT / "agentic_document_intelligence" / args.graph_database_path,
            index=index,
            namespace=args.namespace,
            alpha=0.6,
            top_k=6,
            openai_client=openai_client,
            pinecone_client=pinecone_client,
            child_index=child_index,
            parent_index=parent_index,
            chunk_to_record_id=chunk_to_record_id,
            model=args.model,
            rerank_top_n=6,
            mmr_top_m=4,
            mmr_lambda=0.75,
        )
        evaluation = evaluate_case(result, case)
        case_results.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "execution_summary": result["execution_summary"],
                "sub_query_results": result["sub_query_results"],
                "evaluation": evaluation,
            }
        )

    passed_count = sum(1 for item in case_results if item["evaluation"]["passed"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "case_count": len(case_results),
        "passed_count": passed_count,
        "pass_rate": round(passed_count / len(case_results), 4) if case_results else 0.0,
        "case_results": case_results,
    }
    report_path = write_report(PROJECT_ROOT, args.run_id, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(case_results),
                "passed_count": passed_count,
                "pass_rate": report["pass_rate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
