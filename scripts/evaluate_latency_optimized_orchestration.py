import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_multi_source_orchestration import evaluate_case
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
from agentic_document_intelligence.scripts.latency_optimized_orchestration_policy import (
    MODEL_NAME,
    build_latency_optimized_policy,
)
from agentic_document_intelligence.scripts.mmr_diversification import load_embedding_records
from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_multi_source_routing_plan,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import build_chunk_indexes, load_chunk_artifact


def load_cases(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_report(project_root: Path, run_id: str, report: dict[str, object]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "orchestration"
        / "latency_optimized_orchestration_eval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate latency-optimized orchestration quality and latency.")
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
    parser.add_argument("--run-id", default="component7_latency_optimized_orchestration_eval")
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
    total_runtime = 0.0
    for case in cases:
        case_start = time.perf_counter()
        transformed_bundle = build_minimal_orchestration_bundle(case["query"], model=args.model, openai_client=openai_client)
        routing_plan = build_multi_source_routing_plan(
            transformed_bundle,
            sql_capability_summary,
            graph_capability_summary,
            model=args.model,
        )
        policy_plan = build_latency_optimized_policy(
            case["query"],
            transformed_bundle,
            routing_plan,
            model=args.model,
        )
        result = execute_latency_optimized_orchestration(
            query=case["query"],
            transformed_bundle=transformed_bundle,
            routing_plan=routing_plan,
            policy_plan=policy_plan,
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
        )
        runtime_seconds = round(time.perf_counter() - case_start, 3)
        total_runtime += runtime_seconds
        evaluation = evaluate_case(result, case)
        case_results.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "runtime_seconds": runtime_seconds,
                "execution_summary": result["execution_summary"],
                "policy_summary": result["policy_summary"],
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
        "average_runtime_seconds": round(total_runtime / len(case_results), 3) if case_results else 0.0,
        "total_runtime_seconds": round(total_runtime, 3),
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
                "average_runtime_seconds": report["average_runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
