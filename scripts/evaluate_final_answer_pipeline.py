import argparse
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


from agentic_document_intelligence.scripts.benchmark_answer_generation_models import (
    DEFAULT_CASES_PATH,
    evaluate_answer,
    load_cases,
    prepare_fused_bundle_for_case,
    write_json,
)
from agentic_document_intelligence.scripts.corrective_answer_repair import repair_grounded_answer
from agentic_document_intelligence.scripts.execute_latency_optimized_orchestration import (
    DEFAULT_EMBEDDING_INPUT,
    DEFAULT_GRAPH_DATABASE_PATH,
    DEFAULT_SQL_DATABASE_PATH,
    DEFAULT_SQL_SCHEMA_PATH,
)
from agentic_document_intelligence.scripts.execute_multi_source_orchestration import (
    DEFAULT_CHUNK_INPUT,
    DEFAULT_NAMESPACE,
    load_project_env,
)
from agentic_document_intelligence.scripts.generate_grounded_answer import generate_grounded_answer
from agentic_document_intelligence.scripts.mmr_diversification import load_embedding_records
from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_sql_capability_summary,
    load_json_result,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import build_chunk_indexes, load_chunk_artifact
from agentic_document_intelligence.scripts.ragas_style_llm_judge import judge_final_answer
from agentic_document_intelligence.scripts.self_reflective_answer_critique import critique_grounded_answer


DEFAULT_PIPELINE_MODEL = "gpt-5-mini"
DEFAULT_ANSWER_MODEL = "gpt-5.1"
DEFAULT_CRITIQUE_MODEL = "gpt-5-mini"
DEFAULT_REPAIR_MODEL = "gpt-5.1"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"


def build_cache_dir(run_id: str) -> Path:
    path = (
        PROJECT_ROOT
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "final_answer_eval_cache"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def evaluate_case_end_to_end(
    case: dict[str, Any],
    fused_bundle: dict[str, Any],
    answer_model: str,
    critique_model: str,
    repair_model: str,
    judge_model: str,
    openai_client: OpenAI,
) -> dict[str, Any]:
    answer_start = time.perf_counter()
    initial_answer = generate_grounded_answer(fused_bundle, model=answer_model, client=openai_client)
    answer_runtime_seconds = round(time.perf_counter() - answer_start, 3)

    critique_start = time.perf_counter()
    critique_result = critique_grounded_answer(
        fused_bundle,
        initial_answer,
        model=critique_model,
        client=openai_client,
    )
    critique_runtime_seconds = round(time.perf_counter() - critique_start, 3)

    repair_start = time.perf_counter()
    repair_result = repair_grounded_answer(
        fused_bundle,
        initial_answer,
        critique_result,
        model=repair_model,
        client=openai_client,
    )
    repair_runtime_seconds = round(time.perf_counter() - repair_start, 3)

    final_answer = repair_result["repaired_answer"] if repair_result["repair_applied"] else initial_answer
    final_evaluation = evaluate_answer(final_answer, fused_bundle, case)
    initial_evaluation = evaluate_answer(initial_answer, fused_bundle, case)
    judge_start = time.perf_counter()
    judge_result = judge_final_answer(
        fused_bundle,
        final_answer,
        critique_result=critique_result,
        model=judge_model,
        client=openai_client,
    )
    judge_runtime_seconds = round(time.perf_counter() - judge_start, 3)

    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "runtime_seconds": round(answer_runtime_seconds + critique_runtime_seconds + repair_runtime_seconds + judge_runtime_seconds, 3),
        "answer_runtime_seconds": answer_runtime_seconds,
        "critique_runtime_seconds": critique_runtime_seconds,
        "repair_runtime_seconds": repair_runtime_seconds,
        "judge_runtime_seconds": judge_runtime_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "repair_applied": repair_result["repair_applied"],
        "repair_success": repair_result["repair_success"],
        "repair_strategy": repair_result["repair_strategy"],
        "initial_answer": initial_answer,
        "critique_result": critique_result,
        "repair_result": repair_result,
        "final_answer": final_answer,
        "judge_result": judge_result,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answers"
        / "final_answer_pipeline_eval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministically evaluate the final answer pipeline end to end.")
    parser.add_argument("--cases-path", default=DEFAULT_CASES_PATH)
    parser.add_argument("--pipeline-model", default=DEFAULT_PIPELINE_MODEL)
    parser.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    parser.add_argument("--critique-model", default=DEFAULT_CRITIQUE_MODEL)
    parser.add_argument("--repair-model", default=DEFAULT_REPAIR_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--embedding-input", default=DEFAULT_EMBEDDING_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--sql-schema-path", default=DEFAULT_SQL_SCHEMA_PATH)
    parser.add_argument("--sql-database-path", default=DEFAULT_SQL_DATABASE_PATH)
    parser.add_argument("--graph-database-path", default=DEFAULT_GRAPH_DATABASE_PATH)
    parser.add_argument("--run-id", default="component9_final_answer_pipeline_eval")
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
    cache_dir = build_cache_dir(args.run_id)
    fused_case_bundles = {}
    for case in cases:
        fused_bundle = prepare_fused_bundle_for_case(
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
        fused_case_bundles[case["case_id"]] = fused_bundle
        write_json(
            cache_dir / f"{case['case_id']}_fused_bundle_snapshot.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "result": fused_bundle,
            },
        )

    case_results = []
    total_runtime = 0.0
    initial_passed_count = 0
    final_passed_count = 0
    repaired_count = 0
    recovered_count = 0
    judge_passed_count = 0
    judge_average_score_total = 0.0
    for case in cases:
        case_result = evaluate_case_end_to_end(
            case,
            fused_case_bundles[case["case_id"]],
            args.answer_model,
            args.critique_model,
            args.repair_model,
            args.judge_model,
            openai_client,
        )
        case_results.append(case_result)
        total_runtime += case_result["runtime_seconds"]
        if case_result["initial_evaluation"]["passed"]:
            initial_passed_count += 1
        if case_result["final_evaluation"]["passed"]:
            final_passed_count += 1
        if case_result["repair_applied"]:
            repaired_count += 1
        if (not case_result["initial_evaluation"]["passed"]) and case_result["final_evaluation"]["passed"]:
            recovered_count += 1
        if case_result["judge_result"]["metrics"]["overall_verdict"] == "pass":
            judge_passed_count += 1
        judge_average_score_total += case_result["judge_result"]["average_score"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_model": args.pipeline_model,
        "answer_model": args.answer_model,
        "critique_model": args.critique_model,
        "repair_model": args.repair_model,
        "judge_model": args.judge_model,
        "case_count": len(case_results),
        "initial_passed_count": initial_passed_count,
        "initial_pass_rate": round(initial_passed_count / len(case_results), 4) if case_results else 0.0,
        "final_passed_count": final_passed_count,
        "final_pass_rate": round(final_passed_count / len(case_results), 4) if case_results else 0.0,
        "repaired_count": repaired_count,
        "recovered_count": recovered_count,
        "judge_passed_count": judge_passed_count,
        "judge_pass_rate": round(judge_passed_count / len(case_results), 4) if case_results else 0.0,
        "judge_average_score": round(judge_average_score_total / len(case_results), 3) if case_results else 0.0,
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
                "case_count": report["case_count"],
                "initial_pass_rate": report["initial_pass_rate"],
                "final_pass_rate": report["final_pass_rate"],
                "repaired_count": report["repaired_count"],
                "recovered_count": report["recovered_count"],
                "judge_pass_rate": report["judge_pass_rate"],
                "judge_average_score": report["judge_average_score"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
