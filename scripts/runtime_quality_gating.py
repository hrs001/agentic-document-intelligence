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


from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.corrective_answer_repair import repair_grounded_answer
from agentic_document_intelligence.scripts.contextual_compression import attach_contextual_compression
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
from agentic_document_intelligence.scripts.ragas_style_llm_judge import judge_final_answer
from agentic_document_intelligence.scripts.self_reflective_answer_critique import critique_grounded_answer


DEFAULT_PIPELINE_MODEL = "gpt-5-mini"
DEFAULT_ANSWER_MODEL = "gpt-5.1"
DEFAULT_COMPRESSION_MODEL = "gpt-5-mini"
DEFAULT_MIN_FACTS_FOR_COMPRESSION = 30
DEFAULT_CRITIQUE_MODEL = "gpt-5-mini"
DEFAULT_REPAIR_MODEL = "gpt-5.1"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"

DEFAULT_MAX_TOTAL_ROUNDS = 2
DEFAULT_MAX_PIPELINE_RERUNS = 1
DEFAULT_MAX_ANSWER_REGENERATIONS = 1
DEFAULT_MAX_REPAIRS = 1


def build_runtime_bundle_for_query(
    query: str,
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
    transformed_bundle = build_minimal_orchestration_bundle(query, model=pipeline_model, openai_client=openai_client)
    routing_plan = build_multi_source_routing_plan(
        transformed_bundle,
        sql_capability_summary,
        graph_capability_summary,
        model=pipeline_model,
        client=openai_client,
    )
    policy_plan = build_latency_optimized_policy(
        query,
        transformed_bundle,
        routing_plan,
        model=pipeline_model,
        client=openai_client,
    )
    orchestration_result = execute_latency_optimized_orchestration(
        query=query,
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
    return attach_contextual_compression(
        fused_bundle,
        model=DEFAULT_COMPRESSION_MODEL,
        client=openai_client,
        min_facts_for_compression=DEFAULT_MIN_FACTS_FOR_COMPRESSION,
    )


def build_retry_state() -> dict[str, Any]:
    return {
        "total_rounds_used": 0,
        "pipeline_reruns_used": 0,
        "answer_regenerations_used": 0,
        "repairs_used": 0,
        "actions_taken": [],
        "last_judge_average_score": None,
    }


def judge_is_passing(judge_result: dict[str, Any]) -> bool:
    return judge_result["metrics"]["overall_verdict"] == "pass"


def decide_runtime_action(
    critique_result: dict[str, Any],
    judge_result: dict[str, Any],
    retry_state: dict[str, Any],
    max_total_rounds: int = DEFAULT_MAX_TOTAL_ROUNDS,
    max_pipeline_reruns: int = DEFAULT_MAX_PIPELINE_RERUNS,
    max_answer_regenerations: int = DEFAULT_MAX_ANSWER_REGENERATIONS,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
) -> dict[str, Any]:
    final_critique = critique_result.get("final_critique", {})
    deterministic = critique_result.get("deterministic_signals", {})
    issues = final_critique.get("issues", [])
    issue_types = {str(issue.get("issue_type", "")).strip() for issue in issues}
    verdict = judge_result["metrics"]["overall_verdict"]
    faithfulness = int(judge_result["metrics"]["faithfulness"])
    answer_relevancy = int(judge_result["metrics"]["answer_relevancy"])
    context_precision = int(judge_result["metrics"]["context_precision"])
    citation_grounding = int(judge_result["metrics"]["citation_grounding"])
    current_score = float(judge_result["average_score"])
    last_score = retry_state.get("last_judge_average_score")
    non_improving = last_score is not None and current_score <= last_score
    has_integrity_failure = bool(
        deterministic.get("missing_inline_for_used")
        or deterministic.get("inline_not_declared")
        or deterministic.get("unknown_used_fact_ids")
    )
    has_coverage_failure = bool(deterministic.get("unanswered_sub_queries"))

    if not final_critique.get("needs_correction", False) and verdict == "pass":
        return {"action": "stop_accept", "reason": "Critique and judge both accept the answer."}

    if (
        verdict == "pass"
        and not final_critique.get("needs_correction", False)
        and not has_integrity_failure
        and faithfulness >= 4
        and answer_relevancy >= 4
        and citation_grounding >= 4
    ):
        return {"action": "stop_accept", "reason": "Judge passed the answer and citation integrity is intact."}

    if (
        not has_integrity_failure
        and not has_coverage_failure
        and not final_critique.get("needs_correction", False)
        and verdict in {"pass", "borderline"}
        and current_score >= 3.75
        and faithfulness >= 4
        and answer_relevancy >= 4
        and citation_grounding >= 4
    ):
        return {"action": "stop_accept", "reason": "Answer is grounded enough to return without an expensive retry."}

    if retry_state["total_rounds_used"] >= max_total_rounds:
        return {"action": "stop_best_effort", "reason": "Total retry budget exhausted."}

    if non_improving and retry_state["actions_taken"]:
        return {"action": "stop_best_effort", "reason": "Quality did not improve after the previous retry."}

    if (deterministic.get("missing_inline_for_used") or deterministic.get("inline_not_declared")) and retry_state["repairs_used"] < max_repairs:
        return {"action": "citation_strict_repair", "reason": "Citation integrity failed deterministic checks."}

    if (
        final_critique.get("needs_correction", False)
        and "coverage_gap" in issue_types
        and has_coverage_failure
        and verdict != "pass"
        and retry_state["repairs_used"] < max_repairs
    ):
        return {"action": "targeted_answer_repair", "reason": "Coverage gap detected by critique."}

    if verdict in {"fail", "borderline"} and answer_relevancy <= 3 and retry_state["answer_regenerations_used"] < max_answer_regenerations:
        return {"action": "answer_regeneration_only", "reason": "Answer relevancy is too low."}

    if verdict == "fail" and (faithfulness <= 2 or context_precision <= 2) and retry_state["pipeline_reruns_used"] < max_pipeline_reruns:
        return {"action": "full_pipeline_rerun_once", "reason": "Judge indicates low faithfulness or noisy context."}

    if final_critique.get("needs_correction", False) and retry_state["repairs_used"] < max_repairs:
        return {"action": "targeted_answer_repair", "reason": "Critique still requests correction."}

    if verdict == "borderline" and citation_grounding <= 3 and retry_state["repairs_used"] < max_repairs:
        return {"action": "citation_strict_repair", "reason": "Citation grounding is borderline and repair budget remains."}

    return {"action": "stop_best_effort", "reason": "No safe retry action remains within budget."}


def run_single_answer_cycle(
    fused_bundle: dict[str, Any],
    answer_model: str,
    critique_model: str,
    judge_model: str,
    openai_client: OpenAI,
) -> dict[str, Any]:
    answer_result = generate_grounded_answer(fused_bundle, model=answer_model, client=openai_client)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        critique_future = executor.submit(critique_grounded_answer, fused_bundle, answer_result, critique_model, openai_client)
        judge_future = executor.submit(judge_final_answer, fused_bundle, answer_result, None, judge_model, openai_client)
        critique_result = critique_future.result()
        judge_result = judge_future.result()
    return {
        "answer_result": answer_result,
        "critique_result": critique_result,
        "judge_result": judge_result,
    }


def execute_runtime_quality_gated_query(
    query: str,
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
    pipeline_model: str = DEFAULT_PIPELINE_MODEL,
    answer_model: str = DEFAULT_ANSWER_MODEL,
    critique_model: str = DEFAULT_CRITIQUE_MODEL,
    repair_model: str = DEFAULT_REPAIR_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_total_rounds: int = DEFAULT_MAX_TOTAL_ROUNDS,
    max_pipeline_reruns: int = DEFAULT_MAX_PIPELINE_RERUNS,
    max_answer_regenerations: int = DEFAULT_MAX_ANSWER_REGENERATIONS,
    max_repairs: int = DEFAULT_MAX_REPAIRS,
) -> dict[str, Any]:
    retry_state = build_retry_state()
    fused_bundle = build_runtime_bundle_for_query(
        query,
        schema_package,
        sql_capability_summary,
        graph_capability_summary,
        sql_database_path,
        graph_database_path,
        index,
        namespace,
        openai_client,
        pinecone_client,
        child_index,
        parent_index,
        chunk_to_record_id,
        pipeline_model,
    )

    cycle = run_single_answer_cycle(fused_bundle, answer_model, critique_model, judge_model, openai_client)
    history = [
        {
            "round": 0,
            "action": "initial_generation",
            "judge_average_score": cycle["judge_result"]["average_score"],
            "judge_verdict": cycle["judge_result"]["metrics"]["overall_verdict"],
            "critique_needs_correction": cycle["critique_result"]["final_critique"]["needs_correction"],
        }
    ]
    retry_state["last_judge_average_score"] = cycle["judge_result"]["average_score"]

    while True:
        decision = decide_runtime_action(
            cycle["critique_result"],
            cycle["judge_result"],
            retry_state,
            max_total_rounds=max_total_rounds,
            max_pipeline_reruns=max_pipeline_reruns,
            max_answer_regenerations=max_answer_regenerations,
            max_repairs=max_repairs,
        )
        if decision["action"] in {"stop_accept", "stop_best_effort"}:
            return {
                "original_query": query,
                "policy": {
                    "pipeline_model": pipeline_model,
                    "answer_model": answer_model,
                    "critique_model": critique_model,
                    "repair_model": repair_model,
                    "judge_model": judge_model,
                    "max_total_rounds": max_total_rounds,
                    "max_pipeline_reruns": max_pipeline_reruns,
                    "max_answer_regenerations": max_answer_regenerations,
                    "max_repairs": max_repairs,
                },
                "retry_state": retry_state,
                "termination": decision,
                "history": history,
                "fused_bundle_summary": fused_bundle.get("bundle_summary", {}),
                "final_answer": cycle["answer_result"],
                "final_critique": cycle["critique_result"],
                "final_judge": cycle["judge_result"],
            }

        retry_state["total_rounds_used"] += 1
        retry_state["actions_taken"].append(decision["action"])

        if decision["action"] == "answer_regeneration_only":
            retry_state["answer_regenerations_used"] += 1
            cycle = run_single_answer_cycle(fused_bundle, answer_model, critique_model, judge_model, openai_client)
        elif decision["action"] in {"targeted_answer_repair", "citation_strict_repair"}:
            retry_state["repairs_used"] += 1
            repair_result = repair_grounded_answer(
                fused_bundle,
                cycle["answer_result"],
                cycle["critique_result"],
                model=repair_model,
                client=openai_client,
            )
            answer_result = repair_result["repaired_answer"] if repair_result["repair_applied"] else cycle["answer_result"]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                critique_future = executor.submit(
                    critique_grounded_answer,
                    fused_bundle,
                    answer_result,
                    critique_model,
                    openai_client,
                )
                judge_future = executor.submit(
                    judge_final_answer,
                    fused_bundle,
                    answer_result,
                    None,
                    judge_model,
                    openai_client,
                )
                critique_result = critique_future.result()
                judge_result = judge_future.result()
            cycle = {
                "answer_result": answer_result,
                "critique_result": critique_result,
                "judge_result": judge_result,
                "repair_result": repair_result,
            }
        elif decision["action"] == "full_pipeline_rerun_once":
            retry_state["pipeline_reruns_used"] += 1
            fused_bundle = build_runtime_bundle_for_query(
                query,
                schema_package,
                sql_capability_summary,
                graph_capability_summary,
                sql_database_path,
                graph_database_path,
                index,
                namespace,
                openai_client,
                pinecone_client,
                child_index,
                parent_index,
                chunk_to_record_id,
                pipeline_model,
            )
            cycle = run_single_answer_cycle(fused_bundle, answer_model, critique_model, judge_model, openai_client)

        retry_state["last_judge_average_score"] = cycle["judge_result"]["average_score"]
        history.append(
            {
                "round": retry_state["total_rounds_used"],
                "action": decision["action"],
                "reason": decision["reason"],
                "judge_average_score": cycle["judge_result"]["average_score"],
                "judge_verdict": cycle["judge_result"]["metrics"]["overall_verdict"],
                "critique_needs_correction": cycle["critique_result"]["final_critique"]["needs_correction"],
            }
        )


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
    report_path = output_dir / "runtime_quality_gating_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run runtime quality gating and escalation for a single query.")
    parser.add_argument("--query", default="Which segment includes GitHub and what was its FY2025 revenue?")
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
    parser.add_argument("--run-id", default="component10_runtime_quality_gating")
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

    start = time.perf_counter()
    result = execute_runtime_quality_gated_query(
        query=args.query,
        schema_package=schema_package,
        sql_capability_summary=sql_capability_summary,
        graph_capability_summary=graph_capability_summary,
        sql_database_path=PROJECT_ROOT / "agentic_document_intelligence" / args.sql_database_path,
        graph_database_path=PROJECT_ROOT / "agentic_document_intelligence" / args.graph_database_path,
        index=index,
        namespace=args.namespace,
        openai_client=openai_client,
        pinecone_client=pinecone_client,
        child_index=child_index,
        parent_index=parent_index,
        chunk_to_record_id=chunk_to_record_id,
        pipeline_model=args.pipeline_model,
        answer_model=args.answer_model,
        critique_model=args.critique_model,
        repair_model=args.repair_model,
        judge_model=args.judge_model,
    )
    result["runtime_seconds"] = round(time.perf_counter() - start, 3)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "termination": result["termination"],
                "retry_state": result["retry_state"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
