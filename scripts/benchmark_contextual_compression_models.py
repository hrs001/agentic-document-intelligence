import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.backend.app.service import DemoResources, build_ui_citations_and_answer
from agentic_document_intelligence.scripts.contextual_compression import (
    MODEL_PRICING_PER_M_TOKEN,
    attach_contextual_compression,
)
from agentic_document_intelligence.scripts.cross_source_evidence_fusion import fuse_cross_source_evidence
from agentic_document_intelligence.scripts.execute_latency_optimized_orchestration import (
    DEFAULT_GRAPH_DATABASE_PATH,
    DEFAULT_SQL_DATABASE_PATH,
    build_minimal_orchestration_bundle,
    execute_latency_optimized_orchestration,
)
from agentic_document_intelligence.scripts.execute_multi_source_orchestration import DEFAULT_NAMESPACE
from agentic_document_intelligence.scripts.generate_grounded_answer import generate_grounded_answer
from agentic_document_intelligence.scripts.latency_optimized_orchestration_policy import build_latency_optimized_policy
from agentic_document_intelligence.scripts.multi_source_routing import build_multi_source_routing_plan
from agentic_document_intelligence.scripts.ragas_style_llm_judge import judge_final_answer
from agentic_document_intelligence.scripts.self_reflective_answer_critique import critique_grounded_answer


DEFAULT_INPUT = "agentic_document_intelligence/evals/contextual_compression_benchmark_cases.json"
DEFAULT_COMPRESSION_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.1", "gpt-5-mini"]
DEFAULT_PIPELINE_MODEL = "gpt-5-mini"
DEFAULT_ANSWER_MODEL = "gpt-5.1"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_cost_from_usage(model: str, usage: dict[str, Any]) -> float:
    pricing = MODEL_PRICING_PER_M_TOKEN.get(model)
    if not pricing:
        return 0.0
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    return ((prompt_tokens / 1_000_000) * pricing["input"]) + ((completion_tokens / 1_000_000) * pricing["output"])


def build_fused_bundle(resources: DemoResources, query: str) -> dict[str, Any]:
    transformed_bundle = build_minimal_orchestration_bundle(
        query,
        model=DEFAULT_PIPELINE_MODEL,
        openai_client=resources.openai_client,
    )
    routing_plan = build_multi_source_routing_plan(
        transformed_bundle,
        resources.sql_capability_summary,
        resources.graph_capability_summary,
        model=DEFAULT_PIPELINE_MODEL,
        client=resources.openai_client,
    )
    policy_plan = build_latency_optimized_policy(
        query,
        transformed_bundle,
        routing_plan,
        model=DEFAULT_PIPELINE_MODEL,
        client=resources.openai_client,
    )
    orchestration_result = execute_latency_optimized_orchestration(
        query=query,
        transformed_bundle=transformed_bundle,
        routing_plan=routing_plan,
        policy_plan=policy_plan,
        schema_package=resources.schema_package,
        sql_database_path=PROJECT_ROOT / "agentic_document_intelligence" / DEFAULT_SQL_DATABASE_PATH,
        graph_database_path=PROJECT_ROOT / "agentic_document_intelligence" / DEFAULT_GRAPH_DATABASE_PATH,
        index=resources.index,
        namespace=DEFAULT_NAMESPACE,
        alpha=0.6,
        top_k=6,
        openai_client=resources.openai_client,
        pinecone_client=resources.pinecone_client,
        child_index=resources.child_index,
        parent_index=resources.parent_index,
        chunk_to_record_id=resources.chunk_to_record_id,
        model=DEFAULT_PIPELINE_MODEL,
    )
    return fuse_cross_source_evidence(orchestration_result)


def evaluate_answer_bundle(
    bundle: dict[str, Any],
    resources: DemoResources,
) -> dict[str, Any]:
    answer_result = generate_grounded_answer(bundle, model=DEFAULT_ANSWER_MODEL, client=resources.openai_client)
    critique_result = critique_grounded_answer(bundle, answer_result, model=DEFAULT_JUDGE_MODEL, client=resources.openai_client)
    judge_result = judge_final_answer(
        bundle,
        answer_result,
        critique_result=critique_result,
        model=DEFAULT_JUDGE_MODEL,
        client=resources.openai_client,
    )
    answer_text, citations = build_ui_citations_and_answer(bundle, answer_result)
    citation_integrity = bool(citations) and all(item.get("display_label") and item.get("source_type") for item in citations)
    return {
        "answer_result": answer_result,
        "critique_result": critique_result,
        "judge_result": judge_result,
        "ui_answer_preview": answer_text[:400],
        "citation_count": len(citations),
        "citation_integrity": citation_integrity,
    }


def benchmark_model(resources: DemoResources, cases: list[dict[str, Any]], model: str, fused_bundles: list[dict[str, Any]]) -> dict[str, Any]:
    case_results = []
    total_score = 0.0
    total_cost = 0.0
    total_latency = 0.0
    pass_count = 0
    citation_integrity_passes = 0

    for case, fused_bundle in zip(cases, fused_bundles):
        started_at = time.perf_counter()
        compressed_bundle = attach_contextual_compression(
            fused_bundle,
            model=model,
            client=resources.openai_client,
        )
        compression_latency = time.perf_counter() - started_at
        compression_usage = compressed_bundle.get("compression_context", {}).get("usage", {})
        answer_eval = evaluate_answer_bundle(compressed_bundle, resources)
        avg_score = float(answer_eval["judge_result"]["average_score"])
        passed = answer_eval["judge_result"]["metrics"]["overall_verdict"] == "pass"
        total_score += avg_score
        total_cost += float(compression_usage.get("estimated_cost_usd", estimate_cost_from_usage(model, compression_usage)))
        total_latency += compression_latency
        pass_count += int(passed)
        citation_integrity_passes += int(answer_eval["citation_integrity"])
        case_results.append(
            {
                "query": case["query"],
                "compression_model": model,
                "compression_applied": compressed_bundle.get("compression_context", {}).get("applied", False),
                "compression_latency_seconds": round(compression_latency, 3),
                "compression_usage": compression_usage,
                "judge_average_score": avg_score,
                "judge_verdict": answer_eval["judge_result"]["metrics"]["overall_verdict"],
                "citation_integrity": answer_eval["citation_integrity"],
                "citation_count": answer_eval["citation_count"],
                "answer_preview": answer_eval["ui_answer_preview"],
            }
        )

    case_count = max(len(case_results), 1)
    return {
        "model": model,
        "summary": {
            "avg_judge_score": round(total_score / case_count, 3),
            "pass_rate": round(pass_count / case_count, 3),
            "citation_integrity_rate": round(citation_integrity_passes / case_count, 3),
            "avg_compression_latency_seconds": round(total_latency / case_count, 3),
            "estimated_compression_cost_usd": round(total_cost, 6),
        },
        "cases": case_results,
    }


def benchmark_baseline(resources: DemoResources, cases: list[dict[str, Any]], fused_bundles: list[dict[str, Any]]) -> dict[str, Any]:
    case_results = []
    total_score = 0.0
    pass_count = 0
    citation_integrity_passes = 0
    for case, fused_bundle in zip(cases, fused_bundles):
        answer_eval = evaluate_answer_bundle(fused_bundle, resources)
        avg_score = float(answer_eval["judge_result"]["average_score"])
        passed = answer_eval["judge_result"]["metrics"]["overall_verdict"] == "pass"
        total_score += avg_score
        pass_count += int(passed)
        citation_integrity_passes += int(answer_eval["citation_integrity"])
        case_results.append(
            {
                "query": case["query"],
                "judge_average_score": avg_score,
                "judge_verdict": answer_eval["judge_result"]["metrics"]["overall_verdict"],
                "citation_integrity": answer_eval["citation_integrity"],
                "citation_count": answer_eval["citation_count"],
                "answer_preview": answer_eval["ui_answer_preview"],
            }
        )
    case_count = max(len(case_results), 1)
    return {
        "model": "baseline_no_compression",
        "summary": {
            "avg_judge_score": round(total_score / case_count, 3),
            "pass_rate": round(pass_count / case_count, 3),
            "citation_integrity_rate": round(citation_integrity_passes / case_count, 3),
            "avg_compression_latency_seconds": 0.0,
            "estimated_compression_cost_usd": 0.0,
        },
        "cases": case_results,
    }


def write_report(run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        PROJECT_ROOT
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "contextual_compression_benchmark.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark citation-preserving contextual compression models.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component8e_contextual_compression_model_benchmark_live")
    args = parser.parse_args()

    resources = DemoResources()
    resources.ensure_initialized()
    cases = load_cases(PROJECT_ROOT / args.input)
    fused_bundles = [build_fused_bundle(resources, case["query"]) for case in cases]

    baseline = benchmark_baseline(resources, cases, fused_bundles)
    model_results = [benchmark_model(resources, cases, model, fused_bundles) for model in DEFAULT_COMPRESSION_MODELS]
    ranked = sorted(
        model_results,
        key=lambda item: (
            -item["summary"]["avg_judge_score"],
            -item["summary"]["pass_rate"],
            -item["summary"]["citation_integrity_rate"],
            item["summary"]["estimated_compression_cost_usd"],
            item["summary"]["avg_compression_latency_seconds"],
        ),
    )
    result = {
        "baseline": baseline,
        "model_results": model_results,
        "ranked_models": [item["model"] for item in ranked],
        "recommended_model": ranked[0]["model"] if ranked else None,
    }
    report_path = write_report(args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "recommended_model": result["recommended_model"],
                "ranked_models": result["ranked_models"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
