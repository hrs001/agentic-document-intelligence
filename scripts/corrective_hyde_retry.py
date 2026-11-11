import argparse
import json
import os
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

from agentic_document_intelligence.scripts.hyde_query_generation import (
    MODEL_NAME,
    generate_hyde_passage,
)
from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import (
    build_chunk_indexes,
    load_chunk_artifact,
)
from agentic_document_intelligence.scripts.sub_query_coverage_scoring import load_merge_report
from agentic_document_intelligence.scripts.transformed_retrieval_executor import (
    DEFAULT_ALPHA,
    DEFAULT_CHUNK_INPUT,
    DEFAULT_NAMESPACE,
    DEFAULT_TOP_K,
    execute_variant_retrieval,
)


DEFAULT_COVERAGE_INPUT = (
    "artifacts/experiments/component4_sub_query_coverage_live/retrieval/sub_query_coverage_report.json"
)
DEFAULT_MERGE_INPUT = (
    "artifacts/experiments/component4_retrieval_merge_dedup_live/retrieval/retrieval_merge_dedup_report.json"
)


def load_coverage_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def should_trigger_hyde(score: dict[str, Any]) -> tuple[bool, list[str]]:
    breakdown = score["signal_breakdown"]
    reasons: list[str] = []
    if score["coverage_label"] == "weak":
        reasons.append("coverage_label_weak")
    if breakdown["query_overlap_ratio"] < 0.2 and breakdown["top_score"] < 2.2:
        reasons.append("very_low_overlap_and_unconvincing_score")
    if breakdown["query_overlap_ratio"] < 0.12 and breakdown["reinforced_match_strength"] < 0.34:
        reasons.append("low_overlap_and_low_reinforcement")
    if breakdown["score_strength"] < 0.35 and breakdown["result_depth_strength"] < 0.4:
        reasons.append("shallow_low_score_result_set")
    return bool(reasons), reasons


def merge_retry_matches(existing_sub_query_result: dict[str, Any], retry_matches: list[dict[str, Any]]) -> dict[str, Any]:
    merged_by_chunk = {
        item["source_chunk_id"]: {
            **item,
            "provenance_list": list(item["provenance_list"]),
            "variant_types": list(item["variant_types"]),
            "query_angles": list(item["query_angles"]),
            "matched_query_texts": list(item["matched_query_texts"]),
        }
        for item in existing_sub_query_result["merged_matches"]
    }

    for match in retry_matches:
        source_chunk_id = match["metadata"]["source_chunk_id"]
        provenance = match["provenance"]
        existing = merged_by_chunk.get(source_chunk_id)
        if existing is None:
            merged_by_chunk[source_chunk_id] = {
                "id": match["id"],
                "source_chunk_id": source_chunk_id,
                "best_score": match["score"],
                "match_count": 1,
                "metadata": match["metadata"],
                "child_text": match["child_text"],
                "parent_text": match["parent_text"],
                "provenance_list": [provenance],
                "variant_types": [provenance["variant_type"]],
                "query_angles": [provenance["query_angle"]],
                "matched_query_texts": [provenance["query_text"]],
            }
            continue

        existing["match_count"] += 1
        existing["provenance_list"].append(provenance)
        if provenance["variant_type"] not in existing["variant_types"]:
            existing["variant_types"].append(provenance["variant_type"])
        if provenance["query_angle"] not in existing["query_angles"]:
            existing["query_angles"].append(provenance["query_angle"])
        if provenance["query_text"] not in existing["matched_query_texts"]:
            existing["matched_query_texts"].append(provenance["query_text"])
        if match["score"] > existing["best_score"]:
            existing["best_score"] = match["score"]
            existing["id"] = match["id"]
            existing["metadata"] = match["metadata"]
            existing["child_text"] = match["child_text"]
            existing["parent_text"] = match["parent_text"]

    merged_matches = sorted(
        merged_by_chunk.values(),
        key=lambda item: (item["best_score"], item["match_count"]),
        reverse=True,
    )
    return {
        **existing_sub_query_result,
        "merged_match_count": len(merged_matches),
        "merged_matches": merged_matches,
    }


def run_corrective_hyde_retry(
    coverage_result: dict[str, Any],
    merge_result: dict[str, Any],
    index: Any,
    namespace: str,
    alpha: float,
    top_k: int,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
    child_index: dict[str, dict[str, Any]],
    parent_index: dict[str, str],
    hyde_generator: Callable[[str], dict[str, Any]] | None = None,
    retry_executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generator = hyde_generator or (lambda q: generate_hyde_passage(q, model=MODEL_NAME, client=openai_client))
    executor = retry_executor or execute_variant_retrieval
    merge_by_sub_query = {
        item["sub_query_id"]: item for item in merge_result["sub_query_results"]
    }

    retry_results = []
    updated_sub_query_results = []
    triggered_sub_query_ids: list[str] = []

    for score in coverage_result["sub_query_scores"]:
        sub_query_id = score["sub_query_id"]
        existing = merge_by_sub_query[sub_query_id]
        should_retry, reasons = should_trigger_hyde(score)
        if not should_retry:
            updated_sub_query_results.append(existing)
            retry_results.append(
                {
                    "sub_query_id": sub_query_id,
                    "original_sub_query": score["original_sub_query"],
                    "triggered_hyde": False,
                    "trigger_reasons": reasons,
                }
            )
            continue

        triggered_sub_query_ids.append(sub_query_id)
        hyde_result = generator(score["original_sub_query"])
        variant = {
            "sub_query_id": sub_query_id,
            "variant_id": f"{sub_query_id}_hyde",
            "variant_type": "hyde",
            "query_text": hyde_result["hypothetical_passage"],
            "query_angle": hyde_result["generation_style"],
        }
        retry_result = executor(
            variant,
            index,
            namespace,
            alpha,
            top_k,
            openai_client,
            pinecone_client,
            child_index,
            parent_index,
        )
        updated = merge_retry_matches(existing, retry_result["matches"])
        updated_sub_query_results.append(updated)
        retry_results.append(
            {
                "sub_query_id": sub_query_id,
                "original_sub_query": score["original_sub_query"],
                "triggered_hyde": True,
                "trigger_reasons": reasons,
                "hyde_result": hyde_result,
                "retry_match_count": retry_result["deduped_match_count"],
            }
        )

    updated_sub_query_results.sort(key=lambda item: item["sub_query_id"])
    return {
        "original_query": merge_result["original_query"],
        "policy": {
            **merge_result["policy"],
            "corrective_hyde_model": MODEL_NAME,
            "hyde_retry_policy": "conservative_retrieval_feedback",
        },
        "retry_summary": {
            "sub_query_count": len(coverage_result["sub_query_scores"]),
            "triggered_hyde_count": len(triggered_sub_query_ids),
            "triggered_sub_query_ids": triggered_sub_query_ids,
        },
        "retry_results": retry_results,
        "updated_sub_query_results": updated_sub_query_results,
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "corrective_hyde_retry_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run corrective HyDE only for weak retrieval cases.")
    parser.add_argument("--coverage-input", default=DEFAULT_COVERAGE_INPUT)
    parser.add_argument("--merge-input", default=DEFAULT_MERGE_INPUT)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--run-id", default="component4_corrective_hyde_retry")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env")
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    coverage_result = load_coverage_report(PROJECT_ROOT / "agentic_document_intelligence" / args.coverage_input)
    merge_result = load_merge_report(PROJECT_ROOT / "agentic_document_intelligence" / args.merge_input)
    chunk_artifact = load_chunk_artifact(PROJECT_ROOT / "agentic_document_intelligence" / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)

    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index(index_name)

    result = run_corrective_hyde_retry(
        coverage_result,
        merge_result,
        index,
        args.namespace,
        args.alpha,
        args.top_k,
        openai_client,
        pinecone_client,
        child_index,
        parent_index,
    )
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "triggered_hyde_count": result["retry_summary"]["triggered_hyde_count"],
                "triggered_sub_query_ids": result["retry_summary"]["triggered_sub_query_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
