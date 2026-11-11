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

from agentic_document_intelligence.scripts.pinecone_hybrid_retrieval import (
    DENSE_MODEL,
    SPARSE_MODEL,
    build_chunk_indexes,
    dedupe_matches,
    embed_dense_query,
    embed_sparse_query,
    hydrate_matches,
    load_chunk_artifact,
    run_hybrid_query,
    scale_dense,
    scale_sparse,
)
from agentic_document_intelligence.scripts.transformed_query_bundle_orchestrator import (
    MODEL_NAME,
    build_transformed_query_bundle,
)


DEFAULT_CHUNK_INPUT = (
    "artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json"
)
DEFAULT_NAMESPACE = "microsoft_fy2025_fixed_corpus"
DEFAULT_TOP_K = 6
DEFAULT_ALPHA = 0.6


def build_query_variants(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    variants = []
    for sub_bundle in bundle["sub_query_bundles"]:
        sub_query_id = sub_bundle["sub_query_id"]
        original_sub_query = sub_bundle["original_sub_query"]
        variants.append(
            {
                "sub_query_id": sub_query_id,
                "variant_id": f"{sub_query_id}_original",
                "variant_type": "original_sub_query",
                "query_text": original_sub_query,
                "query_angle": "original",
            }
        )
        for index, rewrite in enumerate(sub_bundle["multi_query_result"]["rewrites"], start=1):
            variants.append(
                {
                    "sub_query_id": sub_query_id,
                    "variant_id": f"{sub_query_id}_mq_{index}",
                    "variant_type": "multi_query",
                    "query_text": rewrite["query"],
                    "query_angle": rewrite["angle"],
                }
            )
        variants.append(
            {
                "sub_query_id": sub_query_id,
                "variant_id": f"{sub_query_id}_step_back",
                "variant_type": "step_back",
                "query_text": sub_bundle["step_back_result"]["step_back_query"],
                "query_angle": sub_bundle["step_back_result"]["broadening_strategy"],
            }
        )
    return variants


def execute_variant_retrieval(
    variant: dict[str, Any],
    index: Any,
    namespace: str,
    alpha: float,
    top_k: int,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
    child_index: dict[str, dict[str, Any]],
    parent_index: dict[str, str],
) -> dict[str, Any]:
    dense = embed_dense_query(openai_client, variant["query_text"])
    sparse = embed_sparse_query(pinecone_client, variant["query_text"])
    dense_scaled = scale_dense(dense, alpha)
    sparse_scaled = scale_sparse(sparse["indices"], sparse["values"], alpha)
    raw_matches = run_hybrid_query(index, namespace, dense_scaled, sparse_scaled, top_k)
    deduped_matches = dedupe_matches(raw_matches)
    hydrated_matches = hydrate_matches(deduped_matches, child_index, parent_index)
    for match in hydrated_matches:
        match["provenance"] = {
            "sub_query_id": variant["sub_query_id"],
            "variant_id": variant["variant_id"],
            "variant_type": variant["variant_type"],
            "query_text": variant["query_text"],
            "query_angle": variant["query_angle"],
        }
    return {
        "variant": variant,
        "raw_match_count": len(raw_matches),
        "deduped_match_count": len(deduped_matches),
        "matches": hydrated_matches,
    }


def execute_transformed_retrieval(
    bundle: dict[str, Any],
    index: Any,
    namespace: str,
    alpha: float,
    top_k: int,
    openai_client: OpenAI,
    pinecone_client: Pinecone,
    child_index: dict[str, dict[str, Any]],
    parent_index: dict[str, str],
    variant_executor: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    executor = variant_executor or execute_variant_retrieval
    variants = build_query_variants(bundle)
    variant_results = [
        executor(
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
        for variant in variants
    ]
    return {
        "original_query": bundle["original_query"],
        "policy": bundle["policy"],
        "retrieval_summary": {
            "sub_query_count": bundle["bundle_summary"]["sub_query_count"],
            "variant_count": len(variants),
            "retrieval_call_count": len(variants),
            "total_deduped_matches": sum(item["deduped_match_count"] for item in variant_results),
        },
        "variant_results": variant_results,
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
    report_path = output_dir / "transformed_retrieval_executor_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute hybrid retrieval over a transformed query bundle.")
    parser.add_argument(
        "--query",
        default="Who is the CEO and what segment drove the most revenue growth and what did they say about AI?",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--chunk-input", default=DEFAULT_CHUNK_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--run-id", default="component4_transformed_retrieval_executor")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env")
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    bundle = build_transformed_query_bundle(args.query, model=args.model)
    chunk_artifact = load_chunk_artifact(PROJECT_ROOT / "agentic_document_intelligence" / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)

    openai_client = OpenAI(api_key=openai_api_key)
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    index = pinecone_client.Index(index_name)

    result = execute_transformed_retrieval(
        bundle,
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
                "variant_count": result["retrieval_summary"]["variant_count"],
                "total_deduped_matches": result["retrieval_summary"]["total_deduped_matches"],
                "dense_model": DENSE_MODEL,
                "sparse_model": SPARSE_MODEL,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
