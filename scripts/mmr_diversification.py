import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RERANK_INPUT = (
    "artifacts/experiments/component4_sub_query_rerank_live/retrieval/"
    "rerank_sub_query_candidates_report.json"
)
DEFAULT_EMBEDDING_INPUT = (
    "artifacts/experiments/component2_embedding_ready_records/embeddings/"
    "microsoft_fy2025_10k_summary_embedding_records.json"
)
DEFAULT_NAMESPACE = "microsoft_fy2025_fixed_corpus"
DEFAULT_TOP_M = 4
DEFAULT_LAMBDA = 0.75


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def load_embedding_records(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        record["source_chunk_id"]: record["record_id"]
        for record in payload["records"]
    }


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def fetch_candidate_vectors(
    pc_index: Any,
    namespace: str,
    matches: list[dict[str, Any]],
    chunk_to_record_id: dict[str, str],
) -> dict[str, list[float]]:
    ids = []
    match_id_to_record_id: dict[str, str] = {}
    for match in matches:
        record_id = chunk_to_record_id.get(match["source_chunk_id"])
        if not record_id:
            continue
        ids.append(record_id)
        match_id_to_record_id[match["id"]] = record_id

    vectors_by_match_id: dict[str, list[float]] = {}
    for start in range(0, len(ids), 100):
        batch_ids = ids[start : start + 100]
        fetched = pc_index.fetch(ids=batch_ids, namespace=namespace)
        vectors = getattr(fetched, "vectors", {}) or {}
        for match_id, record_id in match_id_to_record_id.items():
            if record_id in vectors:
                values = getattr(vectors[record_id], "values", None)
                if values is not None:
                    vectors_by_match_id[match_id] = list(values)
    return vectors_by_match_id


def select_with_mmr(
    matches: list[dict[str, Any]],
    vectors_by_match_id: dict[str, list[float]],
    top_m: int,
    lambda_weight: float,
) -> list[dict[str, Any]]:
    candidates = [match for match in matches if match["id"] in vectors_by_match_id]
    if not candidates:
        return []

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)

    # Start with the highest rerank score.
    remaining.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < top_m:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(remaining):
            relevance = candidate.get("rerank_score", 0.0)
            candidate_vector = vectors_by_match_id[candidate["id"]]
            diversity_penalty = max(
                cosine_similarity(candidate_vector, vectors_by_match_id[selected_item["id"]])
                for selected_item in selected
            )
            mmr_score = (lambda_weight * relevance) - ((1.0 - lambda_weight) * diversity_penalty)
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        chosen = remaining.pop(best_index)
        enriched = dict(chosen)
        enriched["mmr_score"] = round(best_score, 6)
        selected.append(enriched)

    # Ensure the first selected item also exposes an mmr score for consistency.
    if "mmr_score" not in selected[0]:
        selected[0] = {**selected[0], "mmr_score": round(selected[0].get("rerank_score", 0.0), 6)}

    return selected


def diversify_sub_query_results(
    result: dict[str, Any],
    pc_index: Any,
    namespace: str,
    chunk_to_record_id: dict[str, str],
    top_m: int,
    lambda_weight: float,
) -> dict[str, Any]:
    diversified_results = []
    for sub_query_result in result["sub_query_results"]:
        reranked_matches = sub_query_result["reranked_matches"]
        vectors_by_match_id = fetch_candidate_vectors(
            pc_index,
            namespace,
            reranked_matches,
            chunk_to_record_id,
        )
        diversified = select_with_mmr(
            reranked_matches,
            vectors_by_match_id,
            top_m=top_m,
            lambda_weight=lambda_weight,
        )
        diversified_results.append(
            {
                "sub_query_id": sub_query_result["sub_query_id"],
                "original_sub_query": sub_query_result["original_sub_query"],
                "candidate_count": sub_query_result["candidate_count"],
                "reranked_count": sub_query_result["reranked_count"],
                "diversified_count": len(diversified),
                "diversified_matches": diversified,
            }
        )

    return {
        "original_query": result["original_query"],
        "policy": {
            **result["policy"],
            "mmr_lambda": lambda_weight,
            "mmr_top_m": top_m,
        },
        "mmr_summary": {
            "sub_query_count": len(diversified_results),
            "top_m": top_m,
            "lambda_weight": lambda_weight,
        },
        "sub_query_results": diversified_results,
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
    report_path = output_dir / "mmr_diversification_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply MMR diversification to reranked sub-query candidates.")
    parser.add_argument("--rerank-input", default=DEFAULT_RERANK_INPUT)
    parser.add_argument("--embedding-input", default=DEFAULT_EMBEDDING_INPUT)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--top-m", type=int, default=DEFAULT_TOP_M)
    parser.add_argument("--lambda-weight", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--run-id", default="component4_mmr_diversification")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env")
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    result = load_report(PROJECT_ROOT / "agentic_document_intelligence" / args.rerank_input)
    chunk_to_record_id = load_embedding_records(PROJECT_ROOT / "agentic_document_intelligence" / args.embedding_input)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
    diversified = diversify_sub_query_results(
        result,
        index,
        args.namespace,
        chunk_to_record_id,
        top_m=args.top_m,
        lambda_weight=args.lambda_weight,
    )
    report_path = write_report(PROJECT_ROOT, args.run_id, diversified)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": diversified["mmr_summary"]["sub_query_count"],
                "top_m": diversified["mmr_summary"]["top_m"],
                "lambda_weight": diversified["mmr_summary"]["lambda_weight"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
