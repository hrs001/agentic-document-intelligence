import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = (
    "artifacts/experiments/component4_transformed_retrieval_executor_live/retrieval/"
    "transformed_retrieval_executor_report.json"
)


def load_executor_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def _base_merged_match(match: dict[str, Any]) -> dict[str, Any]:
    provenance = match["provenance"]
    return {
        "id": match["id"],
        "source_chunk_id": match["metadata"]["source_chunk_id"],
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


def merge_variant_results(result: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    original_sub_queries: dict[str, str] = {}

    for variant_result in result["variant_results"]:
        variant = variant_result["variant"]
        sub_query_id = variant["sub_query_id"]
        if variant["variant_type"] == "original_sub_query":
            original_sub_queries[sub_query_id] = variant["query_text"]

        for match in variant_result["matches"]:
            source_chunk_id = match["metadata"]["source_chunk_id"]
            existing = grouped[sub_query_id].get(source_chunk_id)
            if existing is None:
                grouped[sub_query_id][source_chunk_id] = _base_merged_match(match)
                continue

            existing["match_count"] += 1
            existing["provenance_list"].append(match["provenance"])
            if match["provenance"]["variant_type"] not in existing["variant_types"]:
                existing["variant_types"].append(match["provenance"]["variant_type"])
            if match["provenance"]["query_angle"] not in existing["query_angles"]:
                existing["query_angles"].append(match["provenance"]["query_angle"])
            if match["provenance"]["query_text"] not in existing["matched_query_texts"]:
                existing["matched_query_texts"].append(match["provenance"]["query_text"])
            if match["score"] > existing["best_score"]:
                existing["best_score"] = match["score"]
                existing["id"] = match["id"]
                existing["metadata"] = match["metadata"]
                existing["child_text"] = match["child_text"]
                existing["parent_text"] = match["parent_text"]

    merged_sub_queries = []
    for sub_query_id, matches_by_chunk in grouped.items():
        merged_matches = sorted(
            matches_by_chunk.values(),
            key=lambda item: (item["best_score"], item["match_count"]),
            reverse=True,
        )
        merged_sub_queries.append(
            {
                "sub_query_id": sub_query_id,
                "original_sub_query": original_sub_queries.get(sub_query_id, ""),
                "merged_match_count": len(merged_matches),
                "merged_matches": merged_matches,
            }
        )

    merged_sub_queries.sort(key=lambda item: item["sub_query_id"])
    total_variant_matches = sum(
        len(variant_result["matches"]) for variant_result in result["variant_results"]
    )
    total_merged_matches = sum(
        item["merged_match_count"] for item in merged_sub_queries
    )

    return {
        "original_query": result["original_query"],
        "policy": result["policy"],
        "merge_summary": {
            "sub_query_count": len(merged_sub_queries),
            "variant_count": result["retrieval_summary"]["variant_count"],
            "total_variant_matches": total_variant_matches,
            "total_merged_matches": total_merged_matches,
            "duplicates_removed": total_variant_matches - total_merged_matches,
        },
        "sub_query_results": merged_sub_queries,
    }


def write_report(project_root: Path, run_id: str, merged_result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "retrieval_merge_dedup_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": merged_result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate retrieval results across query variants.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component4_retrieval_merge_dedup")
    args = parser.parse_args()

    result = load_executor_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    merged_result = merge_variant_results(result)
    report_path = write_report(PROJECT_ROOT, args.run_id, merged_result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": merged_result["merge_summary"]["sub_query_count"],
                "total_variant_matches": merged_result["merge_summary"]["total_variant_matches"],
                "total_merged_matches": merged_result["merge_summary"]["total_merged_matches"],
                "duplicates_removed": merged_result["merge_summary"]["duplicates_removed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
