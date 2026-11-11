import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = (
    "artifacts/experiments/component4_retrieval_merge_dedup_live/retrieval/"
    "retrieval_merge_dedup_report.json"
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "say",
    "said",
    "the",
    "they",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "why",
}


def load_merge_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"]


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-\+]*", text.lower())
        if token not in STOPWORDS
    ]


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def compute_overlap_ratio(query: str, matches: list[dict[str, Any]], top_n: int = 3) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    evidence_text = " ".join(
        f"{match['metadata'].get('section_title', '')} {match['child_text']}"
        for match in matches[:top_n]
    )
    evidence_tokens = set(tokenize(evidence_text))
    return _safe_divide(len(query_tokens & evidence_tokens), len(query_tokens))


def normalize_top_score(top_score: float) -> float:
    if top_score <= 0:
        return 0.0
    return min(top_score / 3.0, 1.0)


def normalize_match_strength(matches: list[dict[str, Any]]) -> float:
    if not matches:
        return 0.0
    reinforced_hits = sum(1 for match in matches[:5] if match["match_count"] >= 2)
    return min(reinforced_hits / 3.0, 1.0)


def normalize_result_depth(match_count: int) -> float:
    return min(match_count / 5.0, 1.0)


def score_sub_query_coverage(sub_query_result: dict[str, Any]) -> dict[str, Any]:
    matches = sub_query_result["merged_matches"]
    top_score = matches[0]["best_score"] if matches else 0.0
    overlap_ratio = compute_overlap_ratio(sub_query_result["original_sub_query"], matches)
    score_strength = normalize_top_score(top_score)
    match_strength = normalize_match_strength(matches)
    depth_strength = normalize_result_depth(sub_query_result["merged_match_count"])

    weighted_score = (
        (0.45 * score_strength)
        + (0.35 * overlap_ratio)
        + (0.10 * match_strength)
        + (0.10 * depth_strength)
    )
    weighted_score = round(weighted_score, 4)

    if weighted_score >= 0.72:
        coverage_label = "strong"
    elif weighted_score >= 0.48:
        coverage_label = "moderate"
    else:
        coverage_label = "weak"

    return {
        "sub_query_id": sub_query_result["sub_query_id"],
        "original_sub_query": sub_query_result["original_sub_query"],
        "coverage_score": weighted_score,
        "coverage_label": coverage_label,
        "should_consider_hyde": coverage_label == "weak",
        "signal_breakdown": {
            "top_score": round(top_score, 4),
            "score_strength": round(score_strength, 4),
            "query_overlap_ratio": round(overlap_ratio, 4),
            "reinforced_match_strength": round(match_strength, 4),
            "result_depth_strength": round(depth_strength, 4),
        },
        "top_match_ids": [match["source_chunk_id"] for match in matches[:3]],
    }


def score_coverage(result: dict[str, Any]) -> dict[str, Any]:
    sub_query_scores = [
        score_sub_query_coverage(sub_query_result)
        for sub_query_result in result["sub_query_results"]
    ]
    weak_sub_queries = [item["sub_query_id"] for item in sub_query_scores if item["should_consider_hyde"]]
    average_score = _safe_divide(
        sum(item["coverage_score"] for item in sub_query_scores),
        len(sub_query_scores),
    )
    return {
        "original_query": result["original_query"],
        "policy": result["policy"],
        "coverage_summary": {
            "sub_query_count": len(sub_query_scores),
            "average_coverage_score": round(average_score, 4),
            "weak_sub_query_count": len(weak_sub_queries),
            "weak_sub_query_ids": weak_sub_queries,
        },
        "sub_query_scores": sub_query_scores,
    }


def write_report(project_root: Path, run_id: str, scored_result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sub_query_coverage_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": scored_result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sub-query coverage quality from merged retrieval results.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component4_sub_query_coverage")
    args = parser.parse_args()

    result = load_merge_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    scored_result = score_coverage(result)
    report_path = write_report(PROJECT_ROOT, args.run_id, scored_result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": scored_result["coverage_summary"]["sub_query_count"],
                "average_coverage_score": scored_result["coverage_summary"]["average_coverage_score"],
                "weak_sub_query_ids": scored_result["coverage_summary"]["weak_sub_query_ids"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
