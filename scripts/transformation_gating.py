import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MAX_SUB_QUERIES = 3

COMPARISON_PATTERN = re.compile(r"\b(compare|compared with|versus|vs\.?|difference between)\b", re.IGNORECASE)
EXPLANATION_PATTERN = re.compile(
    r"\b(why|how|explain|drivers|driver|trend|trends|context|governance|responsibilities|overview|strategy)\b",
    re.IGNORECASE,
)
MULTI_INTENT_PATTERN = re.compile(
    r"\b(and|also|plus|as well as)\b",
    re.IGNORECASE,
)
LIST_PATTERN = re.compile(
    r",\s*|\band\b",
    re.IGNORECASE,
)
SHORT_QUERY_PATTERN = re.compile(r"^\s*\S+(?:\s+\S+){0,4}\s*[?]?\s*$")
AMBIGUOUS_REFERENCE_PATTERN = re.compile(
    r"\b(the company|company|they|them|their|it|its|those|these|management|executives|board)\b",
    re.IGNORECASE,
)
SPARSE_QUERY_PATTERN = re.compile(
    r"\b(revenue|growth|margin|income|capex|datacenter|ai|azure|ceo|cfo|risk|governance)\b",
    re.IGNORECASE,
)


def estimate_sub_query_count(query: str) -> int:
    if COMPARISON_PATTERN.search(query):
        return 1

    parts = [part.strip(" .?") for part in re.split(r"\?|,", query) if part.strip(" .?")]
    if len(parts) > 1:
        return min(len(parts), MAX_SUB_QUERIES)

    if MULTI_INTENT_PATTERN.search(query):
        segments = [part.strip(" .?") for part in re.split(r"\band\b|\balso\b|\bplus\b|\bas well as\b", query, flags=re.IGNORECASE) if part.strip(" .?")]
        return min(max(1, len(segments)), MAX_SUB_QUERIES)

    return 1


def classify_query(query: str) -> dict[str, Any]:
    normalized = query.strip()
    estimated_sub_queries = estimate_sub_query_count(normalized)
    is_comparison = bool(COMPARISON_PATTERN.search(normalized))
    is_explanatory = bool(EXPLANATION_PATTERN.search(normalized))
    has_ambiguous_refs = bool(AMBIGUOUS_REFERENCE_PATTERN.search(normalized))
    is_short = bool(SHORT_QUERY_PATTERN.match(normalized))
    is_sparse_keyword_query = bool(SPARSE_QUERY_PATTERN.search(normalized)) and is_short
    has_multi_intent = estimated_sub_queries > 1

    ambiguity_signals = []
    if has_ambiguous_refs:
        ambiguity_signals.append("ambiguous_reference")
    if has_multi_intent and is_short:
        ambiguity_signals.append("short_but_multi_intent")
    if is_explanatory and has_multi_intent:
        ambiguity_signals.append("mixed_explanatory_multi_intent")
    if not has_multi_intent and MULTI_INTENT_PATTERN.search(normalized) and not is_comparison:
        ambiguity_signals.append("possible_hidden_multi_intent")

    return {
        "is_comparison": is_comparison,
        "is_explanatory": is_explanatory,
        "has_ambiguous_refs": has_ambiguous_refs,
        "is_short": is_short,
        "is_sparse_keyword_query": is_sparse_keyword_query,
        "estimated_sub_queries": estimated_sub_queries,
        "ambiguity_signals": ambiguity_signals,
        "is_ambiguous_case": bool(ambiguity_signals),
    }


def select_transformations(query: str) -> dict[str, Any]:
    features = classify_query(query)
    reasons: list[str] = []

    run_decomposition = features["estimated_sub_queries"] > 1
    if run_decomposition:
        reasons.append("compound_or_multi_intent")

    run_multi_query = features["is_short"] or features["is_sparse_keyword_query"] or features["is_comparison"]
    if run_multi_query:
        reasons.append("needs_retrieval_variation")

    run_step_back = features["is_explanatory"] or features["is_comparison"]
    if run_step_back:
        reasons.append("needs_broader_context")

    run_hyde = features["is_sparse_keyword_query"] or (features["is_short"] and not features["is_comparison"])
    if run_hyde:
        reasons.append("sparse_query_support")

    return {
        "query": query,
        "max_sub_queries": MAX_SUB_QUERIES,
        "estimated_sub_queries": min(features["estimated_sub_queries"], MAX_SUB_QUERIES),
        "run_decomposition": run_decomposition,
        "run_multi_query": run_multi_query,
        "run_step_back": run_step_back,
        "run_hyde": run_hyde,
        "is_ambiguous_case": features["is_ambiguous_case"],
        "ambiguity_signals": features["ambiguity_signals"],
        "reasons": sorted(set(reasons)),
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "query_transforms"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "transformation_gating_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic transformation gating.")
    parser.add_argument("--query", default="What happened with Azure growth and what did management say about AI demand?")
    parser.add_argument("--run-id", default="component3_transformation_gating")
    args = parser.parse_args()

    result = select_transformations(args.query)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "is_ambiguous_case": result["is_ambiguous_case"],
                "estimated_sub_queries": result["estimated_sub_queries"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
