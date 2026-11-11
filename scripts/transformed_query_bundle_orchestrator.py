import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.llm_query_decomposition import (
    MODEL_NAME,
    decompose_query_with_llm,
)
from agentic_document_intelligence.scripts.multi_query_generation import generate_multi_queries
from agentic_document_intelligence.scripts.step_back_query_generation import generate_step_back_query


MAX_SUB_QUERIES = 3
SPARSE_KEYWORD_PATTERN = re.compile(
    r"\b(azure|ai|revenue|growth|margin|income|capex|datacenter|ceo|cfo|risk|governance)\b",
    re.IGNORECASE,
)
AMBIGUOUS_REFERENCE_PATTERN = re.compile(
    r"\b(they|them|their|it|its|the company|company)\b",
    re.IGNORECASE,
)


def _cap_sub_queries(sub_queries: list[str]) -> list[str]:
    seen = set()
    capped = []
    for item in sub_queries:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        capped.append(cleaned)
        if len(capped) >= MAX_SUB_QUERIES:
            break
    return capped


def recommend_hyde(query: str) -> dict[str, Any]:
    normalized = query.strip()
    word_count = len(normalized.split())
    sparse_terms = SPARSE_KEYWORD_PATTERN.findall(normalized)
    has_ambiguous_refs = bool(AMBIGUOUS_REFERENCE_PATTERN.search(normalized))
    should_consider = (word_count <= 5 and bool(sparse_terms)) or (word_count <= 7 and has_ambiguous_refs)
    reasons = []
    if word_count <= 5 and sparse_terms:
        reasons.append("short_sparse_query")
    if has_ambiguous_refs:
        reasons.append("ambiguous_reference")
    return {
        "should_consider_hyde_after_weak_retrieval": should_consider,
        "candidate_reasons": reasons,
    }


def build_transformed_query_bundle(
    query: str,
    decomposition_fn: Callable[[str], dict[str, Any]] | None = None,
    multi_query_fn: Callable[[str], dict[str, Any]] | None = None,
    step_back_fn: Callable[[str], dict[str, Any]] | None = None,
    model: str = MODEL_NAME,
) -> dict[str, Any]:
    decomposition_callable = decomposition_fn or (lambda q: decompose_query_with_llm(q, model=model))
    multi_query_callable = multi_query_fn or (lambda q: generate_multi_queries(q, model=model))
    step_back_callable = step_back_fn or (lambda q: generate_step_back_query(q, model=model))

    decomposition_result = decomposition_callable(query)
    sub_queries = _cap_sub_queries(decomposition_result.get("sub_queries", [query]))

    sub_query_bundles = []
    for index, sub_query in enumerate(sub_queries, start=1):
        multi_query_result = multi_query_callable(sub_query)
        step_back_result = step_back_callable(sub_query)
        hyde_recommendation = recommend_hyde(sub_query)
        sub_query_bundles.append(
            {
                "sub_query_id": f"sq_{index}",
                "original_sub_query": sub_query,
                "multi_query_result": multi_query_result,
                "step_back_result": step_back_result,
                "hyde_recommendation": hyde_recommendation,
            }
        )

    return {
        "original_query": query,
        "policy": {
            "decomposition_model": model,
            "max_sub_queries": MAX_SUB_QUERIES,
            "run_hyde_in_initial_bundle": False,
            "hyde_policy": "candidate_only_after_weak_retrieval",
        },
        "decomposition_result": {
            **decomposition_result,
            "sub_queries": sub_queries,
        },
        "sub_query_bundles": sub_query_bundles,
        "bundle_summary": {
            "sub_query_count": len(sub_query_bundles),
            "total_multi_query_rewrites": sum(
                bundle["multi_query_result"]["rewrite_count"] for bundle in sub_query_bundles
            ),
            "step_back_count": len(sub_query_bundles),
            "hyde_candidate_count": sum(
                1
                for bundle in sub_query_bundles
                if bundle["hyde_recommendation"]["should_consider_hyde_after_weak_retrieval"]
            ),
        },
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
    report_path = output_dir / "transformed_query_bundle_report.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a transformed query bundle.")
    parser.add_argument(
        "--query",
        default="Who is the CEO and what segment drove the most revenue growth and what did they say about AI?",
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_transformed_query_bundle")
    args = parser.parse_args()

    result = build_transformed_query_bundle(args.query, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "sub_query_count": result["bundle_summary"]["sub_query_count"],
                "total_multi_query_rewrites": result["bundle_summary"]["total_multi_query_rewrites"],
                "hyde_candidate_count": result["bundle_summary"]["hyde_candidate_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
