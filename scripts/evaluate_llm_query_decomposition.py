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

from agentic_document_intelligence.scripts.llm_query_decomposition import (
    MODEL_NAME,
    decompose_query_with_llm,
)
from agentic_document_intelligence.scripts.query_decomposition import decompose_query


DEFAULT_EVAL_CASES_PATH = (
    PROJECT_ROOT
    / "agentic_document_intelligence"
    / "evals"
    / "query_decomposition_cases.json"
)


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[?.!,;:]+$", "", normalized)
    normalized = normalized.replace("fiscal year 2025", "fy2025")
    normalized = normalized.replace("fiscal 2025", "fy2025")
    normalized = normalized.replace("microsoft azure", "azure")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def canonical_signature(text: str) -> str:
    normalized = normalize_text(text)
    rules = [
        (r"((who is|identify|tell me).*(ceo).*(microsoft|microsoft's))|((who is|identify|tell me).*(microsoft|microsoft's).*(ceo))|((who is|identify|tell me).*(ceo))", "microsoft_ceo"),
        (r"(revenue growth).*(fy2025)|fy2025.*revenue growth", "microsoft_fy2025_revenue_growth"),
        (r"(revenue).*(fy2025)|fy2025.*revenue", "microsoft_fy2025_revenue"),
        (r"(operating income).*(fy2025)|fy2025.*operating income", "microsoft_fy2025_operating_income"),
        (r"(net income).*(fy2025)|fy2025.*net income", "microsoft_fy2025_net_income"),
        (r"(cash flow).*(fy2025)|fy2025.*cash flow", "microsoft_fy2025_cash_flow"),
        (r"cybersecurity risks", "microsoft_fy2025_cybersecurity_risks"),
        (r"azure growth", "azure_growth"),
        (r"ai demand", "ai_demand"),
        (r"compare.*azure.*microsoft 365.*revenue drivers|compare.*revenue drivers.*azure.*microsoft 365", "compare_azure_m365_revenue_drivers"),
    ]
    for pattern, signature in rules:
        if re.search(pattern, normalized):
            return signature
    return normalized


def evaluate_case(query: str, actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    actual_queries = [normalize_text(item) for item in actual.get("sub_queries", [])]
    expected_queries = [normalize_text(item) for item in expected.get("sub_queries", [])]
    actual_signatures = [canonical_signature(item) for item in actual.get("sub_queries", [])]
    expected_signatures = [canonical_signature(item) for item in expected.get("sub_queries", [])]

    missing = [item for item in expected_queries if item not in actual_queries]
    extra = [item for item in actual_queries if item not in expected_queries]
    semantic_missing = [item for item in expected_signatures if item not in actual_signatures]
    semantic_extra = [item for item in actual_signatures if item not in expected_signatures]
    exact_pass_flag = (
        actual.get("needs_decomposition") == expected.get("needs_decomposition")
        and not missing
        and not extra
    )
    semantic_pass_flag = (
        actual.get("needs_decomposition") == expected.get("needs_decomposition")
        and not semantic_missing
        and not semantic_extra
    )

    return {
        "query": query,
        "passed": semantic_pass_flag,
        "exact_passed": exact_pass_flag,
        "missing_sub_queries": missing,
        "extra_sub_queries": extra,
        "semantic_missing_sub_queries": semantic_missing,
        "semantic_extra_sub_queries": semantic_extra,
        "expected_needs_decomposition": expected.get("needs_decomposition"),
        "actual_needs_decomposition": actual.get("needs_decomposition"),
    }


def load_eval_cases(cases_path: Path) -> list[dict[str, Any]]:
    return json.loads(cases_path.read_text(encoding="utf-8"))


def write_report(run_id: str, payload: dict[str, Any]) -> Path:
    output_dir = (
        PROJECT_ROOT
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "evaluations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "llm_query_decomposition_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gpt-5-mini query decomposition quality.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_llm_query_decomposition_eval")
    parser.add_argument("--cases-path", default=str(DEFAULT_EVAL_CASES_PATH))
    args = parser.parse_args()

    cases_path = Path(args.cases_path)
    cases = load_eval_cases(cases_path)
    results = []
    baseline_results = []

    for case in cases:
        query = case["query"]
        expected = case["expected"]
        llm_result = decompose_query_with_llm(query, model=args.model)
        baseline_result = decompose_query(query)
        results.append(
            {
                "query": query,
                "llm_result": llm_result,
                "evaluation": evaluate_case(query, llm_result, expected),
            }
        )
        baseline_results.append(
            {
                "query": query,
                "baseline_result": baseline_result,
                "evaluation": evaluate_case(query, baseline_result, expected),
            }
        )

    passed_count = sum(item["evaluation"]["passed"] for item in results)
    exact_passed_count = sum(item["evaluation"]["exact_passed"] for item in results)
    baseline_passed_count = sum(item["evaluation"]["passed"] for item in baseline_results)
    baseline_exact_passed_count = sum(item["evaluation"]["exact_passed"] for item in baseline_results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "cases_path": str(cases_path),
        "case_count": len(cases),
        "passed_count": passed_count,
        "exact_passed_count": exact_passed_count,
        "baseline_passed_count": baseline_passed_count,
        "baseline_exact_passed_count": baseline_exact_passed_count,
        "results": results,
        "baseline_results": baseline_results,
    }
    report_path = write_report(args.run_id, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(cases),
                "passed_count": passed_count,
                "exact_passed_count": exact_passed_count,
                "baseline_passed_count": baseline_passed_count,
                "baseline_exact_passed_count": baseline_exact_passed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
