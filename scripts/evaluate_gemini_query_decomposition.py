import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_llm_query_decomposition import (
    DEFAULT_EVAL_CASES_PATH,
    evaluate_case,
    load_eval_cases,
)
from agentic_document_intelligence.scripts.gemini_query_decomposition import (
    MODEL_NAME,
    decompose_query_with_gemini,
)
from agentic_document_intelligence.scripts.query_decomposition import decompose_query


def write_report(run_id: str, payload: dict) -> Path:
    output_dir = (
        PROJECT_ROOT
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "evaluations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "gemini_query_decomposition_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gemini query decomposition quality.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_gemini_query_decomposition_eval")
    parser.add_argument("--cases-path", default=str(DEFAULT_EVAL_CASES_PATH))
    args = parser.parse_args()

    cases_path = Path(args.cases_path)
    cases = load_eval_cases(cases_path)
    results = []
    baseline_results = []

    for case in cases:
        query = case["query"]
        expected = case["expected"]
        gemini_result = decompose_query_with_gemini(query, model=args.model)
        baseline_result = decompose_query(query)
        results.append(
            {
                "query": query,
                "gemini_result": gemini_result,
                "evaluation": evaluate_case(query, gemini_result, expected),
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
