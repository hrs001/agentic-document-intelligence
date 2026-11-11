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
from agentic_document_intelligence.scripts.llm_query_decomposition import MODEL_NAME, decompose_query_with_llm
from agentic_document_intelligence.scripts.query_decomposition import decompose_query
from agentic_document_intelligence.scripts.query_decomposition_repair import repair_query_decomposition


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
    report_path = output_dir / "query_decomposition_repair_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repaired gpt-5-mini query decomposition quality.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cases-path", default=str(DEFAULT_EVAL_CASES_PATH))
    parser.add_argument("--run-id", default="component3_query_decomposition_repair_eval")
    args = parser.parse_args()

    cases_path = Path(args.cases_path)
    cases = load_eval_cases(cases_path)
    results = []
    baseline_results = []
    repair_applied_count = 0

    for case in cases:
        query = case["query"]
        expected = case["expected"]
        llm_result = decompose_query_with_llm(query, model=args.model)
        repaired = repair_query_decomposition(query, llm_result, model=args.model)
        baseline = decompose_query(query)
        if repaired["repair_applied"]:
            repair_applied_count += 1
        results.append(
            {
                "query": query,
                "original_result": llm_result,
                "repaired_result": repaired,
                "evaluation": evaluate_case(query, repaired, expected),
            }
        )
        baseline_results.append(
            {
                "query": query,
                "baseline_result": baseline,
                "evaluation": evaluate_case(query, baseline, expected),
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
        "repair_applied_count": repair_applied_count,
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
                "repair_applied_count": repair_applied_count,
                "passed_count": passed_count,
                "exact_passed_count": exact_passed_count,
                "baseline_passed_count": baseline_passed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
