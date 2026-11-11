import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.evaluate_transformation_gating import (
    DEFAULT_CASES_PATH,
    evaluate_case,
    load_cases,
)
from agentic_document_intelligence.scripts.llm_transformation_gating import (
    MODEL_NAME,
    select_transformations_with_llm,
)
from agentic_document_intelligence.scripts.transformation_gating import select_transformations


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
    report_path = output_dir / "llm_transformation_gating_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM-assisted transformation gating.")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--run-id", default="component3_llm_transformation_gating_eval")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases_path))
    results = []
    baseline_results = []
    llm_applied_count = 0

    for case in cases:
        query = case["query"]
        expected = case["expected"]
        llm_result = select_transformations_with_llm(query, model=args.model)
        baseline = select_transformations(query)
        if llm_result["llm_selector_applied"]:
            llm_applied_count += 1
        results.append(
            {
                "query": query,
                "result": llm_result,
                "evaluation": evaluate_case(llm_result, expected),
            }
        )
        baseline_results.append(
            {
                "query": query,
                "result": baseline,
                "evaluation": evaluate_case(baseline, expected),
            }
        )

    passed_count = sum(item["evaluation"]["passed"] for item in results)
    baseline_passed_count = sum(item["evaluation"]["passed"] for item in baseline_results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "cases_path": args.cases_path,
        "case_count": len(cases),
        "llm_applied_count": llm_applied_count,
        "passed_count": passed_count,
        "baseline_passed_count": baseline_passed_count,
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
                "llm_applied_count": llm_applied_count,
                "passed_count": passed_count,
                "baseline_passed_count": baseline_passed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
