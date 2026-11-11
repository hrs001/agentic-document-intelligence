import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.transformation_gating import select_transformations


DEFAULT_CASES_PATH = (
    PROJECT_ROOT
    / "agentic_document_intelligence"
    / "evals"
    / "transformation_gating_cases.json"
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_case(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "run_decomposition": actual["run_decomposition"] == expected["run_decomposition"],
        "run_multi_query": actual["run_multi_query"] == expected["run_multi_query"],
        "run_step_back": actual["run_step_back"] == expected["run_step_back"],
        "run_hyde": actual["run_hyde"] == expected["run_hyde"],
        "is_ambiguous_case": actual["is_ambiguous_case"] == expected["is_ambiguous_case"],
        "estimated_sub_queries": actual["estimated_sub_queries"] == expected["estimated_sub_queries"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


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
    report_path = output_dir / "transformation_gating_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic transformation gating.")
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--run-id", default="component3_transformation_gating_eval")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases_path))
    results = []
    for case in cases:
        actual = select_transformations(case["query"])
        results.append(
            {
                "query": case["query"],
                "result": actual,
                "evaluation": evaluate_case(actual, case["expected"]),
            }
        )

    passed_count = sum(item["evaluation"]["passed"] for item in results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": args.cases_path,
        "case_count": len(cases),
        "passed_count": passed_count,
        "results": results,
    }
    report_path = write_report(args.run_id, payload)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(cases),
                "passed_count": passed_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
