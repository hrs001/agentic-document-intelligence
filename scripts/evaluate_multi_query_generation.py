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

from agentic_document_intelligence.scripts.multi_query_generation import (
    MODEL_NAME,
    generate_multi_queries,
)


DEFAULT_CASES_PATH = (
    PROJECT_ROOT
    / "agentic_document_intelligence"
    / "evals"
    / "multi_query_generation_cases.json"
)


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = re.sub(r"[?.!,;:]+$", "", normalized)
    normalized = normalized.replace("microsoft azure", "azure")
    normalized = normalized.replace("fiscal year 2025", "fy2025")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def has_duplicate_rewrites(queries: list[str]) -> bool:
    normalized = [normalize_text(item) for item in queries]
    return len(normalized) != len(set(normalized))


def covers_required_angles(rewrites: list[dict[str, str]], required_angles: list[str]) -> bool:
    actual = {item.get("angle", "") for item in rewrites}
    return all(angle in actual for angle in required_angles)


def evaluate_case(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    rewrite_queries = [item["query"] for item in actual.get("rewrites", [])]
    duplicate_flag = has_duplicate_rewrites(rewrite_queries)
    expected_count_range = expected["rewrite_count_range"]
    count_ok = expected_count_range[0] <= len(rewrite_queries) <= expected_count_range[1]
    angle_ok = covers_required_angles(actual.get("rewrites", []), expected["required_angles"])
    pass_flag = count_ok and angle_ok and not duplicate_flag
    return {
        "passed": pass_flag,
        "count_ok": count_ok,
        "angle_ok": angle_ok,
        "has_duplicates": duplicate_flag,
        "rewrite_count": len(rewrite_queries),
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    report_path = output_dir / "multi_query_generation_eval_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-query generation quality.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cases-path", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--run-id", default="component3_multi_query_generation_eval")
    args = parser.parse_args()

    cases_path = Path(args.cases_path)
    cases = load_cases(cases_path)
    results = []
    for case in cases:
        query = case["query"]
        expected = case["expected"]
        generated = generate_multi_queries(query, model=args.model)
        results.append(
            {
                "query": query,
                "result": generated,
                "evaluation": evaluate_case(generated, expected),
            }
        )

    passed_count = sum(item["evaluation"]["passed"] for item in results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "cases_path": str(cases_path),
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
