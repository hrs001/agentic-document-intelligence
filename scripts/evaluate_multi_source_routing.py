import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.multi_source_routing import (
    build_graph_capability_summary,
    build_sql_capability_summary,
    load_json_result,
    route_sub_query,
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_case(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    selected = set(result["selected_sources"])
    must_include = set(case.get("must_include_sources", []))
    must_exclude = set(case.get("must_exclude_sources", []))
    missing_required = sorted(must_include - selected)
    forbidden_selected = sorted(must_exclude & selected)
    passed = not missing_required and not forbidden_selected
    return {
        "missing_required_sources": missing_required,
        "forbidden_selected_sources": forbidden_selected,
        "passed": passed,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "routing"
        / "multi_source_routing_eval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multi-source routing across vector, graph, and SQL.")
    parser.add_argument(
        "--cases-path",
        default="evals/multi_source_routing_cases.json",
    )
    parser.add_argument(
        "--sql-schema-path",
        default="artifacts/experiments/component6_sql_schema_packaging_live/sql_schema/sql_schema_package.json",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--run-id", default="component7_multi_source_routing_eval")
    args = parser.parse_args()

    schema_package = load_json_result(PROJECT_ROOT / "agentic_document_intelligence" / args.sql_schema_path)
    sql_capability_summary = build_sql_capability_summary(schema_package)
    graph_capability_summary = build_graph_capability_summary()
    cases = load_cases(PROJECT_ROOT / "agentic_document_intelligence" / args.cases_path)

    case_results = []
    for case in cases:
        route = route_sub_query(
            case["query"],
            sql_capability_summary,
            graph_capability_summary,
            model=args.model,
        )
        evaluation = evaluate_case(route, case)
        case_results.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "routing_decision": route,
                "evaluation": evaluation,
            }
        )

    passed_count = sum(1 for item in case_results if item["evaluation"]["passed"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "case_count": len(case_results),
        "passed_count": passed_count,
        "pass_rate": round(passed_count / len(case_results), 4) if case_results else 0.0,
        "case_results": case_results,
    }
    report_path = write_report(PROJECT_ROOT, args.run_id, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "case_count": len(case_results),
                "passed_count": passed_count,
                "pass_rate": report["pass_rate"],
                "model": args.model,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
