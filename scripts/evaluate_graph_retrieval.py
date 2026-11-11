import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.graph_retrieval import normalize_text, retrieve_graph_evidence


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_relation(source: str, relation_type: str, target: str) -> tuple[str, str, str]:
    return (normalize_text(source), relation_type.strip().lower(), normalize_text(target))


def evaluate_case(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    matched_node_names = {normalize_text(node["canonical_name"]) for node in result["matched_nodes"]}
    matched_node_names.update(normalize_text(edge["source_canonical_name"]) for edge in result["matched_edges"])
    matched_node_names.update(normalize_text(edge["target_canonical_name"]) for edge in result["matched_edges"])
    expected_node_names = {normalize_text(name) for name in case["expected_node_names"]}
    node_hit_count = len(expected_node_names & matched_node_names)

    matched_relations = {
        normalize_relation(edge["source_canonical_name"], edge["relation_type"], edge["target_canonical_name"])
        for edge in result["matched_edges"]
    }
    expected_relations = {
        normalize_relation(rel["source"], rel["relation_type"], rel["target"])
        for rel in case["expected_relations"]
    }
    relation_hit_count = len(expected_relations & matched_relations)

    passed = node_hit_count == len(expected_node_names) and relation_hit_count == len(expected_relations)
    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "node_hit_count": node_hit_count,
        "expected_node_count": len(expected_node_names),
        "relation_hit_count": relation_hit_count,
        "expected_relation_count": len(expected_relations),
        "passed": passed,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_retrieval"
        / "graph_retrieval_eval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate graph retrieval against graph-oriented benchmark queries.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--database-path",
        default="artifacts/experiments/component5_kuzu_graph_build_live/kuzu_db/microsoft_fy2025_10k_summary.kuzu",
    )
    parser.add_argument(
        "--cases-path",
        default="evals/graph_retrieval_cases.json",
    )
    parser.add_argument("--run-id", default="component5_graph_retrieval_eval")
    parser.add_argument("--top-node-k", type=int, default=5)
    parser.add_argument("--top-edge-k", type=int, default=8)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    database_path = project_root / args.database_path
    cases = load_cases(project_root / args.cases_path)

    case_results = []
    for case in cases:
        result = retrieve_graph_evidence(database_path, case["query"], args.top_node_k, args.top_edge_k)
        evaluation = evaluate_case(result, case)
        case_results.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "evaluation": evaluation,
                "matched_nodes": result["matched_nodes"],
                "matched_edges": result["matched_edges"],
            }
        )

    passed_count = sum(1 for item in case_results if item["evaluation"]["passed"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_path": str(database_path),
        "case_count": len(case_results),
        "passed_count": passed_count,
        "case_results": case_results,
    }
    report_path = write_report(project_root, args.run_id, report)
    print(
        json.dumps(
            {"ok": True, "report_path": str(report_path), "case_count": len(case_results), "passed_count": passed_count},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
