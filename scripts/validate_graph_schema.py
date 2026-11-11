import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HIGH_VALUE_TYPES = {
    "organization",
    "segment",
    "product_or_service",
    "technology_or_platform",
    "regulatory_or_external_body",
    "geography",
}
CONDITIONAL_TYPES = {
    "initiative_or_strategy",
    "risk_or_issue",
    "financial_or_business_concept",
    "time_period",
    "other",
}
EDGE_TYPES = {
    "includes",
    "integrates",
    "invests_in",
    "supports",
    "faces",
    "uses",
    "positions_as",
    "depends_on",
}


def load_normalized_graph_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_words(value: str) -> int:
    return len([part for part in value.strip().split() if part])


def validate_node(node: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    entity_type = node["entity_type"]
    name = node["canonical_name"].strip()
    mention_count = int(node.get("mention_count", 0))
    evidence_count = len(node.get("evidence_snippets", []))
    word_count = count_words(name)

    if not name:
        reasons.append("empty_name")
    if evidence_count == 0:
        reasons.append("missing_evidence")
    if word_count > 12:
        reasons.append("name_too_long")
    if entity_type not in HIGH_VALUE_TYPES | CONDITIONAL_TYPES:
        reasons.append("unsupported_entity_type")

    if entity_type in HIGH_VALUE_TYPES:
        if mention_count < 1:
            reasons.append("missing_mentions")
    elif entity_type == "initiative_or_strategy":
        if mention_count < 2 and word_count > 4:
            reasons.append("low_confidence_initiative")
    elif entity_type == "risk_or_issue":
        if mention_count < 2 and word_count > 5:
            reasons.append("low_confidence_risk")
    elif entity_type == "financial_or_business_concept":
        if mention_count < 2 and word_count > 4:
            reasons.append("low_confidence_business_concept")
    elif entity_type == "time_period":
        if mention_count < 2:
            reasons.append("low_value_time_period")
    elif entity_type == "other":
        reasons.append("generic_other_type")

    return (len(reasons) == 0, reasons)


def validate_edge(edge: dict[str, Any], valid_node_ids: set[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if edge["relation_type"] not in EDGE_TYPES:
        reasons.append("unsupported_relation_type")
    if edge["source_node_id"] not in valid_node_ids:
        reasons.append("invalid_source_node")
    if edge["target_node_id"] not in valid_node_ids:
        reasons.append("invalid_target_node")
    if edge["source_node_id"] == edge["target_node_id"]:
        reasons.append("self_loop")
    if len(edge.get("evidence_snippets", [])) == 0:
        reasons.append("missing_evidence")

    if edge["relation_type"] in {"supports", "depends_on", "positions_as"} and edge.get("mention_count", 0) < 2:
        reasons.append("weak_single_mention_relation")

    return (len(reasons) == 0, reasons)


def build_graph_schema_validation_artifact(normalized_artifact: dict[str, Any]) -> dict[str, Any]:
    valid_nodes: list[dict[str, Any]] = []
    rejected_nodes: list[dict[str, Any]] = []

    for node in normalized_artifact["nodes"]:
        is_valid, reasons = validate_node(node)
        if is_valid:
            valid_nodes.append(node)
        else:
            rejected_nodes.append(
                {
                    "node_id": node["node_id"],
                    "canonical_name": node["canonical_name"],
                    "entity_type": node["entity_type"],
                    "mention_count": node["mention_count"],
                    "rejection_reasons": reasons,
                }
            )

    valid_node_ids = {node["node_id"] for node in valid_nodes}
    valid_edges: list[dict[str, Any]] = []
    rejected_edges: list[dict[str, Any]] = []

    for edge in normalized_artifact["edges"]:
        is_valid, reasons = validate_edge(edge, valid_node_ids)
        if is_valid:
            valid_edges.append(edge)
        else:
            rejected_edges.append(
                {
                    "edge_id": edge["edge_id"],
                    "source_canonical_name": edge["source_canonical_name"],
                    "relation_type": edge["relation_type"],
                    "target_canonical_name": edge["target_canonical_name"],
                    "mention_count": edge["mention_count"],
                    "rejection_reasons": reasons,
                }
            )

    return {
        "document_id": normalized_artifact["document_id"],
        "input_artifact_type": "graph_normalized",
        "graph_schema_validation_method": {
            "type": "deterministic_schema_filter_v1",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_node_count": normalized_artifact["normalized_node_count"],
        "raw_edge_count": normalized_artifact["normalized_edge_count"],
        "validated_node_count": len(valid_nodes),
        "validated_edge_count": len(valid_edges),
        "rejected_node_count": len(rejected_nodes),
        "rejected_edge_count": len(rejected_edges),
        "validated_nodes": valid_nodes,
        "validated_edges": valid_edges,
        "rejected_nodes": rejected_nodes,
        "rejected_edges": rejected_edges,
        "rejected_node_reason_counts": dict(
            Counter(reason for item in rejected_nodes for reason in item["rejection_reasons"])
        ),
        "rejected_edge_reason_counts": dict(
            Counter(reason for item in rejected_edges for reason in item["rejection_reasons"])
        ),
    }


def write_artifact(project_root: Path, run_id: str, artifact: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_validated"
        / f"{artifact['document_id']}_graph_validated.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate normalized graph artifact against a practical schema.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component5_graph_normalization_live/graph_normalized/microsoft_fy2025_10k_summary_graph_normalized.json",
    )
    parser.add_argument("--run-id", default="component5_graph_schema_validation")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    normalized_artifact = load_normalized_graph_artifact(input_path)
    artifact = build_graph_schema_validation_artifact(normalized_artifact)
    output_path = write_artifact(project_root, args.run_id, artifact)

    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(output_path),
                "validated_node_count": artifact["validated_node_count"],
                "validated_edge_count": artifact["validated_edge_count"],
                "rejected_node_count": artifact["rejected_node_count"],
                "rejected_edge_count": artifact["rejected_edge_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
