import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_INPUT = "artifacts/experiments/component5_graph_retrieval_live/graph_retrieval/graph_retrieval_report.json"


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_node_evidence_item(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node["node_id"],
        "canonical_name": node["canonical_name"],
        "entity_type": node["entity_type"],
        "aliases": node.get("aliases", []),
        "node_score": node.get("node_score", 0.0),
        "mention_count": node.get("mention_count", 0),
        "evidence_snippets": node.get("evidence_snippets", []),
        "source_graph_input_ids": node.get("source_graph_input_ids", []),
        "source_parent_ids": node.get("source_parent_ids", []),
        "source_child_ids": node.get("source_child_ids", []),
        "section_titles": node.get("section_titles", []),
        "page_ranges": node.get("page_ranges", []),
    }


def build_edge_evidence_item(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": edge["edge_id"],
        "source_node_id": edge["source_node_id"],
        "source_canonical_name": edge["source_canonical_name"],
        "relation_type": edge["relation_type"],
        "target_node_id": edge["target_node_id"],
        "target_canonical_name": edge["target_canonical_name"],
        "edge_score": edge.get("edge_score", 0.0),
        "mention_count": edge.get("mention_count", 0),
        "evidence_snippets": edge.get("evidence_snippets", []),
        "source_graph_input_ids": edge.get("source_graph_input_ids", []),
        "source_parent_ids": edge.get("source_parent_ids", []),
        "source_child_ids": edge.get("source_child_ids", []),
        "section_titles": edge.get("section_titles", []),
        "page_ranges": edge.get("page_ranges", []),
    }


def build_node_context_block(index: int, node: dict[str, Any]) -> str:
    first_page = node.get("page_ranges", [{}])[0]
    return "\n".join(
        [
            f"[Graph Node {index}]",
            f"Entity: {node['canonical_name']}",
            f"Type: {node['entity_type']}",
            f"Pages: {first_page.get('page_start')} - {first_page.get('page_end')}",
            f"Sections: {', '.join(node.get('section_titles', []))}",
            "Evidence Snippets:",
            *node.get("evidence_snippets", [])[:3],
        ]
    )


def build_context_block(index: int, edge: dict[str, Any]) -> str:
    first_page = edge.get("page_ranges", [{}])[0]
    return "\n".join(
        [
            f"[Graph Evidence {index}]",
            f"Relation: {edge['source_canonical_name']} -> {edge['relation_type']} -> {edge['target_canonical_name']}",
            f"Pages: {first_page.get('page_start')} - {first_page.get('page_end')}",
            f"Sections: {', '.join(edge.get('section_titles', []))}",
            f"Edge Score: {edge.get('edge_score', 0.0)}",
            "Evidence Snippets:",
            *edge.get("evidence_snippets", []),
        ]
    )


def assemble_graph_evidence_bundle(report: dict[str, Any]) -> dict[str, Any]:
    matched_nodes = [build_node_evidence_item(node) for node in report["matched_nodes"]]
    matched_edges = [build_edge_evidence_item(edge) for edge in report["matched_edges"]]
    context_blocks = [
        build_node_context_block(index, node)
        for index, node in enumerate(matched_nodes[:5], start=1)
    ]
    context_blocks.extend(
        build_context_block(index, edge) for index, edge in enumerate(matched_edges, start=1)
    )
    assembled_text = "\n\n---\n\n".join(context_blocks)

    return {
        "query": report["query"],
        "database_path": report["database_path"],
        "bundle_summary": {
            "matched_node_count": len(matched_nodes),
            "matched_edge_count": len(matched_edges),
        },
        "matched_nodes": matched_nodes,
        "matched_edges": matched_edges,
        "assembled_graph_evidence_text": assembled_text,
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_evidence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "graph_evidence_bundle.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package graph retrieval output into an answer-ready graph evidence bundle.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component5_graph_evidence_packaging")
    args = parser.parse_args()

    report = load_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    bundle = assemble_graph_evidence_bundle(report)
    report_path = write_report(PROJECT_ROOT, args.run_id, bundle)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "matched_node_count": bundle["bundle_summary"]["matched_node_count"],
                "matched_edge_count": bundle["bundle_summary"]["matched_edge_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
