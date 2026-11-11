import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STOP_PREFIXES = ("section ", "table ")
TRAILING_ACRONYM_RE = re.compile(r"\s+\(([A-Z0-9&+/ -]{2,12})\)$")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def load_graph_extraction_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_name(name: str) -> str:
    value = WHITESPACE_RE.sub(" ", name.strip())
    value = value.replace("â€™", "'").replace("’", "'").replace("—", "-").replace("–", "-")
    value = TRAILING_ACRONYM_RE.sub("", value).strip()
    return value


def normalize_key(name: str) -> str:
    cleaned = clean_name(name).lower()
    cleaned = NON_ALNUM_RE.sub(" ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def choose_canonical_name(names: list[str]) -> str:
    cleaned_names = [clean_name(name) for name in names if clean_name(name)]
    if not cleaned_names:
        return ""
    counts = Counter(cleaned_names)
    best = sorted(
        counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0].lower()),
    )[0][0]
    return best


def choose_entity_type(entity_types: list[str]) -> str:
    counts = Counter(entity_types)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def should_keep_entity(name: str) -> bool:
    normalized = normalize_key(name)
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in STOP_PREFIXES):
        return False
    if len(normalized) <= 1:
        return False
    return True


def dedupe_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def aggregate_nodes(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    node_map: dict[str, dict[str, Any]] = {}
    alias_to_canonical: dict[str, str] = {}

    for record in records:
        for entity in record["entities"]:
            raw_name = entity["name"]
            canonical_key = normalize_key(raw_name)
            if not should_keep_entity(raw_name):
                continue

            alias_to_canonical[normalize_key(raw_name)] = canonical_key
            if canonical_key not in node_map:
                node_map[canonical_key] = {
                    "node_id": f"node_{canonical_key.replace(' ', '_')}",
                    "canonical_name": clean_name(raw_name),
                    "entity_type": entity["entity_type"],
                    "aliases": [],
                    "evidence_snippets": [],
                    "source_graph_input_ids": [],
                    "source_parent_ids": [],
                    "source_child_ids": [],
                    "section_titles": [],
                    "page_ranges": [],
                    "mention_count": 0,
                }

            node = node_map[canonical_key]
            node["aliases"].append(raw_name.strip())
            node["evidence_snippets"].append(entity["evidence"])
            node["source_graph_input_ids"].append(entity["source_graph_input_id"])
            node["source_parent_ids"].append(entity["source_parent_id"])
            node["source_child_ids"].extend(entity["source_child_ids"])
            node["section_titles"].append(entity["section_title"])
            node["page_ranges"].append(
                {"page_start": entity["page_start"], "page_end": entity["page_end"]}
            )
            node["mention_count"] += 1

            current_names = node["aliases"]
            node["canonical_name"] = choose_canonical_name(current_names)
            node["entity_type"] = choose_entity_type(
                [node["entity_type"], entity["entity_type"]]
            )

    for node in node_map.values():
        node["aliases"] = dedupe_list(
            [alias for alias in node["aliases"] if alias.strip() and alias.strip() != node["canonical_name"]]
        )
        node["evidence_snippets"] = dedupe_list(node["evidence_snippets"])
        node["source_graph_input_ids"] = dedupe_list(node["source_graph_input_ids"])
        node["source_parent_ids"] = dedupe_list(node["source_parent_ids"])
        node["source_child_ids"] = dedupe_list(node["source_child_ids"])
        node["section_titles"] = dedupe_list(node["section_titles"])
        node["page_ranges"] = dedupe_list(node["page_ranges"])

    return node_map, alias_to_canonical


def aggregate_edges(records: list[dict[str, Any]], node_map: dict[str, dict[str, Any]], alias_to_canonical: dict[str, str]) -> dict[str, dict[str, Any]]:
    edge_map: dict[str, dict[str, Any]] = {}

    for record in records:
        for relationship in record["relationships"]:
            source_key = alias_to_canonical.get(normalize_key(relationship["source"]))
            target_key = alias_to_canonical.get(normalize_key(relationship["target"]))
            if not source_key or not target_key:
                continue
            if source_key not in node_map or target_key not in node_map:
                continue

            relation_type = relationship["relation_type"].strip().lower()
            edge_key = f"{source_key}|{relation_type}|{target_key}"
            if edge_key not in edge_map:
                edge_map[edge_key] = {
                    "edge_id": f"edge_{len(edge_map) + 1}",
                    "source_node_id": node_map[source_key]["node_id"],
                    "source_canonical_name": node_map[source_key]["canonical_name"],
                    "relation_type": relation_type,
                    "target_node_id": node_map[target_key]["node_id"],
                    "target_canonical_name": node_map[target_key]["canonical_name"],
                    "evidence_snippets": [],
                    "source_graph_input_ids": [],
                    "source_parent_ids": [],
                    "source_child_ids": [],
                    "section_titles": [],
                    "page_ranges": [],
                    "mention_count": 0,
                }

            edge = edge_map[edge_key]
            edge["evidence_snippets"].append(relationship["evidence"])
            edge["source_graph_input_ids"].append(relationship["source_graph_input_id"])
            edge["source_parent_ids"].append(relationship["source_parent_id"])
            edge["source_child_ids"].extend(relationship["source_child_ids"])
            edge["section_titles"].append(relationship["section_title"])
            edge["page_ranges"].append(
                {"page_start": relationship["page_start"], "page_end": relationship["page_end"]}
            )
            edge["mention_count"] += 1

    for edge in edge_map.values():
        edge["evidence_snippets"] = dedupe_list(edge["evidence_snippets"])
        edge["source_graph_input_ids"] = dedupe_list(edge["source_graph_input_ids"])
        edge["source_parent_ids"] = dedupe_list(edge["source_parent_ids"])
        edge["source_child_ids"] = dedupe_list(edge["source_child_ids"])
        edge["section_titles"] = dedupe_list(edge["section_titles"])
        edge["page_ranges"] = dedupe_list(edge["page_ranges"])

    return edge_map


def build_normalized_graph_artifact(extraction_artifact: dict[str, Any]) -> dict[str, Any]:
    records = extraction_artifact["records"]
    node_map, alias_to_canonical = aggregate_nodes(records)
    edge_map = aggregate_edges(records, node_map, alias_to_canonical)

    nodes = sorted(node_map.values(), key=lambda item: (item["canonical_name"].lower(), item["node_id"]))
    edges = sorted(
        edge_map.values(),
        key=lambda item: (
            item["source_canonical_name"].lower(),
            item["relation_type"],
            item["target_canonical_name"].lower(),
        ),
    )

    return {
        "document_id": extraction_artifact["document_id"],
        "input_artifact_type": "graph_extraction",
        "graph_normalization_method": {
            "type": "deterministic_alias_merge_v1",
            "source_model": extraction_artifact["graph_extraction_method"]["model"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_record_count": extraction_artifact["record_count"],
        "raw_entity_count": extraction_artifact["entity_count"],
        "raw_relationship_count": extraction_artifact["relationship_count"],
        "normalized_node_count": len(nodes),
        "normalized_edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def write_artifact(project_root: Path, run_id: str, artifact: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_normalized"
        / f"{artifact['document_id']}_graph_normalized.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize and deduplicate graph extraction artifacts.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component5_graph_entity_extraction_live/graph_extraction/microsoft_fy2025_10k_summary_graph_extraction.json",
    )
    parser.add_argument("--run-id", default="component5_graph_normalization")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    extraction_artifact = load_graph_extraction_artifact(input_path)
    artifact = build_normalized_graph_artifact(extraction_artifact)
    output_path = write_artifact(project_root, args.run_id, artifact)

    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(output_path),
                "raw_entity_count": artifact["raw_entity_count"],
                "raw_relationship_count": artifact["raw_relationship_count"],
                "normalized_node_count": artifact["normalized_node_count"],
                "normalized_edge_count": artifact["normalized_edge_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
