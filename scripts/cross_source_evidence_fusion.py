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


DEFAULT_INPUT = (
    "artifacts/experiments/component7_multi_source_orchestration/"
    "orchestration/multi_source_orchestration_report.json"
)

ENTITYISH_VALUE_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9&.,'()/ -]{1,}")
TITLE_ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9&.-]+){0,5}|AI)\b")


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"] if "result" in payload else payload


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def normalize_key(value: Any) -> str:
    normalized = normalize_text(value)
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for item in items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=True) if isinstance(item, (dict, list)) else str(item)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result


def extract_title_entities(text: str, max_entities: int = 8) -> list[str]:
    candidates = []
    for match in TITLE_ENTITY_PATTERN.findall(text or ""):
        cleaned = str(match).strip(" ,.;:()[]{}")
        if len(cleaned) < 2:
            continue
        candidates.append(cleaned)
        if len(candidates) >= max_entities:
            break
    return dedupe_preserve_order(candidates)


def build_vector_fact(sub_query_result: dict[str, Any], source_output: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    citation = item.get("citation", {})
    entities = dedupe_preserve_order(
        extract_title_entities(item.get("child_text", ""))
        + extract_title_entities(item.get("parent_text", ""))
        + [citation.get("section_title", "")]
    )
    return {
        "fact_id": f"vector::{item.get('source_chunk_id', '')}",
        "source_type": "vector_document",
        "sub_query_id": sub_query_result["sub_query_id"],
        "sub_query": sub_query_result["original_sub_query"],
        "resolved_sub_query": sub_query_result.get("resolved_sub_query", sub_query_result["original_sub_query"]),
        "fact_type": "text_evidence",
        "summary": item.get("child_text", "")[:280],
        "child_text": item.get("child_text", ""),
        "parent_text": item.get("parent_text", ""),
        "entities": [entity for entity in entities if entity],
        "section_title": citation.get("section_title", ""),
        "parent_id": item.get("parent_id") or citation.get("parent_id", ""),
        "citation": citation,
        "scores": {
            "best_score": item.get("best_score", 0.0),
            "rerank_score": item.get("rerank_score", 0.0),
            "mmr_score": item.get("mmr_score", 0.0),
        },
    }


def build_graph_node_fact(sub_query_result: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    snippet_text = "\n".join(node.get("evidence_snippets", [])[:2]).strip()
    entities = dedupe_preserve_order(
        [node.get("canonical_name", "")]
        + list(node.get("aliases", []))
        + extract_title_entities(snippet_text)
        + list(node.get("section_titles", []))
    )
    return {
        "fact_id": f"graph_node::{node.get('node_id', node.get('canonical_name', ''))}",
        "source_type": "graph_relationships",
        "sub_query_id": sub_query_result["sub_query_id"],
        "sub_query": sub_query_result["original_sub_query"],
        "resolved_sub_query": sub_query_result.get("resolved_sub_query", sub_query_result["original_sub_query"]),
        "fact_type": "entity",
        "summary": (
            f"{node.get('canonical_name', '')} ({node.get('entity_type', 'unknown')}): "
            f"{snippet_text[:240]}"
        ).strip(),
        "entity_name": node.get("canonical_name", ""),
        "entity_type": node.get("entity_type", ""),
        "entities": [entity for entity in entities if entity],
        "aliases": node.get("aliases", []),
        "section_titles": node.get("section_titles", []),
        "parent_ids": node.get("source_parent_ids", []),
        "child_ids": node.get("source_child_ids", []),
        "page_ranges": node.get("page_ranges", []),
        "evidence_snippets": node.get("evidence_snippets", []),
        "node_score": node.get("node_score", 0.0),
    }


def build_graph_edge_fact(sub_query_result: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any]:
    snippet_text = "\n".join(edge.get("evidence_snippets", [])[:2]).strip()
    entities = dedupe_preserve_order(
        [edge.get("source_canonical_name", ""), edge.get("target_canonical_name", "")]
        + extract_title_entities(snippet_text)
        + list(edge.get("section_titles", []))
    )
    return {
        "fact_id": f"graph_edge::{edge.get('edge_id', '')}",
        "source_type": "graph_relationships",
        "sub_query_id": sub_query_result["sub_query_id"],
        "sub_query": sub_query_result["original_sub_query"],
        "resolved_sub_query": sub_query_result.get("resolved_sub_query", sub_query_result["original_sub_query"]),
        "fact_type": "relationship",
        "summary": (
            f"{edge.get('source_canonical_name', '')} -> "
            f"{edge.get('relation_type', '')} -> {edge.get('target_canonical_name', '')}: "
            f"{snippet_text[:220]}"
        ).strip(),
        "entities": [entity for entity in entities if entity],
        "source_entity": edge.get("source_canonical_name", ""),
        "relation_type": edge.get("relation_type", ""),
        "target_entity": edge.get("target_canonical_name", ""),
        "section_titles": edge.get("section_titles", []),
        "parent_ids": edge.get("source_parent_ids", []),
        "child_ids": edge.get("source_child_ids", []),
        "page_ranges": edge.get("page_ranges", []),
        "evidence_snippets": edge.get("evidence_snippets", []),
        "edge_score": edge.get("edge_score", 0.0),
    }


def build_sql_row_fact(sub_query_result: dict[str, Any], source_output: dict[str, Any], row: dict[str, Any], row_index: int) -> dict[str, Any]:
    evidence_bundle = source_output.get("evidence_bundle", {})
    string_entities = [
        str(value).strip()
        for value in row.values()
        if isinstance(value, str) and ENTITYISH_VALUE_PATTERN.fullmatch(str(value).strip() or "") and str(value).strip()
    ]
    summary_parts = [f"{key}={row.get(key)}" for key in list(row.keys())[:4]]
    return {
        "fact_id": f"sql::{sub_query_result['sub_query_id']}::{row_index}",
        "source_type": "sql_structured",
        "sub_query_id": sub_query_result["sub_query_id"],
        "sub_query": sub_query_result["original_sub_query"],
        "resolved_sub_query": sub_query_result.get("resolved_sub_query", sub_query_result["original_sub_query"]),
        "fact_type": "sql_row",
        "summary": ", ".join(summary_parts),
        "entities": dedupe_preserve_order(string_entities),
        "target_tables": evidence_bundle.get("target_tables", []),
        "validated_sql": evidence_bundle.get("validated_sql", ""),
        "row": row,
        "columns": list(row.keys()),
        "confidence": evidence_bundle.get("confidence", "unknown"),
    }


def collect_normalized_facts(orchestration_result: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    for sub_query_result in orchestration_result.get("sub_query_results", []):
        for source_output in sub_query_result.get("source_outputs", []):
            source = source_output.get("source")
            bundle = source_output.get("evidence_bundle", {})
            if source == "vector_document":
                for sub_bundle in bundle.get("sub_query_bundles", []):
                    for item in sub_bundle.get("evidence_items", []):
                        facts.append(build_vector_fact(sub_query_result, source_output, item))
            elif source == "graph_relationships":
                for node in bundle.get("matched_nodes", []):
                    facts.append(build_graph_node_fact(sub_query_result, node))
                for edge in bundle.get("matched_edges", []):
                    facts.append(build_graph_edge_fact(sub_query_result, edge))
            elif source == "sql_structured":
                for row_index, row in enumerate(bundle.get("preview_rows", []), start=1):
                    facts.append(build_sql_row_fact(sub_query_result, source_output, row, row_index))
    return facts


def detect_overlap_signals(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps = []

    entity_sources: dict[str, set[str]] = {}
    entity_fact_ids: dict[str, list[str]] = {}
    parent_sources: dict[str, set[str]] = {}
    parent_fact_ids: dict[str, list[str]] = {}
    section_sources: dict[str, set[str]] = {}
    section_fact_ids: dict[str, list[str]] = {}

    for fact in facts:
        for entity in fact.get("entities", []):
            key = normalize_key(entity)
            if not key:
                continue
            entity_sources.setdefault(key, set()).add(fact["source_type"])
            entity_fact_ids.setdefault(key, []).append(fact["fact_id"])
        for parent_id in fact.get("parent_ids", []) + ([fact.get("parent_id")] if fact.get("parent_id") else []):
            key = normalize_key(parent_id)
            if not key:
                continue
            parent_sources.setdefault(key, set()).add(fact["source_type"])
            parent_fact_ids.setdefault(key, []).append(fact["fact_id"])
        for section in fact.get("section_titles", []) + ([fact.get("section_title")] if fact.get("section_title") else []):
            key = normalize_key(section)
            if not key:
                continue
            section_sources.setdefault(key, set()).add(fact["source_type"])
            section_fact_ids.setdefault(key, []).append(fact["fact_id"])

    for key, sources in entity_sources.items():
        if len(sources) > 1:
            overlaps.append(
                {
                    "overlap_type": "entity",
                    "key": key,
                    "sources": sorted(sources),
                    "fact_ids": dedupe_preserve_order(entity_fact_ids[key]),
                }
            )
    for key, sources in parent_sources.items():
        if len(sources) > 1:
            overlaps.append(
                {
                    "overlap_type": "parent_context",
                    "key": key,
                    "sources": sorted(sources),
                    "fact_ids": dedupe_preserve_order(parent_fact_ids[key]),
                }
            )
    for key, sources in section_sources.items():
        if len(sources) > 1:
            overlaps.append(
                {
                    "overlap_type": "section",
                    "key": key,
                    "sources": sorted(sources),
                    "fact_ids": dedupe_preserve_order(section_fact_ids[key]),
                }
            )
    return overlaps


def detect_sql_conflicts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    conflicts = []
    for fact in facts:
        if fact.get("source_type") != "sql_structured":
            continue
        row = fact.get("row", {})
        target_tables = fact.get("target_tables", [])
        table_name = target_tables[0] if target_tables else "unknown_table"
        identifier_cells = [
            (column, str(value))
            for column, value in row.items()
            if isinstance(value, str) and str(value).strip()
        ]
        identifier_key = json.dumps(sorted(identifier_cells), sort_keys=True)
        for column, value in row.items():
            if not isinstance(value, (int, float, str)) or column in {name for name, _ in identifier_cells}:
                continue
            key = (table_name, identifier_key, column)
            comparable = normalize_text(value) if isinstance(value, str) else value
            if key in observed and observed[key]["value"] != comparable:
                conflicts.append(
                    {
                        "conflict_type": "sql_value_mismatch",
                        "table": table_name,
                        "identifier": sorted(identifier_cells),
                        "column": column,
                        "values": [observed[key]["raw_value"], value],
                        "fact_ids": [observed[key]["fact_id"], fact["fact_id"]],
                    }
                )
            else:
                observed[key] = {
                    "value": comparable,
                    "raw_value": value,
                    "fact_id": fact["fact_id"],
                }
    return conflicts


def detect_graph_relation_conflicts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relation_map: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts = []
    for fact in facts:
        if fact.get("fact_type") != "relationship":
            continue
        key = (normalize_key(fact.get("source_entity", "")), normalize_key(fact.get("target_entity", "")))
        relation_type = normalize_key(fact.get("relation_type", ""))
        if key in relation_map and relation_map[key]["relation_type"] != relation_type:
            conflicts.append(
                {
                    "conflict_type": "graph_relation_mismatch",
                    "entity_pair": [fact.get("source_entity", ""), fact.get("target_entity", "")],
                    "relation_types": [relation_map[key]["raw_relation_type"], fact.get("relation_type", "")],
                    "fact_ids": [relation_map[key]["fact_id"], fact["fact_id"]],
                }
            )
        else:
            relation_map[key] = {
                "relation_type": relation_type,
                "raw_relation_type": fact.get("relation_type", ""),
                "fact_id": fact["fact_id"],
            }
    return conflicts


def detect_conflict_signals(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return detect_sql_conflicts(facts) + detect_graph_relation_conflicts(facts)


def rank_facts_for_answer(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "sql_structured": 0,
        "graph_relationships": 1,
        "vector_document": 2,
    }
    fact_type_priority = {
        "sql_row": 0,
        "relationship": 1,
        "entity": 2,
        "text_evidence": 3,
    }
    return sorted(
        facts,
        key=lambda fact: (
            priority.get(fact.get("source_type", ""), 9),
            fact_type_priority.get(fact.get("fact_type", ""), 9),
            -(fact.get("scores", {}).get("rerank_score", 0.0) or 0.0),
            -(fact.get("edge_score", 0.0) or 0.0),
            -(fact.get("node_score", 0.0) or 0.0),
        ),
    )


def build_answer_ready_line(fact: dict[str, Any]) -> str:
    if fact["source_type"] == "sql_structured":
        table_name = ", ".join(fact.get("target_tables", []))
        return f"[SQL] {fact['summary']} | tables={table_name}"
    if fact["fact_type"] == "relationship":
        return f"[Graph] {fact['summary']}"
    if fact["fact_type"] == "entity":
        return f"[Graph] {fact['summary']}"
    citation = fact.get("citation", {})
    return (
        f"[Vector] {fact['summary']} | section={citation.get('section_title', '')} "
        f"| pages={citation.get('page')} - {citation.get('page_end')}"
    )


def build_sub_query_fusions(
    orchestration_result: dict[str, Any],
    facts: list[dict[str, Any]],
    overlaps: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    max_fact_lines: int = 8,
) -> list[dict[str, Any]]:
    fact_by_sub_query: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        fact_by_sub_query.setdefault(fact["sub_query_id"], []).append(fact)

    overlap_by_fact = {}
    for overlap in overlaps:
        for fact_id in overlap.get("fact_ids", []):
            overlap_by_fact.setdefault(fact_id, []).append(overlap)

    conflict_by_fact = {}
    for conflict in conflicts:
        for fact_id in conflict.get("fact_ids", []):
            conflict_by_fact.setdefault(fact_id, []).append(conflict)

    sub_query_fusions = []
    for sub_query_result in orchestration_result.get("sub_query_results", []):
        sub_query_id = sub_query_result["sub_query_id"]
        sub_query_facts = rank_facts_for_answer(fact_by_sub_query.get(sub_query_id, []))
        sources = sorted({fact["source_type"] for fact in sub_query_facts})
        local_overlaps = dedupe_preserve_order(
            [signal for fact in sub_query_facts for signal in overlap_by_fact.get(fact["fact_id"], [])]
        )
        local_conflicts = dedupe_preserve_order(
            [signal for fact in sub_query_facts for signal in conflict_by_fact.get(fact["fact_id"], [])]
        )
        answer_lines = [build_answer_ready_line(fact) for fact in sub_query_facts[:max_fact_lines]]
        assembled_text = "\n".join(
            [
                f"[Fused Sub-query] {sub_query_result['original_sub_query']}",
                f"Resolved Query: {sub_query_result.get('resolved_sub_query', sub_query_result['original_sub_query'])}",
                f"Active Sources: {', '.join(sources) if sources else 'none'}",
                f"Fact Count: {len(sub_query_facts)}",
                f"Overlap Signals: {len(local_overlaps)}",
                f"Conflict Signals: {len(local_conflicts)}",
                "Answer-Ready Facts:",
                *(answer_lines or ["(no fused facts)"]),
            ]
        )
        sub_query_fusions.append(
            {
                "sub_query_id": sub_query_id,
                "original_sub_query": sub_query_result["original_sub_query"],
                "resolved_sub_query": sub_query_result.get("resolved_sub_query", sub_query_result["original_sub_query"]),
                "active_sources": sources,
                "fact_count": len(sub_query_facts),
                "facts": sub_query_facts,
                "overlap_signals": local_overlaps,
                "conflict_signals": local_conflicts,
                "assembled_fused_sub_query_text": assembled_text,
            }
        )
    return sub_query_fusions


def fuse_cross_source_evidence(orchestration_result: dict[str, Any]) -> dict[str, Any]:
    facts = collect_normalized_facts(orchestration_result)
    overlap_signals = detect_overlap_signals(facts)
    conflict_signals = detect_conflict_signals(facts)
    sub_query_fusions = build_sub_query_fusions(orchestration_result, facts, overlap_signals, conflict_signals)
    assembled_text = "\n\n---\n\n".join(item["assembled_fused_sub_query_text"] for item in sub_query_fusions)
    return {
        "original_query": orchestration_result["original_query"],
        "policy": orchestration_result.get("policy", {}),
        "routing_summary": orchestration_result.get("routing_summary", {}),
        "execution_summary": orchestration_result.get("execution_summary", {}),
        "bundle_summary": {
            "sub_query_count": len(sub_query_fusions),
            "fact_count": len(facts),
            "overlap_signal_count": len(overlap_signals),
            "conflict_signal_count": len(conflict_signals),
        },
        "normalized_facts": facts,
        "overlap_signals": overlap_signals,
        "conflict_signals": conflict_signals,
        "sub_query_fusions": sub_query_fusions,
        "assembled_fused_evidence_text": assembled_text,
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answer_context"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "cross_source_fused_evidence.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse vector, graph, and SQL evidence into a conflict-aware answer context.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--run-id", default="component8_cross_source_evidence_fusion")
    args = parser.parse_args()

    orchestration_result = load_report(PROJECT_ROOT / "agentic_document_intelligence" / args.input)
    result = fuse_cross_source_evidence(orchestration_result)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "bundle_summary": result["bundle_summary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
