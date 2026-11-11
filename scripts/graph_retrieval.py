import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kuzu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "did",
    "does",
    "for",
    "from",
    "how",
    "include",
    "includes",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
}


def normalize_text(value: str) -> str:
    cleaned = value.lower().replace("â€™", "'").replace("’", "'").replace("—", " ").replace("–", " ")
    cleaned = re.sub(r"[^a-z0-9$%+\-./ ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def tokenize(value: str) -> list[str]:
    tokens = []
    for token in normalize_text(value).split():
        if not token or token in STOPWORDS:
            continue
        tokens.append(stem_token(token))
    return tokens


def stem_token(token: str) -> str:
    if len(token) <= 4:
        return token
    for suffix in ("ing", "ates", "ate", "ated", "es", "s", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def load_nodes(conn: kuzu.Connection) -> list[dict[str, Any]]:
    df = conn.execute(
        """
        MATCH (n:Entity)
        RETURN
            n.node_id AS node_id,
            n.canonical_name AS canonical_name,
            n.entity_type AS entity_type,
            n.aliases_json AS aliases_json,
            n.evidence_snippets_json AS evidence_snippets_json,
            n.source_graph_input_ids_json AS source_graph_input_ids_json,
            n.source_parent_ids_json AS source_parent_ids_json,
            n.source_child_ids_json AS source_child_ids_json,
            n.section_titles_json AS section_titles_json,
            n.page_ranges_json AS page_ranges_json,
            n.mention_count AS mention_count
        """
    ).get_as_df()
    rows = df.to_dict(orient="records")
    nodes = []
    for row in rows:
        nodes.append(
            {
                "node_id": row["node_id"],
                "canonical_name": row["canonical_name"],
                "entity_type": row["entity_type"],
                "aliases": json.loads(row["aliases_json"]),
                "evidence_snippets": json.loads(row["evidence_snippets_json"]),
                "source_graph_input_ids": json.loads(row["source_graph_input_ids_json"]),
                "source_parent_ids": json.loads(row["source_parent_ids_json"]),
                "source_child_ids": json.loads(row["source_child_ids_json"]),
                "section_titles": json.loads(row["section_titles_json"]),
                "page_ranges": json.loads(row["page_ranges_json"]),
                "mention_count": int(row["mention_count"]),
            }
        )
    return nodes


def score_node(query: str, node: dict[str, Any]) -> float:
    query_norm = normalize_text(query)
    query_tokens = set(tokenize(query))
    canonical_norm = normalize_text(node["canonical_name"])
    canonical_tokens = set(tokenize(node["canonical_name"]))
    alias_tokens = set()
    for alias in node.get("aliases", []):
        alias_tokens.update(tokenize(alias))

    evidence_tokens = set()
    for snippet in node.get("evidence_snippets", [])[:3]:
        evidence_tokens.update(tokenize(snippet))

    overlap = len(query_tokens & canonical_tokens)
    alias_overlap = len(query_tokens & alias_tokens)
    evidence_overlap = len(query_tokens & evidence_tokens)
    phrase_bonus = 0.0
    if canonical_norm and canonical_norm in query_norm:
        phrase_bonus += 3.0
    if any(normalize_text(alias) in query_norm for alias in node.get("aliases", []) if alias):
        phrase_bonus += 2.0
    if any(token in query_tokens for token in canonical_tokens):
        phrase_bonus += 0.5

    mention_bonus = min(node.get("mention_count", 0), 5) * 0.1
    score = (overlap * 2.0) + (alias_overlap * 1.5) + (evidence_overlap * 0.4) + phrase_bonus + mention_bonus
    return round(score, 4)


def select_top_nodes(nodes: list[dict[str, Any]], query: str, top_n: int) -> list[dict[str, Any]]:
    scored = []
    for node in nodes:
        score = score_node(query, node)
        if score <= 0:
            continue
        item = dict(node)
        item["node_score"] = score
        scored.append(item)
    return sorted(scored, key=lambda item: (-item["node_score"], -item["mention_count"], item["canonical_name"].lower()))[:top_n]


def load_node_neighborhood(conn: kuzu.Connection, node_ids: list[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node_id in node_ids:
        df = conn.execute(
            """
            MATCH (src:Entity)-[r:GraphEdge]->(dst:Entity)
            WHERE src.node_id = $node_id OR dst.node_id = $node_id
            RETURN
                r.edge_id AS edge_id,
                src.node_id AS source_node_id,
                src.canonical_name AS source_canonical_name,
                dst.node_id AS target_node_id,
                dst.canonical_name AS target_canonical_name,
                r.relation_type AS relation_type,
                r.evidence_snippets_json AS evidence_snippets_json,
                r.source_graph_input_ids_json AS source_graph_input_ids_json,
                r.source_parent_ids_json AS source_parent_ids_json,
                r.source_child_ids_json AS source_child_ids_json,
                r.section_titles_json AS section_titles_json,
                r.page_ranges_json AS page_ranges_json,
                r.mention_count AS mention_count
            """,
            parameters={"node_id": node_id},
        ).get_as_df()
        for row in df.to_dict(orient="records"):
            edge_id = row["edge_id"]
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edges.append(
                {
                    "edge_id": edge_id,
                    "source_node_id": row["source_node_id"],
                    "source_canonical_name": row["source_canonical_name"],
                    "target_node_id": row["target_node_id"],
                    "target_canonical_name": row["target_canonical_name"],
                    "relation_type": row["relation_type"],
                    "evidence_snippets": json.loads(row["evidence_snippets_json"]),
                    "source_graph_input_ids": json.loads(row["source_graph_input_ids_json"]),
                    "source_parent_ids": json.loads(row["source_parent_ids_json"]),
                    "source_child_ids": json.loads(row["source_child_ids_json"]),
                    "section_titles": json.loads(row["section_titles_json"]),
                    "page_ranges": json.loads(row["page_ranges_json"]),
                    "mention_count": int(row["mention_count"]),
                }
            )
    return edges


def score_edge(query: str, edge: dict[str, Any], top_node_ids: set[str]) -> float:
    query_norm = normalize_text(query)
    query_tokens = set(tokenize(query))
    edge_tokens = set(tokenize(edge["source_canonical_name"])) | set(tokenize(edge["target_canonical_name"]))
    for snippet in edge.get("evidence_snippets", [])[:2]:
        edge_tokens.update(tokenize(snippet))

    overlap = len(query_tokens & edge_tokens)
    score = overlap * 1.5
    source_norm = normalize_text(edge["source_canonical_name"])
    target_norm = normalize_text(edge["target_canonical_name"])
    if source_norm and source_norm in query_norm:
        score += 3.0
    if target_norm and target_norm in query_norm:
        score += 3.0
    if edge["source_node_id"] in top_node_ids:
        score += 1.0
    if edge["target_node_id"] in top_node_ids:
        score += 1.0
    score += min(edge.get("mention_count", 0), 3) * 0.2
    return round(score, 4)


def retrieve_graph_evidence(
    database_path: Path,
    query: str,
    top_node_k: int = 5,
    top_edge_k: int = 8,
) -> dict[str, Any]:
    db = kuzu.Database(str(database_path))
    conn = kuzu.Connection(db)
    nodes = load_nodes(conn)
    top_nodes = select_top_nodes(nodes, query, top_node_k)
    top_node_ids = {node["node_id"] for node in top_nodes}
    neighborhood_edges = load_node_neighborhood(conn, [node["node_id"] for node in top_nodes])

    scored_edges = []
    for edge in neighborhood_edges:
        edge_copy = dict(edge)
        edge_copy["edge_score"] = score_edge(query, edge, top_node_ids)
        scored_edges.append(edge_copy)

    top_edges = sorted(
        [edge for edge in scored_edges if edge["edge_score"] > 0],
        key=lambda item: (-item["edge_score"], -item["mention_count"], item["edge_id"]),
    )[:top_edge_k]

    return {
        "query": query,
        "top_node_k": top_node_k,
        "top_edge_k": top_edge_k,
        "matched_nodes": top_nodes,
        "matched_edges": top_edges,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_retrieval"
        / "graph_retrieval_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve graph evidence from the local Kuzu graph database.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--database-path",
        default="artifacts/experiments/component5_kuzu_graph_build_live/kuzu_db/microsoft_fy2025_10k_summary.kuzu",
    )
    parser.add_argument("--query", default="Which segment includes GitHub?")
    parser.add_argument("--top-node-k", type=int, default=5)
    parser.add_argument("--top-edge-k", type=int, default=8)
    parser.add_argument("--run-id", default="component5_graph_retrieval")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    database_path = project_root / args.database_path

    result = retrieve_graph_evidence(
        database_path=database_path,
        query=args.query,
        top_node_k=args.top_node_k,
        top_edge_k=args.top_edge_k,
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_path": str(database_path),
        **result,
    }
    report_path = write_report(project_root, args.run_id, report)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "matched_node_count": len(result["matched_nodes"]),
                "matched_edge_count": len(result["matched_edges"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
