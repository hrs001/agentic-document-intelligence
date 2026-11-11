import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kuzu

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_validated_graph_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def create_schema(conn: kuzu.Connection) -> None:
    conn.execute(
        """
        CREATE NODE TABLE Entity(
            node_id STRING,
            canonical_name STRING,
            entity_type STRING,
            aliases_json STRING,
            evidence_snippets_json STRING,
            source_graph_input_ids_json STRING,
            source_parent_ids_json STRING,
            source_child_ids_json STRING,
            section_titles_json STRING,
            page_ranges_json STRING,
            mention_count INT64,
            PRIMARY KEY(node_id)
        )
        """
    )
    conn.execute(
        """
        CREATE REL TABLE GraphEdge(
            FROM Entity TO Entity,
            edge_id STRING,
            relation_type STRING,
            evidence_snippets_json STRING,
            source_graph_input_ids_json STRING,
            source_parent_ids_json STRING,
            source_child_ids_json STRING,
            section_titles_json STRING,
            page_ranges_json STRING,
            mention_count INT64
        )
        """
    )


def insert_nodes(conn: kuzu.Connection, nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        conn.execute(
            """
            CREATE (n:Entity {
                node_id: $node_id,
                canonical_name: $canonical_name,
                entity_type: $entity_type,
                aliases_json: $aliases_json,
                evidence_snippets_json: $evidence_snippets_json,
                source_graph_input_ids_json: $source_graph_input_ids_json,
                source_parent_ids_json: $source_parent_ids_json,
                source_child_ids_json: $source_child_ids_json,
                section_titles_json: $section_titles_json,
                page_ranges_json: $page_ranges_json,
                mention_count: $mention_count
            })
            """,
            parameters={
                "node_id": node["node_id"],
                "canonical_name": node["canonical_name"],
                "entity_type": node["entity_type"],
                "aliases_json": json_string(node.get("aliases", [])),
                "evidence_snippets_json": json_string(node.get("evidence_snippets", [])),
                "source_graph_input_ids_json": json_string(node.get("source_graph_input_ids", [])),
                "source_parent_ids_json": json_string(node.get("source_parent_ids", [])),
                "source_child_ids_json": json_string(node.get("source_child_ids", [])),
                "section_titles_json": json_string(node.get("section_titles", [])),
                "page_ranges_json": json_string(node.get("page_ranges", [])),
                "mention_count": int(node.get("mention_count", 0)),
            },
        )


def insert_edges(conn: kuzu.Connection, edges: list[dict[str, Any]]) -> None:
    for edge in edges:
        conn.execute(
            """
            MATCH (src:Entity {node_id: $source_node_id}),
                  (dst:Entity {node_id: $target_node_id})
            CREATE (src)-[:GraphEdge {
                edge_id: $edge_id,
                relation_type: $relation_type,
                evidence_snippets_json: $evidence_snippets_json,
                source_graph_input_ids_json: $source_graph_input_ids_json,
                source_parent_ids_json: $source_parent_ids_json,
                source_child_ids_json: $source_child_ids_json,
                section_titles_json: $section_titles_json,
                page_ranges_json: $page_ranges_json,
                mention_count: $mention_count
            }]->(dst)
            """,
            parameters={
                "source_node_id": edge["source_node_id"],
                "target_node_id": edge["target_node_id"],
                "edge_id": edge["edge_id"],
                "relation_type": edge["relation_type"],
                "evidence_snippets_json": json_string(edge.get("evidence_snippets", [])),
                "source_graph_input_ids_json": json_string(edge.get("source_graph_input_ids", [])),
                "source_parent_ids_json": json_string(edge.get("source_parent_ids", [])),
                "source_child_ids_json": json_string(edge.get("source_child_ids", [])),
                "section_titles_json": json_string(edge.get("section_titles", [])),
                "page_ranges_json": json_string(edge.get("page_ranges", [])),
                "mention_count": int(edge.get("mention_count", 0)),
            },
        )


def query_scalar(conn: kuzu.Connection, query: str, column: str) -> int:
    df = conn.execute(query).get_as_df()
    return int(df.iloc[0][column])


def build_kuzu_graph_database(validated_artifact: dict[str, Any], database_path: Path) -> dict[str, Any]:
    if database_path.exists():
        if database_path.is_dir():
            shutil.rmtree(database_path)
        else:
            database_path.unlink()
    database_path.parent.mkdir(parents=True, exist_ok=True)

    db = kuzu.Database(str(database_path))
    conn = kuzu.Connection(db)

    create_schema(conn)
    insert_nodes(conn, validated_artifact["validated_nodes"])
    insert_edges(conn, validated_artifact["validated_edges"])

    stored_node_count = query_scalar(conn, "MATCH (n:Entity) RETURN COUNT(*) AS c", "c")
    stored_edge_count = query_scalar(conn, "MATCH ()-[r:GraphEdge]->() RETURN COUNT(*) AS c", "c")
    distinct_relation_type_count = query_scalar(
        conn,
        "MATCH ()-[r:GraphEdge]->() RETURN COUNT(DISTINCT r.relation_type) AS c",
        "c",
    )

    sample_edges = conn.execute(
        """
        MATCH (src:Entity)-[r:GraphEdge]->(dst:Entity)
        RETURN src.canonical_name AS source, r.relation_type AS relation_type, dst.canonical_name AS target
        ORDER BY source, relation_type, target
        LIMIT 5
        """
    ).get_as_df().to_dict(orient="records")

    return {
        "document_id": validated_artifact["document_id"],
        "input_artifact_type": "graph_validated",
        "graph_storage_method": {
            "type": "kuzu_local_db_v1",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_path": str(database_path),
        "input_validated_node_count": validated_artifact["validated_node_count"],
        "input_validated_edge_count": validated_artifact["validated_edge_count"],
        "stored_node_count": stored_node_count,
        "stored_edge_count": stored_edge_count,
        "distinct_relation_type_count": distinct_relation_type_count,
        "sample_edges": sample_edges,
    }


def write_report(project_root: Path, run_id: str, report: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_storage"
        / "kuzu_graph_build_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a local Kuzu graph database from the validated graph artifact.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component5_graph_schema_validation_live/graph_validated/microsoft_fy2025_10k_summary_graph_validated.json",
    )
    parser.add_argument("--run-id", default="component5_kuzu_graph_build")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    validated_artifact = load_validated_graph_artifact(input_path)
    database_path = (
        project_root
        / "artifacts"
        / "experiments"
        / args.run_id
        / "kuzu_db"
        / f"{validated_artifact['document_id']}.kuzu"
    )
    report = build_kuzu_graph_database(validated_artifact, database_path)
    report_path = write_report(project_root, args.run_id, report)

    print(
        json.dumps(
            {
                "ok": True,
                "database_path": report["database_path"],
                "report_path": str(report_path),
                "stored_node_count": report["stored_node_count"],
                "stored_edge_count": report["stored_edge_count"],
                "distinct_relation_type_count": report["distinct_relation_type_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
