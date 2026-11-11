import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.benchmark_graph_entity_extraction import (
    extract_graph_entities_and_relationships,
)


MODEL_NAME = "gpt-5.4-mini"


def load_graph_input_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_entities(
    graph_input: dict[str, Any],
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for entity in entities:
        enriched.append(
            {
                "name": entity["name"],
                "entity_type": entity["entity_type"],
                "evidence": entity["evidence"],
                "source_graph_input_id": graph_input["graph_input_id"],
                "source_parent_id": graph_input["source_parent_id"],
                "source_child_ids": list(graph_input["source_child_ids"]),
                "section_title": graph_input["section_title"],
                "page_start": graph_input["page_start"],
                "page_end": graph_input["page_end"],
            }
        )
    return enriched


def enrich_relationships(
    graph_input: dict[str, Any],
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for relationship in relationships:
        enriched.append(
            {
                "source": relationship["source"],
                "relation_type": relationship["relation_type"],
                "target": relationship["target"],
                "evidence": relationship["evidence"],
                "source_graph_input_id": graph_input["graph_input_id"],
                "source_parent_id": graph_input["source_parent_id"],
                "source_child_ids": list(graph_input["source_child_ids"]),
                "section_title": graph_input["section_title"],
                "page_start": graph_input["page_start"],
                "page_end": graph_input["page_end"],
            }
        )
    return enriched


def build_graph_extraction_artifact(
    graph_input_artifact: dict[str, Any],
    extractor: Callable[[str, str, str], dict[str, Any]],
    model: str = MODEL_NAME,
) -> dict[str, Any]:
    extraction_records: list[dict[str, Any]] = []
    total_entities = 0
    total_relationships = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0

    for graph_input in graph_input_artifact["graph_inputs"]:
        extraction = extractor(graph_input["extraction_text"], graph_input["section_title"], model)
        usage = extraction["usage"]
        enriched_entities = enrich_entities(graph_input, extraction["entities"])
        enriched_relationships = enrich_relationships(graph_input, extraction["relationships"])

        extraction_records.append(
            {
                "graph_input_id": graph_input["graph_input_id"],
                "source_parent_id": graph_input["source_parent_id"],
                "section_title": graph_input["section_title"],
                "page_start": graph_input["page_start"],
                "page_end": graph_input["page_end"],
                "source_child_ids": list(graph_input["source_child_ids"]),
                "entity_count": len(enriched_entities),
                "relationship_count": len(enriched_relationships),
                "entities": enriched_entities,
                "relationships": enriched_relationships,
                "usage": usage,
            }
        )

        total_entities += len(enriched_entities)
        total_relationships += len(enriched_relationships)
        total_prompt_tokens += usage["prompt_tokens"]
        total_completion_tokens += usage["completion_tokens"]
        total_cost_usd += usage["estimated_cost_usd"]

    return {
        "document_id": graph_input_artifact["document_id"],
        "input_artifact_type": "graph_inputs",
        "graph_extraction_method": {
            "type": "llm_entity_relationship_extraction_v1",
            "model": model,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph_input_count": graph_input_artifact["graph_input_count"],
        "record_count": len(extraction_records),
        "entity_count": total_entities,
        "relationship_count": total_relationships,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "estimated_cost_usd": round(total_cost_usd, 6),
        },
        "records": extraction_records,
    }


def write_artifact(project_root: Path, run_id: str, artifact: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_extraction"
        / f"{artifact['document_id']}_graph_extraction.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract graph entities and relationships from packaged graph inputs.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component5_graph_extraction_input_packaging_live/graph_inputs/microsoft_fy2025_10k_summary_graph_inputs.json",
    )
    parser.add_argument("--run-id", default="component5_graph_entity_extraction")
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    graph_input_artifact = load_graph_input_artifact(input_path)
    artifact = build_graph_extraction_artifact(
        graph_input_artifact,
        extractor=extract_graph_entities_and_relationships,
        model=args.model,
    )
    output_path = write_artifact(project_root, args.run_id, artifact)

    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(output_path),
                "record_count": artifact["record_count"],
                "entity_count": artifact["entity_count"],
                "relationship_count": artifact["relationship_count"],
                "estimated_cost_usd": artifact["usage"]["estimated_cost_usd"],
                "model": args.model,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
