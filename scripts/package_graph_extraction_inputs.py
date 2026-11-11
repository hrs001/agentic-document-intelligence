import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GRAPH_EXTRACTION_MODEL = "gpt-5.4-mini"


def load_chunk_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_extraction_text(section_title: str, parent_text: str) -> str:
    normalized_section_title = section_title.strip()
    normalized_parent_text = parent_text.strip()
    if not normalized_section_title:
        return normalized_parent_text
    if normalized_parent_text.startswith(normalized_section_title):
        return normalized_parent_text
    return f"{normalized_section_title}\n\n{normalized_parent_text}"


def build_graph_extraction_input_artifact(chunk_artifact: dict[str, Any]) -> dict[str, Any]:
    parent_records: dict[str, dict[str, Any]] = {}
    skipped_table_count = 0

    for chunk in chunk_artifact["chunks"]:
        metadata = chunk["metadata"]
        if metadata.get("content_type") == "table":
            skipped_table_count += 1
            continue

        parent_id = chunk["parent_id"]
        if parent_id not in parent_records:
            section_title = metadata.get("section_title", "")
            parent_text = chunk["parent_text"]
            parent_records[parent_id] = {
                "graph_input_id": f"{parent_id}_graph_input",
                "document_id": metadata["document_id"],
                "source_file": metadata["source_file"],
                "source_parent_id": parent_id,
                "section_title": section_title,
                "page_start": metadata["page"],
                "page_end": metadata.get("page_end", metadata["page"]),
                "content_type": "text",
                "source_child_ids": [],
                "source_child_count": 0,
                "parent_text": parent_text,
                "extraction_text": build_extraction_text(section_title, parent_text),
            }

        parent_records[parent_id]["source_child_ids"].append(chunk["child_id"])
        parent_records[parent_id]["source_child_count"] += 1

    graph_inputs = sorted(
        parent_records.values(),
        key=lambda item: (item["page_start"], item["section_title"], item["source_parent_id"]),
    )

    return {
        "document_id": chunk_artifact["document_id"],
        "input_artifact_type": "chunk_artifact",
        "graph_input_strategy": {
            "type": "unique_text_parent_chunks_v1",
            "active_content_types": ["text"],
            "excluded_content_types": ["table"],
            "target_extraction_model": GRAPH_EXTRACTION_MODEL,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": chunk_artifact["page_count"],
        "section_count": chunk_artifact["section_count"],
        "graph_input_count": len(graph_inputs),
        "skipped_table_chunk_count": skipped_table_count,
        "graph_inputs": graph_inputs,
    }


def write_artifact(project_root: Path, run_id: str, artifact: dict[str, Any]) -> Path:
    output_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_inputs"
        / f"{artifact['document_id']}_graph_inputs.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package stable graph extraction inputs from chunk artifacts.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component5_graph_extraction_input_packaging")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    chunk_artifact = load_chunk_artifact(input_path)
    graph_input_artifact = build_graph_extraction_input_artifact(chunk_artifact)
    output_path = write_artifact(project_root, args.run_id, graph_input_artifact)

    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(output_path),
                "graph_input_count": graph_input_artifact["graph_input_count"],
                "skipped_table_chunk_count": graph_input_artifact["skipped_table_chunk_count"],
                "target_extraction_model": GRAPH_EXTRACTION_MODEL,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
