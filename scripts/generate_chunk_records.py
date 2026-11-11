import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50


def load_cleaned_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_layout_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


SECTION_HEADER_RE = re.compile(
    r"^(?:\d+\.\s+.+|Appendix\s+[A-Z]\.\s+.+)$",
    flags=re.MULTILINE,
)


def extract_structured_sections(layout_artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current_title = ""
    current_page_start = 1
    current_parts: List[str] = []
    current_tables: List[Dict[str, Any]] = []

    for page in layout_artifact["pages"]:
        page_number = page["page_number"]
        text_items = [item for item in page["items"] if item["type"] == "text"]
        page_text = "\n".join(item["text"] for item in text_items)
        stripped_page = page_text.strip()
        is_contents_page = stripped_page.startswith("Contents")
        buffer: List[str] = []

        for item in page["items"]:
            if item["type"] == "table":
                table_copy = dict(item)
                table_copy["page_number"] = page_number
                current_tables.append(table_copy)
                continue

            for line in item["text"].splitlines():
                stripped = line.strip()
                if not stripped:
                    buffer.append(line)
                    continue

                if is_contents_page and SECTION_HEADER_RE.match(stripped):
                    continue

                if (not is_contents_page) and SECTION_HEADER_RE.match(stripped):
                    if buffer:
                        current_parts.append("\n".join(buffer).strip())
                        buffer = []
                    if current_parts or current_tables:
                        sections.append(
                            {
                                "section_title": current_title,
                                "page_start": current_page_start,
                                "page_end": page_number,
                                "text": "\n\n".join(part for part in current_parts if part).strip(),
                                "tables": list(current_tables),
                            }
                        )
                        current_parts = []
                        current_tables = []
                    current_title = stripped
                    current_page_start = page_number
                    buffer.append(stripped)
                else:
                    buffer.append(line)

        if buffer:
            current_parts.append("\n".join(buffer).strip())

    if current_parts:
        sections.append(
            {
                "section_title": current_title or "Document Overview",
                "page_start": current_page_start,
                "page_end": layout_artifact["page_count"],
                "text": "\n\n".join(part for part in current_parts if part).strip(),
                "tables": list(current_tables),
            }
        )

    normalized_sections: List[Dict[str, Any]] = []
    for section in sections:
        text = section["text"].strip()
        if not text:
            continue
        title = section["section_title"] or "Document Overview"
        normalized_sections.append(
            {
                "section_title": title,
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "text": text,
                "tables": section.get("tables", []),
            }
        )
    return normalized_sections


def build_chunk_artifact(layout_artifact: Dict[str, Any]) -> Dict[str, Any]:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    document_id = layout_artifact["document_id"]
    source_file = layout_artifact["source_file"]
    chunk_records: List[Dict[str, Any]] = []
    parent_count = 0
    sections = extract_structured_sections(layout_artifact)

    for section_index, section in enumerate(sections):
        page_number = section["page_start"]
        page_end = section["page_end"]
        section_title = section["section_title"]
        section_text = section["text"]
        parent_chunks = parent_splitter.split_text(section_text)

        for parent_index, parent_text in enumerate(parent_chunks):
            parent_id = f"{document_id}_sec{section_index}_p{page_number}_par{parent_index}"
            child_chunks = child_splitter.split_text(parent_text)

            for child_index, child_text in enumerate(child_chunks):
                chunk_records.append(
                    {
                        "child_id": f"{parent_id}_ch{child_index}",
                        "parent_id": parent_id,
                        "child_text": child_text,
                        "parent_text": parent_text,
                        "metadata": {
                            "document_id": document_id,
                            "source_file": source_file,
                            "page": page_number,
                            "page_end": page_end,
                            "section_title": section_title,
                            "section_index": section_index,
                            "parent_index_on_page": parent_index,
                            "child_index_in_parent": child_index,
                        },
                    }
                )

            parent_count += 1

        for table_index, table in enumerate(section.get("tables", [])):
            markdown = table.get("markdown", "").strip()
            if not markdown:
                continue
            parent_id = f"{document_id}_sec{section_index}_table{table_index}"
            chunk_records.append(
                {
                    "child_id": f"{parent_id}_ch0",
                    "parent_id": parent_id,
                    "child_text": markdown,
                    "parent_text": markdown,
                    "metadata": {
                        "document_id": document_id,
                        "source_file": source_file,
                        "page": table["page_number"],
                        "page_end": table["page_number"],
                        "section_title": section_title,
                        "section_index": section_index,
                        "parent_index_on_page": table_index,
                        "child_index_in_parent": 0,
                        "content_type": "table",
                        "table_row_count": table.get("row_count", 0),
                        "table_column_count": table.get("column_count", 0),
                        "table_header": table.get("header", []),
                    },
                }
            )

    return {
        "document_id": document_id,
        "input_artifact_type": "document_layout",
        "chunking_method": {
            "type": "layout_aware_structure_first_parent_child_v2",
            "parent_chunk_size": PARENT_CHUNK_SIZE,
            "parent_chunk_overlap": PARENT_CHUNK_OVERLAP,
            "child_chunk_size": CHILD_CHUNK_SIZE,
            "child_chunk_overlap": CHILD_CHUNK_OVERLAP,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": layout_artifact["page_count"],
        "section_count": len(sections),
        "parent_chunk_count": parent_count,
        "child_chunk_count": len(chunk_records),
        "sections": sections,
        "chunks": chunk_records,
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "chunks"
        / f"{artifact['document_id']}_chunks.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate parent/child chunk records from cleaned document text.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_layout_aware_extraction/document_layout/microsoft_fy2025_10k_summary_layout.json",
    )
    parser.add_argument("--run-id", default="component2_chunk_generation")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    layout_artifact = load_layout_artifact(input_path)
    chunk_artifact = build_chunk_artifact(layout_artifact)
    out_path = write_artifact(project_root, args.run_id, chunk_artifact)

    print(
        json.dumps(
            {
                "ok": True,
                "output_path": str(out_path),
                "parent_chunk_count": chunk_artifact["parent_chunk_count"],
                "child_chunk_count": chunk_artifact["child_chunk_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
