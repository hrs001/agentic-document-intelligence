import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import fitz

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.clean_document_text import normalize_text
from scripts.extract_document_text import load_manifest


def table_to_markdown(rows: List[List[str]]) -> str:
    cleaned_rows = [[(cell or "").strip() for cell in row] for row in rows if any((cell or "").strip() for cell in row)]
    if not cleaned_rows:
        return ""
    header = cleaned_rows[0]
    body = cleaned_rows[1:] or [[]]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def bbox_intersects(a: tuple, b: tuple) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def extract_layout_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_number = page_index + 1
        blocks = page.get_text("blocks")
        tables = page.find_tables()
        table_entries: List[Dict[str, Any]] = []
        table_bboxes: List[tuple] = []

        if tables:
            for table_index, table in enumerate(tables.tables):
                rows = table.extract()
                markdown = table_to_markdown(rows)
                bbox = tuple(table.bbox)
                table_bboxes.append(bbox)
                table_entries.append(
                    {
                        "item_id": f"page{page_number}_table{table_index}",
                        "type": "table",
                        "bbox": [float(x) for x in bbox],
                        "row_count": len(rows),
                        "column_count": len(rows[0]) if rows else 0,
                        "header": getattr(table, "header", None).names if getattr(table, "header", None) else [],
                        "markdown": markdown,
                    }
                )

        text_entries: List[Dict[str, Any]] = []
        for block_index, block in enumerate(blocks):
            x0, y0, x1, y1, text, *_ = block
            bbox = (float(x0), float(y0), float(x1), float(y1))
            if any(bbox_intersects(bbox, tb) for tb in table_bboxes):
                continue
            cleaned_text = normalize_text(text or "")
            if not cleaned_text:
                continue
            text_entries.append(
                {
                    "item_id": f"page{page_number}_text{block_index}",
                    "type": "text",
                    "bbox": list(bbox),
                    "text": cleaned_text,
                }
            )

        items = sorted(text_entries + table_entries, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        pages.append(
            {
                "page_number": page_number,
                "items": items,
                "text_block_count": len(text_entries),
                "table_count": len(table_entries),
            }
        )

    return pages


def build_layout_artifact(project_root: Path, document_id: str) -> Dict[str, Any]:
    manifest = load_manifest(project_root)
    document = next(doc for doc in manifest["documents"] if doc["document_id"] == document_id)
    pdf_path = project_root / document["relative_path"]
    pages = extract_layout_pages(pdf_path)
    return {
        "document_id": document_id,
        "source_file": pdf_path.name,
        "source_relative_path": document["relative_path"],
        "extraction_method": "pymupdf_layout_blocks_and_tables_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "pages": pages,
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "document_layout"
        / f"{artifact['document_id']}_layout.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract layout-aware text blocks and tables from the fixed PDF.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument("--document-id", default="microsoft_fy2025_10k_summary")
    parser.add_argument("--run-id", default="component2_layout_aware_extraction")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent

    artifact = build_layout_artifact(project_root, args.document_id)
    out_path = write_artifact(project_root, args.run_id, artifact)
    print(json.dumps({"ok": True, "output_path": str(out_path), "page_count": artifact["page_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
