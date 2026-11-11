import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader


def load_manifest(project_root: Path) -> Dict[str, Any]:
    manifest_path = project_root / "corpus" / "metadata" / "corpus_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_pdf_text(pdf_path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages: List[Dict[str, Any]] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            {
                "page_number": index,
                "marker": f"[PAGE_MARKER_{index}]",
                "text": text,
                "character_count": len(text),
            }
        )

    return pages


def build_document_text_artifact(project_root: Path, document_id: str) -> Dict[str, Any]:
    manifest = load_manifest(project_root)
    document = next(doc for doc in manifest["documents"] if doc["document_id"] == document_id)
    pdf_path = project_root / document["relative_path"]
    pages = extract_pdf_text(pdf_path)

    stitched_text_parts = []
    for page in pages:
        stitched_text_parts.append(f"\n\n{page['marker']}\n\n{page['text']}")

    stitched_text = "".join(stitched_text_parts).strip()

    return {
        "document_id": document_id,
        "source_file": pdf_path.name,
        "source_relative_path": document["relative_path"],
        "extraction_method": "pypdf_page_text_extraction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "character_count": len(stitched_text),
        "pages": pages,
        "text": stitched_text,
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "document_text"
        / f"{artifact['document_id']}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract page-by-page text from the fixed corpus PDF.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument("--document-id", default="microsoft_fy2025_10k_summary")
    parser.add_argument("--run-id", default="component2_document_text_extraction")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent

    artifact = build_document_text_artifact(project_root, args.document_id)
    out_path = write_artifact(project_root, args.run_id, artifact)

    print(json.dumps({"ok": True, "output_path": str(out_path), "page_count": artifact["page_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
