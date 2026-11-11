import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_extracted_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    text = text.replace("â€™", "'")
    text = text.replace("â€“", "-")
    text = text.replace("â€”", "-")
    text = text.replace("â€œ", '"')
    text = text.replace("â€\x9d", '"')
    text = text.replace("ï‚·", "-")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^Page \d+\n", "", text)
    text = re.sub(r"\nPage \d+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page_text(page: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = normalize_text(page["text"])
    return {
        "page_number": page["page_number"],
        "marker": page["marker"],
        "text": cleaned,
        "character_count": len(cleaned),
    }


def build_cleaned_artifact(raw_artifact: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_pages: List[Dict[str, Any]] = [clean_page_text(page) for page in raw_artifact["pages"]]
    stitched_text = "\n\n".join(f"{page['marker']}\n\n{page['text']}" for page in cleaned_pages).strip()
    return {
        "document_id": raw_artifact["document_id"],
        "source_file": raw_artifact["source_file"],
        "source_relative_path": raw_artifact["source_relative_path"],
        "input_artifact_type": "document_text_extraction",
        "normalization_method": "rule_based_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(cleaned_pages),
        "character_count": len(stitched_text),
        "pages": cleaned_pages,
        "text": stitched_text,
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "document_text"
        / f"{artifact['document_id']}_cleaned.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean extracted PDF text artifact.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_document_text_extraction/document_text/microsoft_fy2025_10k_summary.json",
    )
    parser.add_argument("--run-id", default="component2_document_text_cleaning")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    input_path = project_root / args.input

    raw_artifact = load_extracted_artifact(input_path)
    cleaned_artifact = build_cleaned_artifact(raw_artifact)
    out_path = write_artifact(project_root, args.run_id, cleaned_artifact)

    print(json.dumps({"ok": True, "output_path": str(out_path), "page_count": cleaned_artifact["page_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
