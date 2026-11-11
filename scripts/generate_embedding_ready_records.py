import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_chunk_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_embedding_ready_artifact(chunk_artifact: Dict[str, Any]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []

    for chunk in chunk_artifact["chunks"]:
        metadata = dict(chunk["metadata"])
        content_type = metadata.get("content_type", "text")
        text = chunk["child_text"]
        record_id = f"{chunk['child_id']}_{stable_text_hash(text)}"

        records.append(
            {
                "record_id": record_id,
                "source_chunk_id": chunk["child_id"],
                "parent_id": chunk["parent_id"],
                "text": text,
                "content_type": content_type,
                "text_hash": stable_text_hash(text),
                "metadata": metadata,
            }
        )

    return {
        "document_id": chunk_artifact["document_id"],
        "input_artifact_type": "chunk_records",
        "embedding_provider_target": "pinecone",
        "embedding_model_target": "text-embedding-3-small",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "records": records,
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "embeddings"
        / f"{artifact['document_id']}_embedding_records.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Pinecone-ready embedding records from chunk records.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component2_embedding_ready_records")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    chunk_artifact = load_chunk_artifact(project_root / args.input)
    embedding_artifact = build_embedding_ready_artifact(chunk_artifact)
    out_path = write_artifact(project_root, args.run_id, embedding_artifact)

    print(json.dumps({"ok": True, "output_path": str(out_path), "record_count": embedding_artifact["record_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
