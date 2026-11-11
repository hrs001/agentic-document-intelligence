import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\&\+\.]*")
BM25_K1 = 1.5
BM25_B = 0.75


def load_chunk_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text or "")]


def build_sparse_index(chunk_artifact: Dict[str, Any]) -> Dict[str, Any]:
    documents: List[Dict[str, Any]] = []
    doc_freq: Counter = Counter()

    child_chunks = [chunk for chunk in chunk_artifact["chunks"] if "_ch" in chunk["child_id"]]
    for chunk in child_chunks:
        tokens = tokenize(chunk["child_text"])
        term_freq = Counter(tokens)
        documents.append(
            {
                "doc_id": chunk["child_id"],
                "parent_id": chunk["parent_id"],
                "content_type": chunk["metadata"].get("content_type", "text"),
                "metadata": chunk["metadata"],
                "length": len(tokens),
                "term_freq": dict(term_freq),
            }
        )
        for token in term_freq:
            doc_freq[token] += 1

    doc_count = len(documents)
    avg_doc_len = (sum(doc["length"] for doc in documents) / doc_count) if doc_count else 0.0
    idf = {
        term: math.log(1 + ((doc_count - freq + 0.5) / (freq + 0.5)))
        for term, freq in doc_freq.items()
    }

    return {
        "document_id": chunk_artifact["document_id"],
        "input_artifact_type": "chunk_records",
        "index_type": "bm25_sparse_index_v1",
        "bm25_params": {
            "k1": BM25_K1,
            "b": BM25_B,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": doc_count,
        "average_document_length": avg_doc_len,
        "vocabulary_size": len(idf),
        "idf": idf,
        "documents": documents,
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "sparse_index"
        / f"{artifact['document_id']}_bm25_index.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a reusable BM25 sparse index from child chunks.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component2_sparse_index")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    chunk_artifact = load_chunk_artifact(project_root / args.input)
    sparse_index = build_sparse_index(chunk_artifact)
    out_path = write_artifact(project_root, args.run_id, sparse_index)
    print(json.dumps({"ok": True, "output_path": str(out_path), "document_count": sparse_index["document_count"], "vocabulary_size": sparse_index["vocabulary_size"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
