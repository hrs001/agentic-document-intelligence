import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pinecone import Pinecone


SPARSE_MODEL = "pinecone-sparse-english-v0"


def load_embedding_ready_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_child_records(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [record for record in artifact["records"] if "_ch" in record["source_chunk_id"]]


def chunked(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def fetch_existing_dense_vectors(index, namespace: str, record_ids: List[str]) -> Dict[str, Any]:
    fetched: Dict[str, Any] = {}
    for batch in chunked(record_ids, 100):
        response = index.fetch(ids=batch, namespace=namespace)
        vectors = getattr(response, "vectors", {}) or {}
        fetched.update(vectors)
    return fetched


def embed_sparse_records(pc: Pinecone, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    embeddings: List[Dict[str, Any]] = []
    for batch in chunked(records, 96):
        texts = [record["text"] for record in batch]
        result = pc.inference.embed(
            model=SPARSE_MODEL,
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        embeddings.extend(list(result.data))
    return embeddings


def build_hybrid_vectors(records: List[Dict[str, Any]], fetched_vectors: Dict[str, Any], sparse_embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vectors: List[Dict[str, Any]] = []
    for record, sparse in zip(records, sparse_embeddings):
        existing = fetched_vectors.get(record["record_id"])
        if existing is None:
            raise KeyError(f"Missing existing dense vector for record_id={record['record_id']}")
        values = list(getattr(existing, "values", []) or existing.get("values", []))
        metadata = dict(getattr(existing, "metadata", {}) or existing.get("metadata", {}))
        vectors.append(
            {
                "id": record["record_id"],
                "values": values,
                "sparse_values": {
                    "indices": list(sparse["sparse_indices"]),
                    "values": list(sparse["sparse_values"]),
                },
                "metadata": metadata,
            }
        )
    return vectors


def upsert_vectors(index, namespace: str, vectors: List[Dict[str, Any]], batch_size: int = 100) -> int:
    count = 0
    for batch in chunked(vectors, batch_size):
        index.upsert(vectors=batch, namespace=namespace)
        count += len(batch)
    return count


def write_report(project_root: Path, run_id: str, report: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "pinecone"
        / "pinecone_sparse_upsert_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Pinecone sparse vectors and merge them into existing dense records.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_embedding_ready_records/embeddings/microsoft_fy2025_10k_summary_embedding_records.json",
    )
    parser.add_argument("--run-id", default="component2_pinecone_sparse_upsert")
    parser.add_argument("--namespace", default="microsoft_fy2025_fixed_corpus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    load_dotenv(project_root / ".env")

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    artifact = load_embedding_ready_artifact(project_root / args.input)
    child_records = filter_child_records(artifact)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_id": artifact["document_id"],
        "namespace": args.namespace,
        "index_name": index_name,
        "sparse_model": SPARSE_MODEL,
        "child_record_count": len(child_records),
        "upserted_count": 0,
        "dry_run": bool(args.dry_run),
    }

    fetched_vectors = fetch_existing_dense_vectors(index, args.namespace, [record["record_id"] for record in child_records])
    sparse_embeddings = embed_sparse_records(pc, child_records)
    hybrid_vectors = build_hybrid_vectors(child_records, fetched_vectors, sparse_embeddings)

    if args.dry_run:
        sample = hybrid_vectors[0] if hybrid_vectors else {}
        report["sample_vector"] = {
            "id": sample.get("id"),
            "dense_dim": len(sample.get("values", [])),
            "sparse_nnz": len(sample.get("sparse_values", {}).get("indices", [])),
            "metadata": sample.get("metadata", {}),
        }
        out = write_report(project_root, args.run_id, report)
        print(json.dumps({"ok": True, "report_path": str(out), "dry_run": True, "child_record_count": len(child_records)}, indent=2))
        return 0

    upserted_count = upsert_vectors(index, args.namespace, hybrid_vectors)
    report["upserted_count"] = upserted_count
    sample = hybrid_vectors[0] if hybrid_vectors else {}
    report["sample_vector"] = {
        "id": sample.get("id"),
        "dense_dim": len(sample.get("values", [])),
        "sparse_nnz": len(sample.get("sparse_values", {}).get("indices", [])),
        "metadata": sample.get("metadata", {}),
    }
    out = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out), "upserted_count": upserted_count, "namespace": args.namespace}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
