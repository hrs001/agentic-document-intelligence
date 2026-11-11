import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536


def load_embedding_ready_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_pinecone_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(record["metadata"])
    meta["parent_id"] = record["parent_id"]
    meta["source_chunk_id"] = record["source_chunk_id"]
    meta["content_type"] = record["content_type"]
    meta["text_hash"] = record["text_hash"]
    return meta


def filter_child_records(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [record for record in artifact["records"] if record["source_chunk_id"].endswith("_ch0") or "_ch" in record["source_chunk_id"]]


def ensure_index(pc: Pinecone, index_name: str, cloud: str, region: str) -> None:
    existing = pc.list_indexes().names()
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


def embed_records(client: OpenAI, records: List[Dict[str, Any]]) -> List[List[float]]:
    texts = [record["text"] for record in records]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_vectors(records: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    vectors = []
    for record, embedding in zip(records, embeddings):
        vectors.append(
            {
                "id": record["record_id"],
                "values": embedding,
                "metadata": build_pinecone_metadata(record),
            }
        )
    return vectors


def upsert_vectors(index, namespace: str, vectors: List[Dict[str, Any]], batch_size: int = 100) -> int:
    count = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
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
        / "pinecone_upsert_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed child chunks and upsert them to Pinecone.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_embedding_ready_records/embeddings/microsoft_fy2025_10k_summary_embedding_records.json",
    )
    parser.add_argument("--run-id", default="component2_pinecone_upsert")
    parser.add_argument("--namespace", default="microsoft_fy2025_fixed_corpus")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    load_dotenv(project_root / ".env")

    artifact = load_embedding_ready_artifact(project_root / args.input)
    child_records = filter_child_records(artifact)

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_id": artifact["document_id"],
        "namespace": args.namespace,
        "embedding_model": EMBEDDING_MODEL,
        "provider": "openai+pinecone",
        "child_record_count": len(child_records),
        "upserted_count": 0,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        report["sample_metadata"] = build_pinecone_metadata(child_records[0]) if child_records else {}
        out = write_report(project_root, args.run_id, report)
        print(json.dumps({"ok": True, "report_path": str(out), "dry_run": True, "child_record_count": len(child_records)}, indent=2))
        return 0

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
    pinecone_region = os.getenv("PINECONE_REGION", "us-east-1").strip()

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    openai_client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    ensure_index(pc, index_name=index_name, cloud=pinecone_cloud, region=pinecone_region)
    index = pc.Index(index_name)

    embeddings = embed_records(openai_client, child_records)
    vectors = build_vectors(child_records, embeddings)
    upserted_count = upsert_vectors(index, namespace=args.namespace, vectors=vectors)

    report["upserted_count"] = upserted_count
    report["index_name"] = index_name
    report["sample_metadata"] = build_pinecone_metadata(child_records[0]) if child_records else {}
    out = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out), "upserted_count": upserted_count, "namespace": args.namespace}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
