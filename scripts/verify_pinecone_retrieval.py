import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


EMBEDDING_MODEL = "text-embedding-3-small"


def load_chunk_artifact(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_chunk_indexes(chunk_artifact: Dict[str, Any]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_child_id: Dict[str, Dict[str, Any]] = {}
    parent_text_by_id: Dict[str, str] = {}
    for chunk in chunk_artifact["chunks"]:
        by_child_id[chunk["child_id"]] = chunk
        parent_text_by_id.setdefault(chunk["parent_id"], chunk["parent_text"])
    return by_child_id, parent_text_by_id


def embed_query(client: OpenAI, query: str) -> List[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    return response.data[0].embedding


def query_pinecone(index, namespace: str, vector: List[float], top_k: int) -> List[Any]:
    response = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return list(response.matches)


def hydrate_matches(matches: List[Any], child_index: Dict[str, Dict[str, Any]], parent_index: Dict[str, str]) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for match in matches:
        metadata = dict(getattr(match, "metadata", {}) or {})
        child_id = metadata.get("source_chunk_id")
        parent_id = metadata.get("parent_id")
        child_chunk = child_index.get(child_id)
        parent_text = parent_index.get(parent_id, "")
        hydrated.append(
            {
                "id": match.id,
                "score": float(getattr(match, "score", 0.0)),
                "metadata": metadata,
                "child_text": child_chunk["child_text"] if child_chunk else "",
                "parent_text": parent_text,
            }
        )
    return hydrated


def write_report(project_root: Path, run_id: str, report: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "retrieval"
        / "pinecone_retrieval_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Pinecone retrieval and parent-child hydration.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to the script parent.")
    parser.add_argument(
        "--chunk-input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component2_pinecone_retrieval_verification")
    parser.add_argument("--namespace", default="microsoft_fy2025_fixed_corpus")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--query",
        default="What were Microsoft's FY2025 revenue, operating income, and net income?",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    load_dotenv(project_root / ".env")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-document-intelligence").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in .env")

    chunk_artifact = load_chunk_artifact(project_root / args.chunk_input)
    child_index, parent_index = build_chunk_indexes(chunk_artifact)

    openai_client = OpenAI(api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    query_vector = embed_query(openai_client, args.query)
    matches = query_pinecone(index=index, namespace=args.namespace, vector=query_vector, top_k=args.top_k)
    hydrated = hydrate_matches(matches, child_index=child_index, parent_index=parent_index)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "namespace": args.namespace,
        "index_name": index_name,
        "embedding_model": EMBEDDING_MODEL,
        "query": args.query,
        "top_k": args.top_k,
        "match_count": len(hydrated),
        "matches": hydrated,
        "lambda_notes": {
            "stateless": True,
            "network_dependencies": ["OpenAI embeddings", "Pinecone query"],
            "artifact_hydration": "Local chunk artifact now; can later be moved to S3 or packaged asset for Lambda."
        }
    }
    out_path = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out_path), "match_count": len(hydrated)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
