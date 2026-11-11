import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


DENSE_MODEL = "text-embedding-3-small"
SPARSE_MODEL = "pinecone-sparse-english-v0"


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


def scale_dense(vector: List[float], alpha: float) -> List[float]:
    return [v * alpha for v in vector]


def scale_sparse(indices: List[int], values: List[float], alpha: float) -> Dict[str, List[float]]:
    sparse_weight = 1.0 - alpha
    return {"indices": list(indices), "values": [v * sparse_weight for v in values]}


def embed_dense_query(client: OpenAI, query: str) -> List[float]:
    response = client.embeddings.create(model=DENSE_MODEL, input=[query])
    return response.data[0].embedding


def embed_sparse_query(pc: Pinecone, query: str) -> Dict[str, List[float]]:
    response = pc.inference.embed(
        model=SPARSE_MODEL,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    item = response.data[0]
    return {"indices": list(item["sparse_indices"]), "values": list(item["sparse_values"])}


def run_hybrid_query(index, namespace: str, dense_vector: List[float], sparse_vector: Dict[str, List[float]], top_k: int) -> List[Any]:
    response = index.query(
        vector=dense_vector,
        sparse_vector=sparse_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    return list(response.matches)


def dedupe_matches(matches: List[Any]) -> List[Any]:
    seen = set()
    deduped = []
    for match in matches:
        metadata = dict(getattr(match, "metadata", {}) or {})
        source_chunk_id = metadata.get("source_chunk_id", match.id)
        if source_chunk_id in seen:
            continue
        seen.add(source_chunk_id)
        deduped.append(match)
    return deduped


def hydrate_matches(matches: List[Any], child_index: Dict[str, Dict[str, Any]], parent_index: Dict[str, str]) -> List[Dict[str, Any]]:
    hydrated = []
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
        / "pinecone_hybrid_retrieval_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native Pinecone hybrid retrieval and hydrate matched chunks.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to script parent.")
    parser.add_argument(
        "--chunk-input",
        default="artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json",
    )
    parser.add_argument("--run-id", default="component2_pinecone_hybrid_retrieval")
    parser.add_argument("--namespace", default="microsoft_fy2025_fixed_corpus")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--query", default="What were Microsoft's FY2025 revenue, operating income, and net income?")
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

    dense = embed_dense_query(openai_client, args.query)
    sparse = embed_sparse_query(pc, args.query)
    dense_scaled = scale_dense(dense, args.alpha)
    sparse_scaled = scale_sparse(sparse["indices"], sparse["values"], args.alpha)
    raw_matches = run_hybrid_query(index, args.namespace, dense_scaled, sparse_scaled, args.top_k)
    deduped = dedupe_matches(raw_matches)
    hydrated = hydrate_matches(deduped, child_index, parent_index)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "namespace": args.namespace,
        "index_name": index_name,
        "query": args.query,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "dense_model": DENSE_MODEL,
        "sparse_model": SPARSE_MODEL,
        "raw_match_count": len(raw_matches),
        "deduped_match_count": len(deduped),
        "matches": hydrated,
        "lambda_notes": {
            "stateless": True,
            "network_dependencies": ["OpenAI embeddings", "Pinecone sparse embed", "Pinecone hybrid query"],
            "artifact_hydration": "Local chunk artifact now; can later move to packaged assets or S3 for Lambda."
        }
    }
    out_path = write_report(project_root, args.run_id, report)
    print(json.dumps({"ok": True, "report_path": str(out_path), "raw_match_count": len(raw_matches), "deduped_match_count": len(deduped)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
