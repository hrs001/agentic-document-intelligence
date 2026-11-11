import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_rerank_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assemble_parent_context(report: Dict[str, Any], max_parents: int = 4, max_children_per_parent: int = 2) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for rank, match in enumerate(report.get("matches", []), start=1):
        metadata = dict(match.get("metadata", {}) or {})
        parent_id = metadata.get("parent_id", "")
        source_chunk_id = metadata.get("source_chunk_id", match.get("id"))
        entry = grouped.get(parent_id)
        if entry is None:
            entry = {
                "parent_id": parent_id,
                "section_title": metadata.get("section_title", ""),
                "page": metadata.get("page"),
                "page_end": metadata.get("page_end"),
                "content_types": [],
                "parent_text": match.get("parent_text", ""),
                "children": [],
                "best_rank": rank,
                "best_score": float(match.get("rerank_score", match.get("score", 0.0))),
            }
            grouped[parent_id] = entry

        content_type = metadata.get("content_type", "text")
        if content_type not in entry["content_types"]:
            entry["content_types"].append(content_type)

        if len(entry["children"]) < max_children_per_parent:
            entry["children"].append(
                {
                    "source_chunk_id": source_chunk_id,
                    "content_type": content_type,
                    "score": float(match.get("rerank_score", match.get("score", 0.0))),
                    "child_text": match.get("child_text", ""),
                }
            )

        entry["best_rank"] = min(entry["best_rank"], rank)
        entry["best_score"] = max(entry["best_score"], float(match.get("rerank_score", match.get("score", 0.0))))

    ordered_parents = sorted(grouped.values(), key=lambda item: (item["best_rank"], -item["best_score"]))[:max_parents]
    assembled_text_parts = []
    for index, parent in enumerate(ordered_parents, start=1):
        child_evidence = "\n\n".join(child["child_text"] for child in parent["children"])
        assembled_text_parts.append(
            "\n".join(
                [
                    f"[Context {index}] Section: {parent['section_title']}",
                    f"Pages: {parent['page']} - {parent['page_end']}",
                    f"Parent ID: {parent['parent_id']}",
                    f"Content Types: {', '.join(parent['content_types'])}",
                    "Evidence Snippets:",
                    child_evidence,
                    "Parent Context:",
                    parent["parent_text"],
                ]
            )
        )

    return {
        "query": report.get("query", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_type": "pinecone_rerank_report",
        "parent_context_count": len(ordered_parents),
        "contexts": ordered_parents,
        "assembled_context_text": "\n\n---\n\n".join(assembled_text_parts),
    }


def write_artifact(project_root: Path, run_id: str, artifact: Dict[str, Any]) -> Path:
    out_path = (
        project_root
        / "artifacts"
        / "experiments"
        / run_id
        / "answer_context"
        / "parent_context_bundle.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble reranked child hits into parent-context bundles for answer generation.")
    parser.add_argument("--project-root", default=None, help="Project root path. Defaults to script parent.")
    parser.add_argument(
        "--input",
        default="artifacts/experiments/component2_pinecone_rerank/retrieval/pinecone_rerank_report.json",
    )
    parser.add_argument("--run-id", default="component2_parent_context_assembly")
    parser.add_argument("--max-parents", type=int, default=4)
    parser.add_argument("--max-children-per-parent", type=int, default=2)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else script_dir.parent
    report = load_rerank_report(project_root / args.input)
    artifact = assemble_parent_context(report, max_parents=args.max_parents, max_children_per_parent=args.max_children_per_parent)
    out_path = write_artifact(project_root, args.run_id, artifact)
    print(json.dumps({"ok": True, "report_path": str(out_path), "parent_context_count": artifact["parent_context_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
