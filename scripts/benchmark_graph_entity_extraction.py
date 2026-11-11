import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2", "gpt-5.1", "gpt-5-mini"]
CASES_PATH = (
    PROJECT_ROOT
    / "agentic_document_intelligence"
    / "evals"
    / "graph_extraction_benchmark_cases.json"
)
CHUNK_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "agentic_document_intelligence"
    / "artifacts"
    / "experiments"
    / "component2_chunk_generation"
    / "chunks"
    / "microsoft_fy2025_10k_summary_chunks.json"
)

MODEL_PRICING_PER_M_TOKEN = {
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.2": {"input": 2.00, "output": 8.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}

ENTITY_TYPES = [
    "organization",
    "segment",
    "product_or_service",
    "technology_or_platform",
    "initiative_or_strategy",
    "risk_or_issue",
    "regulatory_or_external_body",
    "geography",
    "financial_or_business_concept",
    "time_period",
    "other",
]

RELATION_TYPES = [
    "includes",
    "integrates",
    "invests_in",
    "supports",
    "faces",
    "uses",
    "positions_as",
    "depends_on",
]

SYSTEM_PROMPT = """You are a GraphRAG extraction module.
Return only valid JSON matching the schema.

Goal:
- Extract graph-worthy entities and relationships that are explicitly supported by the passage.
- Favor durable business concepts over every possible noun.
- Keep entity names as close to the passage wording as possible.

Rules:
- Do not invent facts or infer unstated relationships.
- Every entity must have a short evidence snippet copied exactly from the passage.
- Every relationship must have a short evidence snippet copied exactly from the passage.
- Only include relationships where both source and target are explicit entities in the passage.
- Prefer fewer, higher-value entities over noisy long lists.
- Keep duplicates out.
- If a numeric fact is not itself useful as a graph node, do not extract it as an entity.
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string", "enum": ENTITY_TYPES},
                    "evidence": {"type": "string"},
                },
                "required": ["name", "entity_type", "evidence"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"},
                    "relation_type": {"type": "string", "enum": RELATION_TYPES},
                    "target": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["source", "relation_type", "target", "evidence"],
            },
        },
    },
    "required": ["entities", "relationships"],
}


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def normalize_text(value: str) -> str:
    cleaned = value.lower().replace("â€™", "'").replace("’", "'").replace("—", "-").replace("–", "-")
    cleaned = re.sub(r"[^a-z0-9$%+\-./ ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_name(value: str) -> str:
    return normalize_text(value)


def normalize_relation(source: str, relation_type: str, target: str) -> tuple[str, str, str]:
    return (normalize_name(source), relation_type.strip().lower(), normalize_name(target))


def is_grounded(passage_text: str, evidence: str) -> bool:
    normalized_passage = normalize_text(passage_text)
    normalized_evidence = normalize_text(evidence)
    if not normalized_evidence:
        return False
    return normalized_evidence in normalized_passage


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_M_TOKEN.get(model)
    if not pricing:
        return 0.0
    return ((prompt_tokens / 1_000_000) * pricing["input"]) + (
        (completion_tokens / 1_000_000) * pricing["output"]
    )


def load_cases(cases_path: Path = CASES_PATH, chunk_artifact_path: Path = CHUNK_ARTIFACT_PATH) -> list[dict[str, Any]]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    chunk_artifact = json.loads(chunk_artifact_path.read_text(encoding="utf-8"))
    section_lookup = {section["section_title"]: section for section in chunk_artifact["sections"]}

    enriched_cases: list[dict[str, Any]] = []
    for case in cases:
        section = section_lookup[case["section_title"]]
        enriched = dict(case)
        enriched["text"] = section["text"]
        enriched_cases.append(enriched)
    return enriched_cases


def dedupe_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for entity in entities:
        key = normalize_name(entity.get("name", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def dedupe_relationships(relationships: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for rel in relationships:
        key = normalize_relation(rel.get("source", ""), rel.get("relation_type", ""), rel.get("target", ""))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped


def normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    entities = []
    for entity in payload.get("entities", []):
        name = str(entity.get("name", "")).strip()
        evidence = str(entity.get("evidence", "")).strip()
        entity_type = str(entity.get("entity_type", "other")).strip()
        if not name:
            continue
        if entity_type not in ENTITY_TYPES:
            entity_type = "other"
        entities.append({"name": name, "entity_type": entity_type, "evidence": evidence})

    relationships = []
    for rel in payload.get("relationships", []):
        source = str(rel.get("source", "")).strip()
        relation_type = str(rel.get("relation_type", "")).strip().lower()
        target = str(rel.get("target", "")).strip()
        evidence = str(rel.get("evidence", "")).strip()
        if not source or not relation_type or not target:
            continue
        if relation_type not in RELATION_TYPES:
            continue
        relationships.append(
            {
                "source": source,
                "relation_type": relation_type,
                "target": target,
                "evidence": evidence,
            }
        )

    entities = dedupe_entities(entities)
    relationships = dedupe_relationships(relationships)
    return {"entities": entities, "relationships": relationships}


def extract_graph_entities_and_relationships(
    passage_text: str,
    section_title: str,
    model: str,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "graph_extraction",
                "schema": JSON_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "section_title": section_title,
                        "passage_text": passage_text,
                        "entity_type_options": ENTITY_TYPES,
                        "relation_type_options": RELATION_TYPES,
                    }
                ),
            },
        ],
    )

    content = completion.choices[0].message.content or "{}"
    payload = normalize_extraction_payload(json.loads(content))
    usage = completion.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    return {
        "entities": payload["entities"],
        "relationships": payload["relationships"],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens),
        },
    }


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_case(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_entities = {normalize_name(name) for name in case["expected_entities"]}
    actual_entities = [normalize_name(entity["name"]) for entity in actual["entities"]]
    actual_entity_set = set(actual_entities)
    matched_entities = sorted(expected_entities & actual_entity_set)

    expected_relations = {
        normalize_relation(rel["source"], rel["relation_type"], rel["target"])
        for rel in case["expected_relationships"]
    }
    actual_relations = [
        normalize_relation(rel["source"], rel["relation_type"], rel["target"])
        for rel in actual["relationships"]
    ]
    actual_relation_set = set(actual_relations)
    matched_relations = sorted(expected_relations & actual_relation_set)

    grounded_entities = [
        entity for entity in actual["entities"] if is_grounded(case["text"], entity.get("evidence", ""))
    ]
    grounded_relationships = [
        rel for rel in actual["relationships"] if is_grounded(case["text"], rel.get("evidence", ""))
    ]

    entity_precision = safe_divide(len(matched_entities), len(actual["entities"]))
    entity_recall = safe_divide(len(matched_entities), len(expected_entities))
    relation_precision = safe_divide(len(matched_relations), len(actual["relationships"]))
    relation_recall = safe_divide(len(matched_relations), len(expected_relations))
    entity_grounding_precision = safe_divide(len(grounded_entities), len(actual["entities"]))
    relation_grounding_precision = safe_divide(len(grounded_relationships), len(actual["relationships"]))

    quality_score = (
        (0.30 * entity_recall)
        + (0.25 * relation_recall)
        + (0.15 * entity_precision)
        + (0.10 * relation_precision)
        + (0.10 * entity_grounding_precision)
        + (0.10 * relation_grounding_precision)
    )

    return {
        "case_id": case["case_id"],
        "section_title": case["section_title"],
        "page_start": case["page_start"],
        "page_end": case["page_end"],
        "expected_entity_count": len(expected_entities),
        "expected_relation_count": len(expected_relations),
        "actual_entity_count": len(actual["entities"]),
        "actual_relation_count": len(actual["relationships"]),
        "matched_entities": matched_entities,
        "matched_relations": [
            {"source": source, "relation_type": relation_type, "target": target}
            for source, relation_type, target in matched_relations
        ],
        "entity_precision": round(entity_precision, 4),
        "entity_recall": round(entity_recall, 4),
        "relation_precision": round(relation_precision, 4),
        "relation_recall": round(relation_recall, 4),
        "entity_grounding_precision": round(entity_grounding_precision, 4),
        "relation_grounding_precision": round(relation_grounding_precision, 4),
        "quality_score": round(quality_score, 4),
    }


def benchmark_model(model: str, cases: list[dict[str, Any]], client: OpenAI | None = None) -> dict[str, Any]:
    case_results: list[dict[str, Any]] = []
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost_usd": 0.0,
    }

    for case in cases:
        extraction = extract_graph_entities_and_relationships(
            passage_text=case["text"],
            section_title=case["section_title"],
            model=model,
            client=client,
        )
        evaluation = evaluate_case(case, extraction)
        case_results.append(
            {
                "case_id": case["case_id"],
                "section_title": case["section_title"],
                "page_start": case["page_start"],
                "page_end": case["page_end"],
                "usage": extraction["usage"],
                "evaluation": evaluation,
                "entities": extraction["entities"],
                "relationships": extraction["relationships"],
            }
        )
        totals["prompt_tokens"] += extraction["usage"]["prompt_tokens"]
        totals["completion_tokens"] += extraction["usage"]["completion_tokens"]
        totals["estimated_cost_usd"] += extraction["usage"]["estimated_cost_usd"]

    avg_quality = safe_divide(
        sum(case_result["evaluation"]["quality_score"] for case_result in case_results),
        len(case_results),
    )
    avg_entity_recall = safe_divide(
        sum(case_result["evaluation"]["entity_recall"] for case_result in case_results),
        len(case_results),
    )
    avg_relation_recall = safe_divide(
        sum(case_result["evaluation"]["relation_recall"] for case_result in case_results),
        len(case_results),
    )
    cost_efficiency = safe_divide(avg_quality, totals["estimated_cost_usd"])

    return {
        "model": model,
        "summary": {
            "case_count": len(case_results),
            "avg_quality_score": round(avg_quality, 4),
            "avg_entity_recall": round(avg_entity_recall, 4),
            "avg_relation_recall": round(avg_relation_recall, 4),
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["prompt_tokens"] + totals["completion_tokens"],
            "estimated_cost_usd": round(totals["estimated_cost_usd"], 6),
            "cost_efficiency_score": round(cost_efficiency, 2),
        },
        "case_results": case_results,
    }


def rank_models(model_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        model_results,
        key=lambda item: (
            item["summary"]["avg_quality_score"],
            item["summary"]["avg_relation_recall"],
            -item["summary"]["estimated_cost_usd"],
        ),
        reverse=True,
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def write_report(project_root: Path, run_id: str, model_results: list[dict[str, Any]]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "graph_extraction"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "graph_entity_extraction_benchmark_report.json"
    ranked_results = rank_models(model_results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_scope": "first_substantive_pages_of_microsoft_fy2025_summary",
        "models_tested": [item["model"] for item in ranked_results],
        "results": ranked_results,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark candidate LLMs for GraphRAG entity extraction.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--run-id", default="component5_graph_entity_model_benchmark")
    args = parser.parse_args()

    cases = load_cases()
    model_results = [benchmark_model(model, cases) for model in args.models]
    report_path = write_report(PROJECT_ROOT, args.run_id, model_results)
    ranked_results = rank_models(model_results)

    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "best_model": ranked_results[0]["model"],
                "ranked_models": [
                    {
                        "model": item["model"],
                        "avg_quality_score": item["summary"]["avg_quality_score"],
                        "estimated_cost_usd": item["summary"]["estimated_cost_usd"],
                    }
                    for item in ranked_results
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
