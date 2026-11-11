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


DEFAULT_FUSED_INPUT = (
    "artifacts/experiments/component8_cross_source_evidence_fusion_sample/"
    "answer_context/cross_source_fused_evidence.json"
)
DEFAULT_ANSWER_INPUT = (
    "artifacts/experiments/component8_grounded_answer_generation/"
    "answers/grounded_answer_report.json"
)
DEFAULT_MODEL = "gpt-5-mini"

CRITIQUE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "grounded": {"type": "boolean"},
        "complete": {"type": "boolean"},
        "needs_correction": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "issue_summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "issue_type": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "description": {"type": "string"},
                    "affected_sub_queries": {"type": "array", "items": {"type": "string"}},
                    "repair_action": {"type": "string"},
                },
                "required": ["issue_type", "severity", "description", "affected_sub_queries", "repair_action"],
            },
        },
        "repair_plan": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "grounded",
        "complete",
        "needs_correction",
        "confidence",
        "issue_summary",
        "strengths",
        "issues",
        "repair_plan",
    ],
}

SYSTEM_PROMPT = """You are the self-reflective critique stage of a production RAG system.
Return only valid JSON.

Your job:
- Judge whether the answer is grounded in the provided evidence and whether it fully answers the user query.
- Use the deterministic signals as hard warnings, not optional hints.
- Prefer high severity when the answer makes an unsupported claim, ignores a required sub-query, or fails to acknowledge a conflict.
- If the answer is acceptable, keep issues short and avoid inventing problems.
- If correction is needed, propose targeted repair actions instead of generic retries.
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["result"] if "result" in payload else payload


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def extract_inline_fact_ids(answer_markdown: str) -> list[str]:
    return re.findall(r"\[([^\[\]]+::[^\[\]]+)\]", answer_markdown or "")


def deterministic_reflection_checks(fused_bundle: dict[str, Any], answer_result: dict[str, Any]) -> dict[str, Any]:
    answer_text = answer_result.get("answer_markdown", "")
    inline_fact_ids = extract_inline_fact_ids(answer_text)
    used_fact_ids = list(answer_result.get("used_fact_ids", []))
    used_fact_id_set = set(used_fact_ids)
    inline_fact_id_set = set(inline_fact_ids)

    fused_facts = {fact["fact_id"]: fact for fact in fused_bundle.get("normalized_facts", [])}
    unanswered = [str(item).strip() for item in answer_result.get("unanswered_sub_queries", []) if str(item).strip()]
    missing_inline_for_used = [fact_id for fact_id in used_fact_ids if fact_id not in inline_fact_id_set]
    inline_not_declared = [fact_id for fact_id in inline_fact_ids if fact_id not in used_fact_id_set]
    unknown_used_fact_ids = [fact_id for fact_id in used_fact_ids if fact_id not in fused_facts]

    answered_sub_query_ids = set()
    used_source_types = set()
    for fact_id in used_fact_ids:
        fact = fused_facts.get(fact_id)
        if not fact:
            continue
        answered_sub_query_ids.add(fact["sub_query_id"])
        used_source_types.add(fact["source_type"])

    uncovered_sub_queries = [
        sub_query["sub_query_id"]
        for sub_query in fused_bundle.get("sub_query_fusions", [])
        if sub_query["sub_query_id"] not in answered_sub_query_ids
    ]

    conflict_fact_ids = {
        fact_id
        for conflict in fused_bundle.get("conflict_signals", [])
        for fact_id in conflict.get("fact_ids", [])
    }
    used_conflict_fact_ids = [fact_id for fact_id in used_fact_ids if fact_id in conflict_fact_ids]
    conflict_acknowledged = "conflict" in normalize_text(answer_text) or "uncertain" in normalize_text(answer_text)

    return {
        "inline_fact_ids": inline_fact_ids,
        "missing_inline_for_used": missing_inline_for_used,
        "inline_not_declared": inline_not_declared,
        "unknown_used_fact_ids": unknown_used_fact_ids,
        "unanswered_sub_queries": unanswered,
        "uncovered_sub_queries": uncovered_sub_queries,
        "used_source_types": sorted(used_source_types),
        "has_non_graph_corroboration": any(source != "graph_relationships" for source in used_source_types),
        "used_conflict_fact_ids": used_conflict_fact_ids,
        "conflict_acknowledged": conflict_acknowledged,
        "needs_correction": bool(
            missing_inline_for_used
            or inline_not_declared
            or unknown_used_fact_ids
            or unanswered
            or (used_conflict_fact_ids and not conflict_acknowledged and not any(source != "graph_relationships" for source in used_source_types))
        ),
    }


def build_critique_input(
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    deterministic_signals: dict[str, Any],
    max_used_fact_count: int = 12,
) -> dict[str, Any]:
    used_fact_ids = list(answer_result.get("used_fact_ids", []))
    fused_facts = {fact["fact_id"]: fact for fact in fused_bundle.get("normalized_facts", [])}
    used_facts = [fused_facts[fact_id] for fact_id in used_fact_ids[:max_used_fact_count] if fact_id in fused_facts]
    return {
        "original_query": fused_bundle["original_query"],
        "answer_markdown": answer_result.get("answer_markdown", ""),
        "confidence": answer_result.get("confidence", "unknown"),
        "sub_queries": [
            {
                "sub_query_id": sub_query["sub_query_id"],
                "original_sub_query": sub_query["original_sub_query"],
                "resolved_sub_query": sub_query.get("resolved_sub_query", sub_query["original_sub_query"]),
                "active_sources": sub_query.get("active_sources", []),
                "fact_count": sub_query.get("fact_count", 0),
                "conflict_signals": sub_query.get("conflict_signals", []),
            }
            for sub_query in fused_bundle.get("sub_query_fusions", [])
        ],
        "used_fact_ids": used_fact_ids,
        "used_facts": [
            {
                "fact_id": fact["fact_id"],
                "source_type": fact["source_type"],
                "fact_type": fact["fact_type"],
                "summary": fact["summary"],
                "sub_query_id": fact["sub_query_id"],
            }
            for fact in used_facts
        ],
        "deterministic_signals": deterministic_signals,
    }


def sanitize_critique_payload(payload: dict[str, Any]) -> dict[str, Any]:
    issues = []
    for issue in payload.get("issues", []):
        issues.append(
            {
                "issue_type": str(issue.get("issue_type", "")).strip(),
                "severity": str(issue.get("severity", "low")).strip() or "low",
                "description": str(issue.get("description", "")).strip(),
                "affected_sub_queries": [
                    str(item).strip() for item in issue.get("affected_sub_queries", []) if str(item).strip()
                ],
                "repair_action": str(issue.get("repair_action", "")).strip(),
            }
        )
    return {
        "grounded": bool(payload.get("grounded", False)),
        "complete": bool(payload.get("complete", False)),
        "needs_correction": bool(payload.get("needs_correction", False)),
        "confidence": str(payload.get("confidence", "low")).strip() or "low",
        "issue_summary": str(payload.get("issue_summary", "")).strip(),
        "strengths": [str(item).strip() for item in payload.get("strengths", []) if str(item).strip()],
        "issues": issues,
        "repair_plan": [str(item).strip() for item in payload.get("repair_plan", []) if str(item).strip()],
    }


def apply_deterministic_overrides(llm_critique: dict[str, Any], deterministic_signals: dict[str, Any]) -> dict[str, Any]:
    critique = dict(llm_critique)
    issues = list(critique.get("issues", []))
    repair_plan = list(critique.get("repair_plan", []))

    if deterministic_signals["missing_inline_for_used"]:
        issues.append(
            {
                "issue_type": "missing_inline_citations",
                "severity": "high",
                "description": "Some used facts are not cited inline in the answer text.",
                "affected_sub_queries": deterministic_signals["uncovered_sub_queries"],
                "repair_action": "Regenerate the answer with inline citations for every used fact.",
            }
        )
        repair_plan.append("Regenerate the answer with inline fact-id citations for all used facts.")
    if deterministic_signals["inline_not_declared"] or deterministic_signals["unknown_used_fact_ids"]:
        issues.append(
            {
                "issue_type": "citation_fact_mismatch",
                "severity": "high",
                "description": "The answer references fact ids that are not declared or not present in the fused evidence.",
                "affected_sub_queries": deterministic_signals["uncovered_sub_queries"],
                "repair_action": "Remove invalid citations and regenerate from valid fused facts only.",
            }
        )
        repair_plan.append("Strip invalid fact ids and regenerate strictly from allowed fused facts.")
    if deterministic_signals["unanswered_sub_queries"]:
        issues.append(
            {
                "issue_type": "coverage_gap",
                "severity": "high",
                "description": "The answer leaves at least one sub-query explicitly unanswered.",
                "affected_sub_queries": deterministic_signals["unanswered_sub_queries"],
                "repair_action": "Run a targeted correction pass for the missing sub-query coverage.",
            }
        )
        repair_plan.append("Run targeted answer repair for explicitly unanswered sub-queries.")
    if (
        deterministic_signals["used_conflict_fact_ids"]
        and not deterministic_signals["conflict_acknowledged"]
        and not deterministic_signals["has_non_graph_corroboration"]
    ):
        issues.append(
            {
                "issue_type": "unacknowledged_conflict",
                "severity": "high",
                "description": "The answer used facts involved in a conflict but did not acknowledge uncertainty or conflict.",
                "affected_sub_queries": deterministic_signals["uncovered_sub_queries"],
                "repair_action": "Regenerate with explicit uncertainty or run corrective retrieval to resolve the conflict.",
            }
        )
        repair_plan.append("Acknowledge evidence conflict explicitly or trigger corrective retrieval.")

    needs_correction = bool(deterministic_signals["needs_correction"] or critique.get("needs_correction"))
    grounded = bool(critique.get("grounded")) and not bool(
        deterministic_signals["unknown_used_fact_ids"] or deterministic_signals["inline_not_declared"]
    )
    complete = bool(critique.get("complete")) and not bool(
        deterministic_signals["unanswered_sub_queries"]
    )

    return {
        **critique,
        "grounded": grounded,
        "complete": complete,
        "needs_correction": needs_correction,
        "issues": issues,
        "repair_plan": list(dict.fromkeys(repair_plan)),
    }


def critique_grounded_answer(
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    deterministic_signals = deterministic_reflection_checks(fused_bundle, answer_result)
    openai_client = client or OpenAI(api_key=api_key)
    critique_input = build_critique_input(fused_bundle, answer_result, deterministic_signals)
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "self_reflective_answer_critique",
                "schema": CRITIQUE_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(critique_input)},
        ],
    )
    payload = json.loads(completion.choices[0].message.content or "{}")
    sanitized = sanitize_critique_payload(payload)
    merged = apply_deterministic_overrides(sanitized, deterministic_signals)
    return {
        "original_query": fused_bundle["original_query"],
        "answer_model": answer_result.get("model", "unknown"),
        "critique_model": model,
        "deterministic_signals": deterministic_signals,
        "llm_critique": sanitized,
        "final_critique": merged,
        "usage": getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else {},
    }


def write_report(project_root: Path, run_id: str, result: dict[str, Any]) -> Path:
    output_dir = (
        project_root
        / "agentic_document_intelligence"
        / "artifacts"
        / "experiments"
        / run_id
        / "answers"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "self_reflective_answer_critique.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run self-reflective critique on a grounded answer.")
    parser.add_argument("--fused-input", default=DEFAULT_FUSED_INPUT)
    parser.add_argument("--answer-input", default=DEFAULT_ANSWER_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="component8_self_reflective_answer_critique")
    args = parser.parse_args()

    fused_bundle = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.fused_input)
    answer_result = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.answer_input)
    result = critique_grounded_answer(fused_bundle, answer_result, model=args.model)
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "needs_correction": result["final_critique"]["needs_correction"],
                "grounded": result["final_critique"]["grounded"],
                "complete": result["final_critique"]["complete"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
