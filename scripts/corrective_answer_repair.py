import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agentic_document_intelligence.scripts.generate_grounded_answer import (
    ANSWER_SCHEMA,
    build_answer_input,
    sanitize_answer_payload,
)
from agentic_document_intelligence.scripts.self_reflective_answer_critique import (
    deterministic_reflection_checks,
    load_result,
)


DEFAULT_FUSED_INPUT = (
    "artifacts/experiments/component8_cross_source_evidence_fusion_sample/"
    "answer_context/cross_source_fused_evidence.json"
)
DEFAULT_ANSWER_INPUT = (
    "artifacts/experiments/component8_grounded_answer_generation_live_smoke/"
    "answers/grounded_answer_report.json"
)
DEFAULT_CRITIQUE_INPUT = (
    "artifacts/experiments/component8_self_reflective_answer_critique_live_smoke/"
    "answers/self_reflective_answer_critique.json"
)
DEFAULT_MODEL = "gpt-5.1"

SYSTEM_PROMPT = """You are the corrective repair stage of a production RAG system.
Return only valid JSON.

You will receive:
- the original user query
- the current answer
- the fused evidence facts
- the self-reflective critique and repair plan

Rules:
- Repair the answer only using the provided fused evidence.
- Do not invent facts, citations, or page numbers.
- Every substantial claim in answer_markdown must include inline fact-id citations.
- Prefer minimal targeted repair over rewriting everything unnecessarily.
- If a gap cannot be repaired from the available evidence, state that limitation clearly.
- used_fact_ids must contain only valid allowed fact ids.
"""


def load_project_env() -> None:
    load_dotenv(PROJECT_ROOT / "agentic_document_intelligence" / ".env", override=False)


def choose_repair_strategy(critique_result: dict[str, Any]) -> dict[str, Any]:
    final_critique = critique_result.get("final_critique", {})
    issues = final_critique.get("issues", [])
    issue_types = {str(issue.get("issue_type", "")).strip() for issue in issues}
    needs_correction = bool(final_critique.get("needs_correction", False))

    if not needs_correction:
        return {
            "strategy": "no_op",
            "reason": "The critique says the answer does not need correction.",
            "focus": [],
        }
    if "coverage_gap" in issue_types:
        return {
            "strategy": "targeted_coverage_repair",
            "reason": "The answer missed at least one sub-query or left part of the question unsupported.",
            "focus": final_critique.get("repair_plan", []),
        }
    if "unacknowledged_conflict" in issue_types:
        return {
            "strategy": "conflict_aware_regeneration",
            "reason": "The answer used conflicting evidence without acknowledging uncertainty.",
            "focus": final_critique.get("repair_plan", []),
        }
    if "missing_inline_citations" in issue_types or "citation_fact_mismatch" in issue_types:
        return {
            "strategy": "citation_strict_regeneration",
            "reason": "The answer needs stricter citation-grounded regeneration.",
            "focus": final_critique.get("repair_plan", []),
        }
    return {
        "strategy": "general_grounded_regeneration",
        "reason": "The answer needs correction, but no narrower repair class dominated.",
        "focus": final_critique.get("repair_plan", []),
    }


def build_repair_input(
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    critique_result: dict[str, Any],
    strategy: dict[str, Any],
    max_facts_per_sub_query: int = 10,
) -> dict[str, Any]:
    answer_input = build_answer_input(fused_bundle, max_facts_per_sub_query=max_facts_per_sub_query)
    return {
        "original_query": fused_bundle["original_query"],
        "current_answer": answer_result.get("answer_markdown", ""),
        "current_used_fact_ids": answer_result.get("used_fact_ids", []),
        "strategy": strategy,
        "critique": critique_result.get("final_critique", {}),
        "deterministic_signals": critique_result.get("deterministic_signals", {}),
        "answer_input": answer_input,
    }


def repair_grounded_answer(
    fused_bundle: dict[str, Any],
    answer_result: dict[str, Any],
    critique_result: dict[str, Any],
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    strategy = choose_repair_strategy(critique_result)
    if strategy["strategy"] == "no_op":
        final_signals = deterministic_reflection_checks(fused_bundle, answer_result)
        return {
            "original_query": fused_bundle["original_query"],
            "repair_model": model,
            "repair_strategy": strategy,
            "repair_applied": False,
            "repaired_answer": answer_result,
            "post_repair_deterministic_signals": final_signals,
            "repair_success": not final_signals["needs_correction"],
            "usage": {},
        }

    load_project_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key and client is None:
        raise RuntimeError("OPENAI_API_KEY is not set in agentic_document_intelligence/.env")

    openai_client = client or OpenAI(api_key=api_key)
    repair_input = build_repair_input(fused_bundle, answer_result, critique_result, strategy)
    fact_source_map = {
        item["fact_id"]: item["source_type"] for item in repair_input["answer_input"]["fact_catalog"]
    }
    completion = openai_client.chat.completions.create(
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "corrective_answer_repair",
                "schema": ANSWER_SCHEMA,
                "strict": True,
            },
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(repair_input)},
        ],
    )
    payload = json.loads(completion.choices[0].message.content or "{}")
    sanitized = sanitize_answer_payload(
        payload,
        set(repair_input["answer_input"]["allowed_fact_ids"]),
        fact_source_map,
    )
    repaired_answer = {
        "original_query": fused_bundle["original_query"],
        "model": model,
        "answer_input_summary": {
            "sub_query_count": len(repair_input["answer_input"]["sub_queries"]),
            "allowed_fact_count": len(repair_input["answer_input"]["allowed_fact_ids"]),
        },
        "answer_markdown": sanitized["answer_markdown"],
        "used_fact_ids": sanitized["used_fact_ids"],
        "citations": sanitized["citations"],
        "unanswered_sub_queries": sanitized["unanswered_sub_queries"],
        "confidence": sanitized["confidence"],
        "usage": getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else {},
    }
    post_repair_signals = deterministic_reflection_checks(fused_bundle, repaired_answer)
    return {
        "original_query": fused_bundle["original_query"],
        "repair_model": model,
        "repair_strategy": strategy,
        "repair_applied": True,
        "repaired_answer": repaired_answer,
        "post_repair_deterministic_signals": post_repair_signals,
        "repair_success": not post_repair_signals["needs_correction"],
        "usage": repaired_answer["usage"],
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
    report_path = output_dir / "corrective_answer_repair.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the corrective RAG repair loop for grounded answers.")
    parser.add_argument("--fused-input", default=DEFAULT_FUSED_INPUT)
    parser.add_argument("--answer-input", default=DEFAULT_ANSWER_INPUT)
    parser.add_argument("--critique-input", default=DEFAULT_CRITIQUE_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="component8_corrective_answer_repair")
    args = parser.parse_args()

    fused_bundle = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.fused_input)
    answer_result = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.answer_input)
    critique_result = load_result(PROJECT_ROOT / "agentic_document_intelligence" / args.critique_input)
    result = repair_grounded_answer(
        fused_bundle,
        answer_result,
        critique_result,
        model=args.model,
    )
    report_path = write_report(PROJECT_ROOT, args.run_id, result)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "repair_applied": result["repair_applied"],
                "repair_success": result["repair_success"],
                "repair_strategy": result["repair_strategy"]["strategy"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
