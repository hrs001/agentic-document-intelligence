import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_document_intelligence.scripts.execute_multi_source_orchestration import load_project_env


MODEL_NAME = "gpt-5-mini"
MAX_RECENT_TURNS = 10

SYSTEM_PROMPT = """You rewrite a user's latest chat turn into a standalone retrieval-friendly query for a RAG system.
Return only valid JSON.

Rules:
- Use recent conversation turns only when they are clearly needed to resolve references like "it", "that segment", "the previous one", or "based on my previous question".
- Preserve the user's intended meaning exactly.
- Do not add new asks or assumptions that are not supported by recent turns.
- If the latest user turn is already standalone, keep it unchanged.
- If the history is insufficient or ambiguous, keep the original user query unchanged and lower confidence.
- Prefer explicit entity names, metrics, and time periods in the rewritten query.
- Do not invent new document sources, channels, venues, or datasets such as earnings calls, press releases, or investor presentations unless the user explicitly mentioned them.
- Do not convert a product or business into a segment unless the history explicitly established the segment name. If the exact segment is not already known, say "the Microsoft segment associated with <entity>".
- Keep the rewrite aligned to the existing document/Q&A context rather than broadening the scope.
- Do not answer the question. Only rewrite it.

Examples:
- Previous turn says: "GitHub is included in Intelligent Cloud, and Intelligent Cloud reported FY2025 revenue of $108,610 million."
  New turn: "How does that compare with LinkedIn's segment?"
  Good rewrite: "Compare Intelligent Cloud's FY2025 revenue with the Microsoft segment associated with LinkedIn, including which segment includes LinkedIn and that segment's FY2025 revenue."
"""

JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resolved_query": {"type": "string"},
        "used_history": {"type": "boolean"},
        "referenced_turn_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "clarification_needed": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "resolved_query",
        "used_history",
        "referenced_turn_ids",
        "confidence",
        "clarification_needed",
        "notes",
    ],
}


def _trim_recent_turns(recent_turns: list[dict[str, Any]], max_recent_turns: int) -> list[dict[str, Any]]:
    trimmed = recent_turns[-max_recent_turns:]
    normalized = []
    for turn in trimmed:
        normalized.append(
            {
                "turn_id": str(turn.get("turn_id", "")).strip(),
                "user_query": str(turn.get("user_query", "")).strip(),
                "resolved_query": str(turn.get("resolved_query", "")).strip(),
                "answer_summary": str(turn.get("answer_summary", "")).strip(),
                "sources_used": list(turn.get("sources_used", [])),
            }
        )
    return normalized


def normalize_resolution(
    query: str,
    payload: dict[str, Any],
    recent_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    resolved_query = str(payload.get("resolved_query", "")).strip() or query.strip()
    referenced_turn_ids = [
        str(turn_id).strip()
        for turn_id in payload.get("referenced_turn_ids", [])
        if str(turn_id).strip()
    ]
    valid_turn_ids = {str(turn.get("turn_id", "")).strip() for turn in recent_turns}
    referenced_turn_ids = [turn_id for turn_id in referenced_turn_ids if turn_id in valid_turn_ids]
    used_history = bool(payload.get("used_history", False) and referenced_turn_ids)

    return {
        "original_query": query.strip(),
        "resolved_query": resolved_query,
        "used_history": used_history,
        "referenced_turn_ids": referenced_turn_ids,
        "confidence": payload.get("confidence", "low"),
        "clarification_needed": bool(payload.get("clarification_needed", False)),
        "notes": str(payload.get("notes", "")).strip(),
        "recent_turn_count": len(recent_turns),
    }


def resolve_conversational_query(
    query: str,
    recent_turns: list[dict[str, Any]],
    model: str = MODEL_NAME,
    client: OpenAI | None = None,
    max_recent_turns: int = MAX_RECENT_TURNS,
) -> dict[str, Any]:
    normalized_turns = _trim_recent_turns(recent_turns, max_recent_turns=max_recent_turns)
    if not normalized_turns:
        return {
            "original_query": query.strip(),
            "resolved_query": query.strip(),
            "used_history": False,
            "referenced_turn_ids": [],
            "confidence": "high",
            "clarification_needed": False,
            "notes": "No recent turns available.",
            "recent_turn_count": 0,
        }

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
                "name": "conversation_query_resolution",
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
                        "latest_user_query": query,
                        "recent_turns": normalized_turns,
                    }
                ),
            },
        ],
    )
    content = completion.choices[0].message.content or "{}"
    payload = json.loads(content)
    return normalize_resolution(query, payload, normalized_turns)
