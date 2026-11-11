import json
import unittest

from agentic_document_intelligence.scripts.conversation_query_resolution import (
    _trim_recent_turns,
    normalize_resolution,
    resolve_conversational_query,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **_: object) -> _FakeCompletion:
        return _FakeCompletion(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeChatCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


class ConversationQueryResolutionTest(unittest.TestCase):
    def test_trim_recent_turns_keeps_latest(self):
        turns = [
            {"turn_id": f"turn-{idx}", "user_query": f"q{idx}", "resolved_query": f"r{idx}", "answer_summary": "a", "sources_used": []}
            for idx in range(1, 13)
        ]
        trimmed = _trim_recent_turns(turns, max_recent_turns=10)
        self.assertEqual(len(trimmed), 10)
        self.assertEqual(trimmed[0]["turn_id"], "turn-3")
        self.assertEqual(trimmed[-1]["turn_id"], "turn-12")

    def test_normalize_resolution_filters_unknown_turn_ids(self):
        payload = {
            "resolved_query": "What was Intelligent Cloud operating income in FY2025?",
            "used_history": True,
            "referenced_turn_ids": ["turn-2", "turn-99"],
            "confidence": "high",
            "clarification_needed": False,
            "notes": "Resolved pronoun.",
        }
        result = normalize_resolution(
            "What about its operating income?",
            payload,
            [
                {"turn_id": "turn-1"},
                {"turn_id": "turn-2"},
            ],
        )
        self.assertEqual(result["referenced_turn_ids"], ["turn-2"])
        self.assertTrue(result["used_history"])

    def test_resolve_conversational_query_returns_passthrough_without_history(self):
        result = resolve_conversational_query("Who is the CEO of Microsoft?", recent_turns=[], client=None)
        self.assertEqual(result["resolved_query"], "Who is the CEO of Microsoft?")
        self.assertFalse(result["used_history"])

    def test_resolve_conversational_query_rewrites_follow_up(self):
        fake_payload = {
            "resolved_query": "What was the FY2025 operating income of Intelligent Cloud?",
            "used_history": True,
            "referenced_turn_ids": ["turn-1"],
            "confidence": "high",
            "clarification_needed": False,
            "notes": "Resolved 'its' to Intelligent Cloud.",
        }
        recent_turns = [
            {
                "turn_id": "turn-1",
                "user_query": "Which Microsoft segment had the highest revenue in FY2025?",
                "resolved_query": "Which Microsoft segment had the highest revenue in FY2025?",
                "answer_summary": "Intelligent Cloud had the highest revenue.",
                "sources_used": ["sql_structured"],
            }
        ]
        result = resolve_conversational_query(
            "What was its operating income?",
            recent_turns=recent_turns,
            client=_FakeClient(json.dumps(fake_payload)),
        )
        self.assertEqual(result["resolved_query"], "What was the FY2025 operating income of Intelligent Cloud?")
        self.assertTrue(result["used_history"])
        self.assertEqual(result["referenced_turn_ids"], ["turn-1"])


if __name__ == "__main__":
    unittest.main()
