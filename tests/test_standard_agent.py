from __future__ import annotations

import json

import pytest

from rag_system.contracts import BenchmarkQuery
from rag_system.workflows.standard_agent import StandardAgentWorkflow


class FakeChatClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), tools))
        return next(self.responses)


class FakeSearchClient:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            {"docid": f"d{index}", "score": 1.0 - index / 10, "snippet": "text"}
            for index in range(5)
        ]


def query():
    return BenchmarkQuery("q1", "private question", "answer", ("d0",), ("d0",))


def test_standard_agent_runs_search_then_saves_official_shape():
    chat = FakeChatClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I should search.",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"query": "focused search"}),
                            },
                        }
                    ],
                },
                "usage": {"prompt_tokens": 10},
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Explanation: result [0].\nExact Answer: answer\nConfidence: 90%",
                },
                "usage": {"prompt_tokens": 20},
            },
        ]
    )
    search = FakeSearchClient()
    record = StandardAgentWorkflow(chat, search).run(query())

    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["retrieved_docids"] == ["d0", "d1", "d2", "d3", "d4"]
    assert record["result"][-1]["type"] == "output_text"
    assert search.queries == ["focused search"]
    second_messages = chat.requests[1][0]
    assert second_messages[-1]["role"] == "tool"
    assert second_messages[-1]["tool_call_id"] == "call-1"


def test_standard_agent_rejects_get_document():
    chat = FakeChatClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "get_document",
                                "arguments": json.dumps({"docid": "d1"}),
                            },
                        }
                    ],
                },
                "usage": None,
            }
        ]
    )
    with pytest.raises(ValueError, match="only permits the search tool"):
        StandardAgentWorkflow(chat, FakeSearchClient()).run(query())


def test_standard_agent_does_not_accept_answer_without_search():
    chat = FakeChatClient(
        [{"message": {"role": "assistant", "content": "guessed"}, "usage": None}]
    )
    record = StandardAgentWorkflow(chat, FakeSearchClient()).run(query())
    assert record["status"] == "incomplete"
    assert record["result"] == []
