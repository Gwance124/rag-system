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
    progress_events = []
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
                "finish_reason": "tool_calls",
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Explanation: result [0].\nExact Answer: answer\nConfidence: 90%",
                },
                "usage": {"prompt_tokens": 20},
                "finish_reason": "stop",
            },
        ]
    )
    search = FakeSearchClient()
    record = StandardAgentWorkflow(
        chat, search, progress_callback=progress_events.append
    ).run(query())

    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["retrieved_docids"] == ["d0", "d1", "d2", "d3", "d4"]
    assert record["result"][-1]["type"] == "output_text"
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert record["diagnostics"]["generation_steps"][0]["finish_reason"] == "tool_calls"
    search_step = record["diagnostics"]["search_steps"][0]
    assert search_step["evidence"]["turn_recall_at_5"] == 1.0
    assert search_step["evidence"]["turn_ndcg_at_5"] == 1.0
    assert search_step["evidence"]["new_hits"] == 1
    assert search_step["evidence"]["cumulative_recall"] == 1.0
    assert [event["event"] for event in progress_events] == [
        "generation_started",
        "generation_completed",
        "search_started",
        "search_completed",
        "generation_started",
        "generation_completed",
    ]
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


def test_standard_agent_records_output_token_exhaustion():
    chat = FakeChatClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "Still thinking.",
                },
                "usage": {"completion_tokens": 10_000},
                "finish_reason": "length",
            }
        ]
    )
    record = StandardAgentWorkflow(chat, FakeSearchClient()).run(query())

    assert record["status"] == "incomplete"
    assert record["result"][0]["type"] == "reasoning"
    assert record["diagnostics"]["termination_reason"] == "max_output_tokens"


def test_standard_agent_preserves_partial_run_after_generation_error():
    class FailingChatClient:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, tools):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("timed out")
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"query": "first search"}),
                            },
                        }
                    ],
                },
                "usage": {"completion_tokens": 10},
                "finish_reason": "tool_calls",
            }

    record = StandardAgentWorkflow(FailingChatClient(), FakeSearchClient()).run(query())

    assert record["status"] == "error"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["retrieved_docids"] == ["d0", "d1", "d2", "d3", "d4"]
    assert record["error"] == {"type": "TimeoutError", "message": "timed out"}
    assert record["diagnostics"]["termination_reason"] == "generation_request_error"
