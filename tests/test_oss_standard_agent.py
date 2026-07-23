from __future__ import annotations

import json

from rag_system.contracts import BenchmarkQuery
from rag_system.workflows.oss_standard_agent import (
    OSS_SEARCH_TOOLS,
    OssStandardAgentWorkflow,
)


class FakeResponsesClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, input_items, tools):
        self.requests.append((list(input_items), tools))
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


def benchmark_query():
    return BenchmarkQuery("q1", "private question", "answer", ("d0",), ("d0",))


def test_oss_standard_tool_definition_matches_upstream_runner():
    assert OSS_SEARCH_TOOLS == [
        {
            "type": "function",
            "name": "local_knowledge_base_retrieval",
            "description": (
                "Perform a search on a knowledge source. Returns top-5 hits with "
                "docid, score, and snippet. The snippet contains the document's "
                "contents (may be truncated based on token limits)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": (
                            "Query to search the local knowledge base for relevant "
                            "information"
                        ),
                    }
                },
                "required": ["user_query"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def test_oss_standard_agent_runs_search_then_returns_answer():
    progress_events = []
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "I should search."}
                        ],
                    },
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-1",
                        "arguments": json.dumps({"user_query": "focused search"}),
                    },
                ],
            },
            {
                "id": "resp-2",
                "status": "completed",
                "usage": {"input_tokens": 20, "output_tokens": 8},
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Explanation: result [d0].\n"
                                    "Exact Answer: answer\nConfidence: 90%"
                                ),
                            }
                        ],
                    }
                ],
            },
        ]
    )
    search = FakeSearchClient()

    record = OssStandardAgentWorkflow(
        responses,
        search,
        progress_callback=progress_events.append,
    ).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["retrieved_docids"] == ["d0", "d1", "d2", "d3", "d4"]
    assert record["result"][0]["type"] == "reasoning"
    assert record["result"][-1]["type"] == "output_text"
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert record["diagnostics"]["final_answer_validation"] == {
        "valid": True,
        "has_explanation": True,
        "has_exact_answer": True,
        "has_confidence": True,
        "citation_count": 1,
        "missing_fields": [],
    }
    search_step = record["diagnostics"]["search_steps"][0]
    assert search_step["evidence"]["turn_recall_at_5"] == 1.0
    assert search_step["evidence"]["turn_ndcg_at_5"] == 1.0
    assert search_step["evidence"]["cumulative_recall"] == 1.0
    assert search.queries == ["focused search"]
    second_input = responses.requests[1][0]
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call-1"
    assert second_input[-1]["output"].startswith('[\n  {\n    "docid": "d0"')
    assert [event["event"] for event in progress_events] == [
        "generation_started",
        "generation_completed",
        "search_started",
        "search_completed",
        "generation_started",
        "generation_completed",
    ]


def test_oss_standard_agent_normalizes_known_mcp_search_alias():
    progress_events = []
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-1",
                        "arguments": json.dumps({"user_query": "first search"}),
                    }
                ],
            },
            {
                "id": "resp-2",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "reasoning",
                        "content": [
                            {"type": "reasoning_text", "text": "Search again."}
                        ],
                    },
                    {
                        "type": "mcp_call",
                        "id": "mcp-2",
                        "status": "completed",
                        "server_label": "browser",
                        "name": "search",
                        "arguments": json.dumps(
                            {
                                "query": "second search",
                                "topn": 10,
                                "source": "news",
                            }
                        ),
                    },
                ],
            },
            {
                "id": "resp-3",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Explanation: result [d0].\nExact Answer: answer",
                            }
                        ],
                    }
                ],
            },
        ]
    )
    search = FakeSearchClient()

    record = OssStandardAgentWorkflow(
        responses,
        search,
        progress_callback=progress_events.append,
    ).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 2}
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert search.queries == ["first search", "second search"]

    third_input = responses.requests[2][0]
    assert not any(item.get("type") == "mcp_call" for item in third_input)
    assert third_input[-2] == {
        "type": "function_call",
        "id": "fc_mcp_compat_2_1",
        "call_id": "call_mcp_compat_2_1",
        "name": "local_knowledge_base_retrieval",
        "arguments": '{"user_query":"second search"}',
        "status": "completed",
    }
    assert third_input[-1]["type"] == "function_call_output"
    assert third_input[-1]["call_id"] == "call_mcp_compat_2_1"

    second_step = record["diagnostics"]["generation_steps"][1]
    assert second_step["output_item_types"] == ["reasoning", "mcp_call"]
    assert second_step["output_item_details"][1] == {
        "type": "mcp_call",
        "id": "mcp-2",
        "status": "completed",
        "name": "search",
        "server_label": "browser",
        "argument_keys": ["query", "source", "topn"],
        "has_output": False,
        "has_error": False,
    }
    assert second_step["mcp_search_aliases"] == [
        {
            "item_index": 1,
            "server_label": "browser",
            "name": "search",
            "normalized_name": "local_knowledge_base_retrieval",
            "call_id": "call_mcp_compat_2_1",
        }
    ]
    generation_event = [
        event
        for event in progress_events
        if event["event"] == "generation_completed" and event["turn"] == 2
    ][0]
    assert generation_event["tool_call_count"] == 1
    assert generation_event["mcp_search_aliases"]


def test_oss_standard_agent_rejects_non_search_mcp_call_explicitly():
    progress_events = []
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "mcp_call",
                        "id": "mcp-1",
                        "status": "completed",
                        "server_label": "browser",
                        "name": "open",
                        "arguments": json.dumps({"url": "cursor:1"}),
                    }
                ],
            }
        ]
    )

    record = OssStandardAgentWorkflow(
        responses,
        FakeSearchClient(),
        progress_callback=progress_events.append,
    ).run(benchmark_query())

    assert record["status"] == "error"
    assert record["tool_call_counts"] == {"search": 0}
    assert record["diagnostics"]["termination_reason"] == (
        "unsupported_mcp_tool_call"
    )
    assert record["error"]["type"] == "UnsupportedMcpToolCallError"
    assert "browser.open" in record["error"]["message"]
    assert [event["event"] for event in progress_events] == [
        "generation_started",
        "generation_completed",
        "tool_call_rejected",
    ]


def test_oss_standard_agent_marks_refusal_final_as_invalid_format():
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "**Explanation:** I cannot provide a "
                                    "definitive answer or credible citations."
                                ),
                            }
                        ],
                    }
                ],
            }
        ]
    )

    record = OssStandardAgentWorkflow(
        responses,
        FakeSearchClient(),
    ).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert record["diagnostics"]["final_answer_validation"] == {
        "valid": False,
        "has_explanation": True,
        "has_exact_answer": False,
        "has_confidence": False,
        "citation_count": 0,
        "missing_fields": ["Exact Answer", "Confidence", "document citation"],
    }


def test_oss_standard_agent_preserves_partial_run_after_generation_error():
    class FailingResponsesClient:
        def __init__(self):
            self.calls = 0

        def complete(self, input_items, tools):
            self.calls += 1
            if self.calls == 2:
                raise TimeoutError("timed out")
            return {
                "id": "resp-1",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-1",
                        "arguments": json.dumps({"user_query": "first search"}),
                    }
                ],
            }

    record = OssStandardAgentWorkflow(
        FailingResponsesClient(), FakeSearchClient()
    ).run(benchmark_query())

    assert record["status"] == "error"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["error"] == {"type": "TimeoutError", "message": "timed out"}
    assert record["diagnostics"]["termination_reason"] == "generation_request_error"
