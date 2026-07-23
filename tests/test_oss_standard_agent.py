from __future__ import annotations

import json

from rag_system.contracts import BenchmarkQuery
from rag_system.generation.vllm_responses import VllmContextLengthError
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


def _final_answer_response(response_id: str = "resp-final"):
    return {
        "id": response_id,
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
    }


def test_oss_standard_agent_tolerates_extra_search_argument_keys():
    # Upstream oss_client.py reads arguments["user_query"] and ignores any
    # other keys; pinned v0.10.1 servers do not enforce strict schemas.
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-1",
                        "arguments": json.dumps(
                            {"user_query": "focused search", "topn": 5}
                        ),
                    },
                ],
            },
            _final_answer_response(),
        ]
    )
    search = FakeSearchClient()

    record = OssStandardAgentWorkflow(responses, search).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 1}
    assert search.queries == ["focused search"]


def test_oss_standard_agent_feeds_back_invalid_search_arguments():
    # Upstream feeds "Error executing ..." back as the tool output and keeps
    # looping; a wrong argument spelling must not end the trajectory.
    progress_events = []
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-1",
                        "arguments": json.dumps({"query": "wrong spelling"}),
                    },
                ],
            },
            {
                "id": "resp-2",
                "status": "completed",
                "usage": {"input_tokens": 15, "output_tokens": 5},
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-2",
                        "arguments": json.dumps({"user_query": "correct search"}),
                    },
                ],
            },
            _final_answer_response(),
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
    assert search.queries == ["correct search"]
    assert record["diagnostics"]["termination_reason"] == "final_answer"

    invalid_calls = record["diagnostics"]["invalid_function_calls"]
    assert len(invalid_calls) == 1
    assert invalid_calls[0]["turn"] == 1
    assert "user_query" in invalid_calls[0]["error_message"]

    second_input = responses.requests[1][0]
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call-1"
    assert second_input[-1]["output"].startswith(
        "Error executing local_knowledge_base_retrieval:"
    )
    assert "invalid_function_call" in [
        event["event"] for event in progress_events
    ]


def test_oss_standard_agent_still_errors_when_feedback_is_impossible():
    # Without a call_id there is no way to route the error back to the model.
    responses = FakeResponsesClient(
        [
            {
                "id": "resp-1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "arguments": json.dumps({"user_query": "no call id"}),
                    },
                ],
            },
        ]
    )
    search = FakeSearchClient()

    record = OssStandardAgentWorkflow(responses, search).run(benchmark_query())

    assert record["status"] == "error"
    assert record["diagnostics"]["termination_reason"] == "invalid_tool_call"
    assert search.queries == []


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


def test_oss_standard_agent_retries_after_rejecting_unsupported_mcp_call():
    """Unlike upstream's silent drop (search_agent/oss_client.py only
    recognizes type=="function_call" items and never validates an mcp_call
    it can't interpret) or this workflow's prior hard-abort, an unsupported
    mcp_call is now turned into a synthetic function_call /
    function_call_output error pair so the model can see why it failed and
    retry with the real search tool on its next turn.
    """

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
            },
            {
                "id": "resp-2",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "function_call",
                        "name": "local_knowledge_base_retrieval",
                        "call_id": "call-1",
                        "arguments": json.dumps({"user_query": "retry search"}),
                    }
                ],
            },
            {
                "id": "resp-3",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Explanation: done\n"
                                    "Exact Answer: answer\n"
                                    "Confidence: 90%\n"
                                    "[d0]"
                                ),
                            }
                        ],
                    }
                ],
            },
        ]
    )

    record = OssStandardAgentWorkflow(
        responses,
        FakeSearchClient(),
        progress_callback=progress_events.append,
    ).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["diagnostics"]["search_steps"][0]["query"] == "retry search"

    assert [event["event"] for event in progress_events][:3] == [
        "generation_started",
        "generation_completed",
        "mcp_call_rejected",
    ]
    assert progress_events[2]["rejected_calls"] == [
        {
            "item_index": 0,
            "error_type": "UnsupportedMcpToolCallError",
            "error_message": (
                "Standard OSS workflow rejected MCP recipient browser.open; "
                "only local search aliases are permitted"
            ),
        }
    ]

    # The retry request must actually include the rejection pair so the
    # model sees why its first attempt failed.
    second_request_items = responses.requests[1][0]
    rejected_items = [
        item
        for item in second_request_items
        if item.get("call_id") == "call_mcp_reject_1_0"
    ]
    assert {item["type"] for item in rejected_items} == {
        "function_call",
        "function_call_output",
    }
    rejected_output = next(
        item for item in rejected_items if item["type"] == "function_call_output"
    )
    assert "no such tool" in rejected_output["output"]
    assert "local_knowledge_base_retrieval" in rejected_output["output"]


def test_validate_final_answer_accepts_markdown_bold_label_variants():
    from rag_system.workflows.oss_standard_agent import _validate_final_answer

    plain = (
        "Explanation: found in the document [d3].\n"
        "Exact Answer: blue\n"
        "Confidence: 90%"
    )
    colon_inside_bold = (
        "**Explanation:** found in the document [d3].\n"
        "**Exact Answer:** blue\n"
        "**Confidence:** 90%"
    )
    colon_outside_bold = (
        "**Explanation**: found in the document [d3].\n"
        "**Exact Answer**: blue\n"
        "**Confidence**: 90%"
    )

    for text in (plain, colon_inside_bold, colon_outside_bold):
        validation = _validate_final_answer(text)
        assert validation["valid"] is True, (text, validation)


def test_validate_final_answer_flags_missing_citation_only():
    from rag_system.workflows.oss_standard_agent import _validate_final_answer

    validation = _validate_final_answer(
        "Explanation: I could not find the answer in the documents.\n"
        "Exact Answer: unknown\n"
        "Confidence: 10%"
    )

    assert validation["valid"] is False
    assert validation["missing_fields"] == ["document citation"]


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


def test_oss_standard_agent_retries_transient_generation_error_and_continues():
    """A one-off generation request error (e.g. a transient vLLM/Harmony
    server error such as vllm-project/vllm#23567) should not throw away an
    otherwise-healthy trajectory. Unlike the old hard-abort-on-first-error
    behavior, it's retried up to max_generation_retries times.
    """

    class FlakyResponsesClient:
        def __init__(self):
            self.calls = 0

        def complete(self, input_items, tools):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("timed out")
            return {
                "id": "resp-1",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Explanation: done\n"
                                    "Exact Answer: answer\n"
                                    "Confidence: 90%\n"
                                    "[d0]"
                                ),
                            }
                        ],
                    }
                ],
            }

    progress_events = []
    record = OssStandardAgentWorkflow(
        FlakyResponsesClient(),
        FakeSearchClient(),
        progress_callback=progress_events.append,
    ).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert [event["event"] for event in progress_events][:2] == [
        "generation_started",
        "generation_retrying",
    ]
    assert progress_events[1]["attempt"] == 1
    assert progress_events[1]["max_generation_retries"] == 2
    assert progress_events[1]["error"] == {
        "type": "TimeoutError",
        "message": "timed out",
    }


def test_oss_standard_agent_gives_up_after_exhausting_generation_retries():
    class AlwaysFailingResponsesClient:
        def __init__(self):
            self.calls = 0

        def complete(self, input_items, tools):
            self.calls += 1
            raise TimeoutError(f"timed out (call {self.calls})")

    progress_events = []
    record = OssStandardAgentWorkflow(
        AlwaysFailingResponsesClient(),
        FakeSearchClient(),
        progress_callback=progress_events.append,
    ).run(benchmark_query())

    assert record["status"] == "error"
    assert record["tool_call_counts"] == {"search": 0}
    assert record["error"] == {"type": "TimeoutError", "message": "timed out (call 3)"}
    assert record["diagnostics"]["termination_reason"] == "generation_request_error"
    retrying_attempts = [
        event["attempt"]
        for event in progress_events
        if event["event"] == "generation_retrying"
    ]
    assert retrying_attempts == [1, 2]
    assert progress_events[-1]["event"] == "generation_failed"


def test_oss_standard_agent_records_context_exhaustion_as_incomplete():
    """Even the bare initial question (one turn) can't be dropped further."""

    class AlwaysOverflowingResponsesClient:
        def complete(self, input_items, tools):
            raise VllmContextLengthError(
                "prompt is too long",
                prompt_tokens=131_848,
                max_model_len=131_072,
            )

    record = OssStandardAgentWorkflow(
        AlwaysOverflowingResponsesClient(),
        FakeSearchClient(),
    ).run(benchmark_query())

    assert record["status"] == "incomplete"
    assert record["tool_call_counts"] == {"search": 0}
    assert record["diagnostics"]["termination_reason"] == (
        "context_length_exceeded"
    )
    assert record["diagnostics"]["context_truncation_events"] == []
    assert record["error"] == {
        "type": "VllmContextLengthError",
        "message": "prompt is too long",
        "prompt_tokens": 131_848,
        "max_model_len": 131_072,
    }


def test_oss_standard_agent_recovers_from_context_overflow_by_dropping_oldest_turn():
    class RecoveringResponsesClient:
        def __init__(self):
            self.calls = 0

        def complete(self, input_items, tools):
            self.calls += 1
            if len(input_items) > 1:
                raise VllmContextLengthError(
                    "prompt is too long",
                    prompt_tokens=200_000,
                    max_model_len=131_072,
                )
            if self.calls == 1:
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
            return {
                "id": "resp-2",
                "status": "completed",
                "usage": None,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Explanation: done\n"
                                    "Exact Answer: answer\n"
                                    "Confidence: 90%\n"
                                    "[d0]"
                                ),
                            }
                        ],
                    }
                ],
            }

    record = OssStandardAgentWorkflow(
        RecoveringResponsesClient(),
        FakeSearchClient(),
    ).run(benchmark_query())

    assert record["status"] == "completed"
    assert record["diagnostics"]["termination_reason"] == "final_answer"
    assert record["tool_call_counts"] == {"search": 1}
    assert record["diagnostics"]["context_truncation_events"] == [
        {
            "turn": 2,
            "prompt_tokens": 200_000,
            "max_model_len": 131_072,
            "remaining_turns": 1,
        }
    ]
