from __future__ import annotations

import json

from rag_system.contracts import BenchmarkQuery
from rag_system.workflows.oss_standard_agent import OssStandardAgentWorkflow


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
    search_step = record["diagnostics"]["search_steps"][0]
    assert search_step["evidence"]["turn_recall_at_5"] == 1.0
    assert search_step["evidence"]["turn_ndcg_at_5"] == 1.0
    assert search_step["evidence"]["cumulative_recall"] == 1.0
    assert search.queries == ["focused search"]
    second_input = responses.requests[1][0]
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call-1"
    assert [event["event"] for event in progress_events] == [
        "generation_started",
        "generation_completed",
        "search_started",
        "search_completed",
        "generation_started",
        "generation_completed",
    ]


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
