from __future__ import annotations

import json

from rag_system.generation.vllm_responses import VllmResponsesClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "id": "resp-1",
                "status": "completed",
                "output": [{"type": "message", "content": []}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ).encode()


def test_vllm_responses_sends_high_effort_tool_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = VllmResponsesClient(
        "http://generator.test/v1",
        "openai/gpt-oss-20b",
        max_output_tokens=10_000,
        reasoning_effort="high",
        timeout_seconds=123.0,
    )
    input_items = [{"role": "user", "content": "question"}]
    tools = [{"type": "function", "name": "search", "parameters": {}}]

    result = client.complete(input_items, tools)

    assert captured["url"] == "http://generator.test/v1/responses"
    assert captured["timeout"] == 123.0
    assert captured["payload"] == {
        "model": "openai/gpt-oss-20b",
        "input": input_items,
        "tools": tools,
        "max_output_tokens": 10_000,
        "truncation": "auto",
        "reasoning": {"effort": "high", "summary": "detailed"},
    }
    assert result["id"] == "resp-1"
    assert result["status"] == "completed"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2}
