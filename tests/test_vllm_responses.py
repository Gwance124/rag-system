from __future__ import annotations

import io
import json
import urllib.error

import pytest

from rag_system.generation.vllm_responses import (
    VllmContextLengthError,
    VllmResponsesClient,
)


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
                "incomplete_details": None,
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
    assert result["incomplete_details"] is None


def test_vllm_responses_classifies_harmony_context_overflow(monkeypatch):
    body = json.dumps(
        {
            "error": {
                "message": (
                    "The engine prompt length 131848 exceeds the "
                    "max_model_len 131072. Please reduce prompt."
                ),
                "type": "invalid_request_error",
                "param": "input",
                "code": 400,
            }
        }
    ).encode()

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = VllmResponsesClient(
        "http://generator.test/v1",
        "openai/gpt-oss-20b",
    )

    with pytest.raises(VllmContextLengthError) as raised:
        client.complete([{"role": "user", "content": "question"}], [])

    assert raised.value.prompt_tokens == 131_848
    assert raised.value.max_model_len == 131_072
