from __future__ import annotations

import json

from rag_system.generation.vllm_chat import VllmChatClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": "answer"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ).encode()


def test_vllm_chat_sends_auto_search_tool_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = VllmChatClient("http://generator.test/v1", "qwen3.6-27b", seed=3)
    result = client.complete([{"role": "user", "content": "q"}], [{"tool": "x"}])

    assert captured["url"] == "http://generator.test/v1/chat/completions"
    assert captured["payload"]["tool_choice"] == "auto"
    assert captured["payload"]["parallel_tool_calls"] is False
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert captured["payload"]["seed"] == 3
    assert result["message"]["content"] == "answer"
