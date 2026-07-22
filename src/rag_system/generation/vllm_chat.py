"""Small OpenAI-compatible vLLM chat client with injectable HTTP transport."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class VllmChatError(RuntimeError):
    """Raised when vLLM cannot return a valid assistant message."""


@dataclass(frozen=True)
class VllmChatClient:
    base_url: str
    model: str
    max_output_tokens: int = 10_000
    temperature: float = 0.7
    top_p: float = 0.8
    seed: int = 0
    timeout_seconds: float = 900.0

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            raise VllmChatError(f"vLLM chat request failed: {exc}") from exc
        try:
            choice = result["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VllmChatError("vLLM response has no assistant message") from exc
        if not isinstance(message, dict):
            raise VllmChatError("vLLM assistant message is invalid")
        return {
            "message": message,
            "usage": result.get("usage"),
            "finish_reason": choice.get("finish_reason"),
        }
