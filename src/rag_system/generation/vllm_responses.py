"""Small vLLM Responses API client for GPT-OSS with injectable HTTP transport."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class VllmResponsesError(RuntimeError):
    """Raised when vLLM cannot return a valid Responses API object."""


class VllmContextLengthError(VllmResponsesError):
    """Raised when the rendered Harmony transcript exceeds model context."""

    def __init__(
        self,
        message: str,
        *,
        prompt_tokens: int,
        max_model_len: int,
    ) -> None:
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.max_model_len = max_model_len


_CONTEXT_LENGTH_PATTERN = re.compile(
    r"engine prompt length (?P<prompt_tokens>\d+) exceeds "
    r"the max_model_len (?P<max_model_len>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VllmResponsesClient:
    base_url: str
    model: str
    max_output_tokens: int = 10_000
    reasoning_effort: str = "high"
    reasoning_summary: str = "detailed"
    timeout_seconds: float = 2400.0

    def complete(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError("reasoning_effort must be low, medium, or high")
        payload = {
            "model": self.model,
            "input": input_items,
            "tools": tools,
            "max_output_tokens": self.max_output_tokens,
            "truncation": "auto",
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"vLLM Responses request failed with HTTP {exc.code}: {body}"
            context_match = _CONTEXT_LENGTH_PATTERN.search(body)
            if exc.code == 400 and context_match is not None:
                raise VllmContextLengthError(
                    message,
                    prompt_tokens=int(context_match.group("prompt_tokens")),
                    max_model_len=int(context_match.group("max_model_len")),
                ) from exc
            raise VllmResponsesError(message) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise VllmResponsesError(f"vLLM Responses request failed: {exc}") from exc

        output = result.get("output")
        if not isinstance(output, list):
            raise VllmResponsesError("vLLM response has no output item list")
        return {
            "id": result.get("id"),
            "output": output,
            "usage": result.get("usage"),
            "status": result.get("status"),
            "error": result.get("error"),
            "incomplete_details": result.get("incomplete_details"),
        }
