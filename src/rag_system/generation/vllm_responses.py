"""Small vLLM Responses API client for GPT-OSS with injectable HTTP transport."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class VllmResponsesError(RuntimeError):
    """Raised when vLLM cannot return a valid Responses API object."""


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
            raise VllmResponsesError(
                f"vLLM Responses request failed with HTTP {exc.code}: {body}"
            ) from exc
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
        }
