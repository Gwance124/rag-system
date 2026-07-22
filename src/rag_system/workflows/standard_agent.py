"""BrowseComp-Plus Standard search-only agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from rag_system.contracts import BenchmarkQuery


STANDARD_AGENT_PROMPT = """You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Perform a search on a knowledge source. Returns top-5 hits with "
                "docid, score, and snippet. Each snippet contains at most the first "
                "512 tokens of the document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class SearchClient(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class StandardAgentWorkflow:
    chat_client: ChatClient
    search_client: SearchClient
    max_search_calls: int = 20

    def run(self, query: BenchmarkQuery) -> dict[str, Any]:
        if self.max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive")
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": STANDARD_AGENT_PROMPT.format(question=query.question),
            }
        ]
        results: list[dict[str, Any]] = []
        retrieved_docids: set[str] = set()
        usage_rows = []
        search_calls = 0
        status = "incomplete"

        for _ in range(self.max_search_calls + 1):
            completion = self.chat_client.complete(messages, SEARCH_TOOLS)
            message = completion.get("message")
            if not isinstance(message, dict):
                raise ValueError("chat client returned an invalid message")
            usage_rows.append(completion.get("usage"))
            tool_calls = message.get("tool_calls") or []

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                results.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": [reasoning.strip()],
                    }
                )

            if not tool_calls:
                content = message.get("content")
                if isinstance(content, str) and content.strip() and search_calls > 0:
                    results.append(
                        {
                            "type": "output_text",
                            "tool_name": None,
                            "arguments": None,
                            "output": content.strip(),
                        }
                    )
                    status = "completed"
                break

            for tool_call in tool_calls:
                if search_calls >= self.max_search_calls:
                    break
                call_id, arguments = self._parse_search_call(tool_call)
                hits = self.search_client.search(arguments["query"])
                output = json.dumps(hits, ensure_ascii=False)
                search_calls += 1
                retrieved_docids.update(hit["docid"] for hit in hits)
                results.append(
                    {
                        "type": "tool_call",
                        "tool_name": "search",
                        "arguments": arguments,
                        "output": output,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output,
                    }
                )
            if search_calls >= self.max_search_calls:
                break

        return {
            "schema_version": "1.0",
            "query_id": query.query_id,
            "tool_call_counts": {"search": search_calls},
            "status": status,
            "retrieved_docids": sorted(retrieved_docids),
            "result": results,
            "diagnostics": {
                "max_search_calls": self.max_search_calls,
                "generation_usage": usage_rows,
            },
        }

    @staticmethod
    def _parse_search_call(tool_call: Any) -> tuple[str, dict[str, str]]:
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("id"), str):
            raise ValueError("assistant returned an invalid tool call")
        function = tool_call.get("function")
        if not isinstance(function, dict) or function.get("name") != "search":
            raise ValueError("Standard workflow only permits the search tool")
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ValueError("search tool arguments are invalid")
        if set(arguments) != {"query"} or not isinstance(arguments["query"], str):
            raise ValueError("search tool requires exactly one string query")
        if not arguments["query"].strip():
            raise ValueError("search query must not be empty")
        return tool_call["id"], {"query": arguments["query"].strip()}
