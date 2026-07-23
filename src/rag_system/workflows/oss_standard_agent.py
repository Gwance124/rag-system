"""BrowseComp-Plus Standard search loop for GPT-OSS via the Responses API."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rag_system.contracts import BenchmarkQuery
from rag_system.workflows.standard_agent import (
    STANDARD_AGENT_PROMPT,
    _ranked_label_metrics,
)


OSS_SEARCH_TOOLS = [
    {
        "type": "function",
        "name": "local_knowledge_base_retrieval",
        "description": (
            "Search the local knowledge base. Returns the top 5 results with docid, "
            "score, and a snippet containing at most the first 512 document tokens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": "Query to search the local knowledge base for.",
                }
            },
            "required": ["user_query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


class ResponsesClient(Protocol):
    def complete(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class SearchClient(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...


def _text_parts(parts: Any, allowed_types: set[str]) -> list[str]:
    if not isinstance(parts, list):
        return []
    texts = []
    for part in parts:
        if not isinstance(part, dict) or part.get("type") not in allowed_types:
            continue
        value = part.get("text")
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def _reasoning_output(item: dict[str, Any]) -> list[str]:
    output = _text_parts(item.get("summary"), {"summary_text", "text"})
    output.extend(
        _text_parts(
            item.get("content"),
            {"reasoning_text", "output_text", "text"},
        )
    )
    return output


def _message_text(item: dict[str, Any]) -> str:
    return "\n".join(
        _text_parts(item.get("content"), {"output_text", "text"})
    ).strip()


@dataclass(frozen=True)
class OssStandardAgentWorkflow:
    responses_client: ResponsesClient
    search_client: SearchClient
    max_iterations: int = 100
    max_search_calls: int = 100
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def _progress(self, event: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **details})

    def run(self, query: BenchmarkQuery) -> dict[str, Any]:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive")

        input_items: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": STANDARD_AGENT_PROMPT.format(question=query.question),
            }
        ]
        results: list[dict[str, Any]] = []
        retrieved_docids: set[str] = set()
        generation_usage = []
        generation_steps = []
        search_steps = []
        search_calls = 0
        status = "incomplete"
        termination_reason = "max_iterations"
        run_error: dict[str, str] | None = None

        for iteration in range(1, self.max_iterations + 1):
            self._progress(
                "generation_started",
                turn=iteration,
                completed_search_calls=search_calls,
            )
            try:
                response = self.responses_client.complete(input_items, OSS_SEARCH_TOOLS)
            except Exception as exc:
                run_error = {"type": type(exc).__name__, "message": str(exc)}
                status = "error"
                termination_reason = "generation_request_error"
                self._progress("generation_failed", turn=iteration, error=run_error)
                break

            output = response["output"]
            usage = response.get("usage")
            generation_usage.append(usage)
            item_types = [
                item.get("type") for item in output if isinstance(item, dict)
            ]
            function_calls = [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
            message_items = [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
            ]
            self._progress(
                "generation_completed",
                turn=iteration,
                response_status=response.get("status"),
                usage=usage,
                output_item_types=item_types,
                tool_call_count=len(function_calls),
            )
            generation_steps.append(
                {
                    "turn": iteration,
                    "response_id": response.get("id"),
                    "response_status": response.get("status"),
                    "output_item_types": item_types,
                    "usage": usage,
                }
            )
            input_items.extend(output)

            for item in output:
                if not isinstance(item, dict) or item.get("type") != "reasoning":
                    continue
                results.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": _reasoning_output(item),
                    }
                )

            if not function_calls:
                final_text = "\n".join(
                    text for text in (_message_text(item) for item in message_items) if text
                ).strip()
                if final_text:
                    results.append(
                        {
                            "type": "output_text",
                            "tool_name": None,
                            "arguments": None,
                            "output": final_text,
                        }
                    )
                    status = "completed"
                    termination_reason = "final_answer"
                    break
                if output and item_types[-1] == "reasoning":
                    # Match the upstream OSS runner: do not feed a dangling
                    # reasoning-only item back as completed conversation state.
                    input_items.pop()
                    termination_reason = "reasoning_only_retry"
                    continue
                termination_reason = "empty_or_unusable_response"
                break

            function_outputs = []
            for function_call in function_calls:
                if search_calls >= self.max_search_calls:
                    termination_reason = "max_search_calls"
                    break
                try:
                    call_id, search_query = self._parse_search_call(function_call)
                except Exception as exc:
                    run_error = {"type": type(exc).__name__, "message": str(exc)}
                    status = "error"
                    termination_reason = "invalid_tool_call"
                    break
                self._progress(
                    "search_started",
                    search_call=search_calls + 1,
                    query=search_query,
                )
                try:
                    hits = self.search_client.search(search_query)
                except Exception as exc:
                    run_error = {"type": type(exc).__name__, "message": str(exc)}
                    status = "error"
                    termination_reason = "search_request_error"
                    self._progress(
                        "search_failed",
                        search_call=search_calls + 1,
                        query=search_query,
                        error=run_error,
                    )
                    break

                search_calls += 1
                ranked_docids = [hit["docid"] for hit in hits]
                new_docids = set(ranked_docids) - retrieved_docids
                new_evidence_hits = len(
                    new_docids.intersection(query.evidence_document_ids)
                )
                new_gold_hits = len(new_docids.intersection(query.gold_document_ids))
                retrieved_docids.update(ranked_docids)
                evidence_turn = _ranked_label_metrics(
                    ranked_docids, query.evidence_document_ids
                )
                gold_turn = _ranked_label_metrics(
                    ranked_docids, query.gold_document_ids
                )
                evidence_cumulative = _ranked_label_metrics(
                    retrieved_docids, query.evidence_document_ids
                )
                gold_cumulative = _ranked_label_metrics(
                    retrieved_docids, query.gold_document_ids
                )
                search_step = {
                    "search_call": search_calls,
                    "query": search_query,
                    "returned_documents": len(hits),
                    "unique_documents_cumulative": len(retrieved_docids),
                    "evidence": {
                        "turn_recall_at_5": evidence_turn["recall"],
                        "turn_ndcg_at_5": evidence_turn["ndcg"],
                        "turn_hits": evidence_turn["hits"],
                        "new_hits": new_evidence_hits,
                        "relevant_documents": evidence_turn["relevant"],
                        "cumulative_recall": evidence_cumulative["recall"],
                        "cumulative_hits": evidence_cumulative["hits"],
                    },
                    "gold": {
                        "turn_recall_at_5": gold_turn["recall"],
                        "turn_ndcg_at_5": gold_turn["ndcg"],
                        "turn_hits": gold_turn["hits"],
                        "new_hits": new_gold_hits,
                        "relevant_documents": gold_turn["relevant"],
                        "cumulative_recall": gold_cumulative["recall"],
                        "cumulative_hits": gold_cumulative["hits"],
                    },
                }
                search_steps.append(search_step)
                self._progress("search_completed", **search_step)
                tool_output = json.dumps(hits, ensure_ascii=False)
                results.append(
                    {
                        "type": "tool_call",
                        "tool_name": "search",
                        "arguments": {"query": search_query},
                        "output": tool_output,
                    }
                )
                function_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_output,
                    }
                )

            input_items.extend(function_outputs)
            if run_error is not None or search_calls >= self.max_search_calls:
                break

        record = {
            "schema_version": "1.0",
            "query_id": query.query_id,
            "tool_call_counts": {"search": search_calls},
            "status": status,
            "retrieved_docids": sorted(retrieved_docids),
            "result": results,
            "diagnostics": {
                "max_iterations": self.max_iterations,
                "max_search_calls": self.max_search_calls,
                "generation_usage": generation_usage,
                "generation_steps": generation_steps,
                "search_steps": search_steps,
                "termination_reason": termination_reason,
            },
        }
        if run_error is not None:
            record["error"] = run_error
        return record

    @staticmethod
    def _parse_search_call(function_call: Any) -> tuple[str, str]:
        if not isinstance(function_call, dict):
            raise ValueError("assistant returned an invalid function call")
        if function_call.get("name") != "local_knowledge_base_retrieval":
            raise ValueError("Standard OSS workflow only permits the search tool")
        call_id = function_call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("search function call has no call_id")
        raw_arguments = function_call.get("arguments")
        if isinstance(raw_arguments, str):
            arguments = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            raise ValueError("search function arguments are invalid")
        if set(arguments) != {"user_query"}:
            raise ValueError("search function requires exactly user_query")
        search_query = arguments["user_query"]
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError("search query must be a non-empty string")
        return call_id, search_query.strip()
