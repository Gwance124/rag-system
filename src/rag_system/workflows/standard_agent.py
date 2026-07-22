"""BrowseComp-Plus Standard search-only agent loop."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
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


def _ranked_label_metrics(
    ranked_document_ids: Iterable[str],
    relevant_document_ids: Iterable[str],
) -> dict[str, Any]:
    ranked = list(ranked_document_ids)
    relevant = set(relevant_document_ids)
    if not relevant:
        return {"relevant": 0, "hits": 0, "recall": None, "ndcg": None}
    relevance = [1 if document_id in relevant else 0 for document_id in ranked]
    hits = sum(relevance)
    dcg = sum(value / math.log2(rank + 2) for rank, value in enumerate(relevance))
    ideal_hits = min(len(relevant), len(ranked))
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return {
        "relevant": len(relevant),
        "hits": hits,
        "recall": hits / len(relevant),
        "ndcg": dcg / idcg,
    }


@dataclass(frozen=True)
class StandardAgentWorkflow:
    chat_client: ChatClient
    search_client: SearchClient
    max_search_calls: int = 20
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def _progress(self, event: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **details})

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
        generation_steps = []
        search_steps = []
        search_calls = 0
        status = "incomplete"
        termination_reason = "unknown"
        run_error: dict[str, str] | None = None

        for step_index in range(self.max_search_calls + 1):
            self._progress(
                "generation_started",
                turn=step_index + 1,
                completed_search_calls=search_calls,
            )
            try:
                completion = self.chat_client.complete(messages, SEARCH_TOOLS)
            except Exception as exc:
                status = "error"
                termination_reason = "generation_request_error"
                run_error = {"type": type(exc).__name__, "message": str(exc)}
                self._progress(
                    "generation_failed",
                    turn=step_index + 1,
                    error=run_error,
                )
                break
            message = completion.get("message")
            if not isinstance(message, dict):
                raise ValueError("chat client returned an invalid message")
            usage_rows.append(completion.get("usage"))
            generation_steps.append(
                {
                    "step": step_index,
                    "finish_reason": completion.get("finish_reason"),
                    "usage": completion.get("usage"),
                }
            )
            tool_calls = message.get("tool_calls") or []
            self._progress(
                "generation_completed",
                turn=step_index + 1,
                finish_reason=completion.get("finish_reason"),
                usage=completion.get("usage"),
                tool_call_count=len(tool_calls),
            )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            reasoning = message.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                # Qwen3.6's interleaved-thinking template expects historical
                # reasoning under this canonical field during tool loops.
                assistant_message["reasoning_content"] = reasoning
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

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
                    termination_reason = "final_answer"
                elif completion.get("finish_reason") == "length":
                    termination_reason = "max_output_tokens"
                else:
                    termination_reason = "empty_or_unusable_assistant_message"
                break

            for tool_call in tool_calls:
                if search_calls >= self.max_search_calls:
                    break
                call_id, arguments = self._parse_search_call(tool_call)
                self._progress(
                    "search_started",
                    search_call=search_calls + 1,
                    query=arguments["query"],
                )
                try:
                    hits = self.search_client.search(arguments["query"])
                except Exception as exc:
                    status = "error"
                    termination_reason = "search_request_error"
                    run_error = {"type": type(exc).__name__, "message": str(exc)}
                    self._progress(
                        "search_failed",
                        search_call=search_calls + 1,
                        query=arguments["query"],
                        error=run_error,
                    )
                    break
                output = json.dumps(hits, ensure_ascii=False)
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
                gold_turn = _ranked_label_metrics(ranked_docids, query.gold_document_ids)
                evidence_cumulative = _ranked_label_metrics(
                    retrieved_docids, query.evidence_document_ids
                )
                gold_cumulative = _ranked_label_metrics(
                    retrieved_docids, query.gold_document_ids
                )
                search_step = {
                    "search_call": search_calls,
                    "query": arguments["query"],
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
            if run_error is not None:
                break
            if search_calls >= self.max_search_calls:
                termination_reason = "max_search_calls"
                break

        record = {
            "schema_version": "1.0",
            "query_id": query.query_id,
            "tool_call_counts": {"search": search_calls},
            "status": status,
            "retrieved_docids": sorted(retrieved_docids),
            "result": results,
            "diagnostics": {
                "max_search_calls": self.max_search_calls,
                "generation_usage": usage_rows,
                "generation_steps": generation_steps,
                "search_steps": search_steps,
                "termination_reason": termination_reason,
            },
        }
        if run_error is not None:
            record["error"] = run_error
        return record

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
