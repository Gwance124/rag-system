"""BrowseComp-Plus Standard search loop for GPT-OSS via the Responses API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rag_system.contracts import BenchmarkQuery
from rag_system.workflows.standard_agent import (
    STANDARD_AGENT_PROMPT,
    _ranked_label_metrics,
)


BROWSECOMP_PLUS_OSS_REPOSITORY = "https://github.com/texttron/BrowseComp-Plus"
BROWSECOMP_PLUS_OSS_COMMIT = "046949032b0328319cc9a02663a759ec601d9402"
BROWSECOMP_PLUS_OSS_RUNNER = "search_agent/oss_client.py"

OSS_SEARCH_TOOLS = [
    {
        "type": "function",
        "name": "local_knowledge_base_retrieval",
        "description": (
            "Perform a search on a knowledge source. Returns top-5 hits with "
            "docid, score, and snippet. The snippet contains the document's "
            "contents (may be truncated based on token limits)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_query": {
                    "type": "string",
                    "description": (
                        "Query to search the local knowledge base for relevant "
                        "information"
                    ),
                }
            },
            "required": ["user_query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]

OSS_SEARCH_TOOL_NAME = "local_knowledge_base_retrieval"
_MCP_SEARCH_NAME_ALIASES = frozenset({"search", OSS_SEARCH_TOOL_NAME})


class UnsupportedMcpToolCallError(ValueError):
    """Raised when GPT-OSS requests a tool outside the Standard scaffold."""


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


def _validate_final_answer(text: str) -> dict[str, Any]:
    """Check the required BrowseComp-Plus fields without changing the response."""

    label_prefix = r"(?im)^\s*(?:\*\*)?"
    label_suffix = r"(?:\*\*)?\s*"
    has_explanation = (
        re.search(label_prefix + r"Explanation:" + label_suffix + r"\S", text)
        is not None
    )
    has_exact_answer = (
        re.search(label_prefix + r"Exact Answer:" + label_suffix + r"\S", text)
        is not None
    )
    confidence_match = re.search(
        label_prefix
        + r"Confidence:"
        + label_suffix
        + r"(?P<value>\d{1,3}(?:\.\d+)?)\s*%",
        text,
    )
    has_confidence = False
    if confidence_match is not None:
        confidence = float(confidence_match.group("value"))
        has_confidence = 0.0 <= confidence <= 100.0
    citation_count = len(re.findall(r"\[[^\[\]\n]+\]", text))

    missing_fields = []
    if not has_explanation:
        missing_fields.append("Explanation")
    if not has_exact_answer:
        missing_fields.append("Exact Answer")
    if not has_confidence:
        missing_fields.append("Confidence")
    if citation_count == 0:
        missing_fields.append("document citation")
    return {
        "valid": not missing_fields,
        "has_explanation": has_explanation,
        "has_exact_answer": has_exact_answer,
        "has_confidence": has_confidence,
        "citation_count": citation_count,
        "missing_fields": missing_fields,
    }


def _output_item_details(output: list[Any]) -> list[dict[str, Any]]:
    """Return useful protocol metadata without copying private argument values."""

    details = []
    for item in output:
        if not isinstance(item, dict):
            details.append({"type": type(item).__name__, "valid_object": False})
            continue
        detail: dict[str, Any] = {"type": item.get("type")}
        for key in ("id", "status", "name", "server_label"):
            value = item.get(key)
            if isinstance(value, str):
                detail[key] = value
        if item.get("type") == "mcp_call":
            raw_arguments = item.get("arguments")
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else raw_arguments
                )
            except json.JSONDecodeError:
                arguments = None
            detail["argument_keys"] = (
                sorted(str(key) for key in arguments)
                if isinstance(arguments, dict)
                else None
            )
            detail["has_output"] = item.get("output") is not None
            detail["has_error"] = item.get("error") is not None
        details.append(detail)
    return details


def _normalize_mcp_search_call(
    item: dict[str, Any],
    *,
    iteration: int,
    item_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map known GPT-OSS search recipients to the one local function tool.

    vLLM emits every Harmony recipient that it does not recognize as a declared
    function as ``mcp_call``.  GPT-OSS may spell the same search intent as
    ``browser.search`` or bare ``search`` on a later turn.  Those aliases still
    execute only our local Standard search service; no MCP or native browser is
    enabled.
    """

    server_label = item.get("server_label")
    name = item.get("name")
    if not isinstance(server_label, str) or not isinstance(name, str):
        raise UnsupportedMcpToolCallError(
            "assistant returned an MCP call without a server_label and name"
        )
    if name not in _MCP_SEARCH_NAME_ALIASES:
        raise UnsupportedMcpToolCallError(
            "Standard OSS workflow rejected MCP recipient "
            f"{server_label}.{name}; only local search aliases are permitted"
        )
    if item.get("output") is not None or item.get("error") is not None:
        raise UnsupportedMcpToolCallError(
            "Standard OSS workflow cannot replay a server-executed MCP call"
        )
    if item.get("status") not in (None, "completed"):
        raise ValueError("search MCP call is incomplete")

    raw_arguments = item.get("arguments")
    if isinstance(raw_arguments, str):
        arguments = json.loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise ValueError("search MCP arguments are invalid")
    argument_keys = set(arguments)
    if argument_keys == {"user_query"}:
        search_query = arguments["user_query"]
    elif "query" in arguments and argument_keys.issubset(
        {"query", "topn", "source"}
    ):
        # GPT-OSS's native browser spelling may include topn/source.  The
        # Standard scaffold deliberately ignores both and keeps fixed top-5
        # retrieval against the local service.
        search_query = arguments["query"]
    else:
        raise ValueError(
            "search MCP call requires user_query, or query with only "
            "optional topn/source"
        )
    if not isinstance(search_query, str) or not search_query.strip():
        raise ValueError("search MCP query must be a non-empty string")

    call_id = f"call_mcp_compat_{iteration}_{item_index}"
    normalized = {
        "type": "function_call",
        "id": f"fc_mcp_compat_{iteration}_{item_index}",
        "call_id": call_id,
        "name": OSS_SEARCH_TOOL_NAME,
        "arguments": json.dumps(
            {"user_query": search_query.strip()},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "status": "completed",
    }
    recovery = {
        "item_index": item_index,
        "server_label": server_label,
        "name": name,
        "normalized_name": OSS_SEARCH_TOOL_NAME,
        "call_id": call_id,
    }
    return normalized, recovery


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
        final_answer_validation: dict[str, Any] | None = None

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
            item_details = _output_item_details(output)
            normalized_output = []
            mcp_search_aliases = []
            normalization_error: Exception | None = None
            for item_index, item in enumerate(output):
                if not isinstance(item, dict) or item.get("type") != "mcp_call":
                    normalized_output.append(item)
                    continue
                try:
                    normalized_item, recovery = _normalize_mcp_search_call(
                        item,
                        iteration=iteration,
                        item_index=item_index,
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    normalization_error = exc
                    break
                normalized_output.append(normalized_item)
                mcp_search_aliases.append(recovery)

            function_calls = [
                item
                for item in normalized_output
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
            message_items = [
                item
                for item in normalized_output
                if isinstance(item, dict) and item.get("type") == "message"
            ]
            self._progress(
                "generation_completed",
                turn=iteration,
                response_status=response.get("status"),
                usage=usage,
                output_item_types=item_types,
                output_item_details=item_details,
                tool_call_count=len(function_calls),
                mcp_search_aliases=mcp_search_aliases,
            )
            generation_steps.append(
                {
                    "turn": iteration,
                    "response_id": response.get("id"),
                    "response_status": response.get("status"),
                    "output_item_types": item_types,
                    "output_item_details": item_details,
                    "mcp_search_aliases": mcp_search_aliases,
                    "usage": usage,
                }
            )

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

            if normalization_error is not None:
                run_error = {
                    "type": type(normalization_error).__name__,
                    "message": str(normalization_error),
                }
                status = "error"
                termination_reason = (
                    "unsupported_mcp_tool_call"
                    if isinstance(
                        normalization_error, UnsupportedMcpToolCallError
                    )
                    else "invalid_tool_call"
                )
                self._progress(
                    "tool_call_rejected",
                    turn=iteration,
                    error=run_error,
                    output_item_details=item_details,
                )
                break

            input_items.extend(normalized_output)

            if not function_calls:
                final_text = "\n".join(
                    text for text in (_message_text(item) for item in message_items) if text
                ).strip()
                if final_text:
                    final_answer_validation = _validate_final_answer(final_text)
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
                if (
                    normalized_output
                    and isinstance(normalized_output[-1], dict)
                    and normalized_output[-1].get("type") == "reasoning"
                ):
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
                # Match BrowseComp-Plus/search_agent/oss_client.py exactly:
                # whitespace is part of the tool result seen by the model.
                tool_output = json.dumps(hits, indent=2, ensure_ascii=False)
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
                "final_answer_validation": final_answer_validation,
            },
        }
        if run_error is not None:
            record["error"] = run_error
        return record

    @staticmethod
    def _parse_search_call(function_call: Any) -> tuple[str, str]:
        if not isinstance(function_call, dict):
            raise ValueError("assistant returned an invalid function call")
        if function_call.get("name") != OSS_SEARCH_TOOL_NAME:
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
