"""BrowseComp-Plus Standard search loop for GPT-OSS via the Responses API."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rag_system.contracts import BenchmarkQuery
from rag_system.generation.vllm_responses import VllmContextLengthError
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

# vllm-project/vllm#32587: vLLM's gpt-oss/Harmony tool-call parser
# intermittently leaks the raw `<|channel|>commentary` special token onto
# the end of the tool name it reports (e.g. "local_knowledge_base_retrieval"
# becomes "local_knowledge_base_retrievalcommentary"), with no separator.
# Open, unowned upstream as of 2026-07-23; `skip_special_tokens` does not
# help because the token leaks during generation, not post-processing. Strip
# a known channel-name suffix before alias-checking so a well-formed call on
# an unlucky turn is recovered immediately instead of being rejected and
# costing a full wasted generation round-trip.
_HARMONY_CHANNEL_NAME_SUFFIXES = ("commentary", "analysis", "final")


def _strip_leaked_channel_suffix(name: str) -> str:
    for suffix in _HARMONY_CHANNEL_NAME_SUFFIXES:
        if name != suffix and name.endswith(suffix):
            candidate = name[: -len(suffix)]
            if candidate in _MCP_SEARCH_NAME_ALIASES:
                return candidate
    return name


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
    # Accept "Label:", "**Label:**", and "**Label**:"; the colon may sit
    # inside or outside the closing markdown bold.
    label_suffix = r"(?::\*\*|\*\*:|:)\s*(?:\*\*)?\s*"
    has_explanation = (
        re.search(label_prefix + r"Explanation" + label_suffix + r"\S", text)
        is not None
    )
    has_exact_answer = (
        re.search(label_prefix + r"Exact Answer" + label_suffix + r"\S", text)
        is not None
    )
    confidence_match = re.search(
        label_prefix
        + r"Confidence"
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
    name = _strip_leaked_channel_suffix(name)
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


def _reject_mcp_call(
    item: dict[str, Any],
    *,
    iteration: int,
    item_index: int,
    error: Exception,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a synthetic function_call/function_call_output pair so the
    model is told why its MCP call failed and can retry with the real
    search tool, instead of the call being silently dropped (what upstream
    does) or the whole run being hard-aborted (this workflow's old
    behavior).
    """

    call_id = f"call_mcp_reject_{iteration}_{item_index}"
    server_label = item.get("server_label")
    name = item.get("name")
    raw_arguments = item.get("arguments")
    if isinstance(raw_arguments, str):
        arguments = raw_arguments
    elif raw_arguments is None:
        arguments = "{}"
    else:
        arguments = json.dumps(raw_arguments, ensure_ascii=False)

    rejected_call = {
        "type": "function_call",
        "id": f"fc_mcp_reject_{iteration}_{item_index}",
        "call_id": call_id,
        "name": name if isinstance(name, str) else "unknown",
        "arguments": arguments,
        "status": "completed",
    }
    recipient = (
        f"{server_label}.{name}" if server_label or name else "the requested tool"
    )
    rejected_output = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": (
            f"Error: no such tool ({recipient}). {error} The only available "
            f"tool is '{OSS_SEARCH_TOOL_NAME}', which accepts exactly one "
            f"argument, user_query (a search string). Call "
            f"'{OSS_SEARCH_TOOL_NAME}' again with a valid user_query to continue."
        ),
    }
    return rejected_call, rejected_output


def _drop_oldest_turn(turns: list[list[dict[str, Any]]]) -> bool:
    """Drop the oldest non-initial turn in place.

    turns[0] is the original user question and is never dropped. Returns
    False once only that turn remains, so the caller can give up instead of
    looping forever.
    """

    if len(turns) <= 1:
        return False
    del turns[1]
    return True


_CONTEXT_BUDGET_STOP_MESSAGE = (
    "You are approaching the model's context limit. Stop searching now and "
    "write your final answer immediately, using only the evidence already "
    "gathered, in the required Explanation / Exact Answer / Confidence "
    "format."
)

_CONTEXT_BUDGET_REJECTED_SEARCH_OUTPUT = (
    "Error: the context budget has been reached. No further searches are "
    "permitted; provide your final answer now."
)


@dataclass(frozen=True)
class OssStandardAgentWorkflow:
    responses_client: ResponsesClient
    search_client: SearchClient
    max_iterations: int = 100
    max_search_calls: int = 100
    max_generation_retries: int = 2
    context_budget_tokens: int = 128_000
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def _progress(self, event: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **details})

    def run(self, query: BenchmarkQuery) -> dict[str, Any]:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_search_calls <= 0:
            raise ValueError("max_search_calls must be positive")

        turns: list[list[dict[str, Any]]] = [
            [
                {
                    "role": "user",
                    "content": STANDARD_AGENT_PROMPT.format(question=query.question),
                }
            ]
        ]
        results: list[dict[str, Any]] = []
        retrieved_docids: set[str] = set()
        generation_usage = []
        generation_steps = []
        search_steps = []
        context_truncation_events: list[dict[str, Any]] = []
        invalid_function_calls: list[dict[str, Any]] = []
        search_calls = 0
        status = "incomplete"
        termination_reason = "max_iterations"
        run_error: dict[str, Any] | None = None
        final_answer_validation: dict[str, Any] | None = None
        last_prompt_tokens = 0
        context_budget_triggered = False

        for iteration in range(1, self.max_iterations + 1):
            if (
                not context_budget_triggered
                and last_prompt_tokens >= self.context_budget_tokens
            ):
                context_budget_triggered = True
                turns.append(
                    [{"role": "user", "content": _CONTEXT_BUDGET_STOP_MESSAGE}]
                )
                self._progress(
                    "context_budget_final_answer_forced",
                    turn=iteration,
                    prompt_tokens=last_prompt_tokens,
                    context_budget_tokens=self.context_budget_tokens,
                )
            tools_for_turn = [] if context_budget_triggered else OSS_SEARCH_TOOLS

            self._progress(
                "generation_started",
                turn=iteration,
                completed_search_calls=search_calls,
            )

            response = None
            generation_attempt = 0
            while True:
                input_items = [item for turn in turns for item in turn]
                try:
                    response = self.responses_client.complete(
                        input_items, tools_for_turn
                    )
                    break
                except VllmContextLengthError as exc:
                    if not _drop_oldest_turn(turns):
                        run_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "prompt_tokens": exc.prompt_tokens,
                            "max_model_len": exc.max_model_len,
                        }
                        status = "incomplete"
                        termination_reason = "context_length_exceeded"
                        self._progress(
                            "generation_failed", turn=iteration, error=run_error
                        )
                        break
                    truncation_event = {
                        "turn": iteration,
                        "prompt_tokens": exc.prompt_tokens,
                        "max_model_len": exc.max_model_len,
                        "remaining_turns": len(turns),
                    }
                    context_truncation_events.append(truncation_event)
                    self._progress("context_truncated", **truncation_event)
                    continue
                except Exception as exc:
                    # vLLM/Harmony can throw transient, request-independent
                    # server errors (e.g. vllm-project/vllm#23567,
                    # "unexpected tokens remaining in message header"). The
                    # upstream OSS runner just blindly retries any
                    # exception forever; we retry the identical request a
                    # bounded number of times instead of immediately
                    # discarding an otherwise-healthy trajectory.
                    generation_attempt += 1
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    if generation_attempt <= self.max_generation_retries:
                        self._progress(
                            "generation_retrying",
                            turn=iteration,
                            attempt=generation_attempt,
                            max_generation_retries=self.max_generation_retries,
                            error=error,
                        )
                        continue
                    run_error = error
                    status = "error"
                    termination_reason = "generation_request_error"
                    self._progress("generation_failed", turn=iteration, error=run_error)
                    break

            if response is None:
                break

            output = response["output"]
            usage = response.get("usage")
            generation_usage.append(usage)
            if isinstance(usage, dict) and isinstance(
                usage.get("input_tokens"), int
            ):
                last_prompt_tokens = usage["input_tokens"]
            item_types = [
                item.get("type") for item in output if isinstance(item, dict)
            ]
            item_details = _output_item_details(output)
            normalized_output = []
            mcp_search_aliases = []
            rejected_call_ids: set[str] = set()
            rejected_outputs: list[dict[str, Any]] = []
            rejected_calls_log: list[dict[str, Any]] = []
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
                    # Neither upstream's silent drop nor a hard run-ending
                    # error: tell the model its call was invalid via a
                    # synthetic function_call/function_call_output pair so
                    # it can retry with the real search tool.
                    rejected_call, rejected_output = _reject_mcp_call(
                        item,
                        iteration=iteration,
                        item_index=item_index,
                        error=exc,
                    )
                    normalized_output.append(rejected_call)
                    rejected_call_ids.add(rejected_call["call_id"])
                    rejected_outputs.append(rejected_output)
                    rejected_calls_log.append(
                        {
                            "item_index": item_index,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    continue
                normalized_output.append(normalized_item)
                mcp_search_aliases.append(recovery)

            function_calls = [
                item
                for item in normalized_output
                if isinstance(item, dict)
                and item.get("type") == "function_call"
                and item.get("call_id") not in rejected_call_ids
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

            if rejected_calls_log:
                self._progress(
                    "mcp_call_rejected",
                    turn=iteration,
                    rejected_calls=rejected_calls_log,
                    output_item_details=item_details,
                )

            current_turn = list(normalized_output)
            if rejected_outputs:
                current_turn.extend(rejected_outputs)

            if context_budget_triggered and function_calls:
                # gpt-oss can still emit a search call even when no tools are
                # declared for this turn (it reaches for tools it has used
                # earlier in the transcript regardless of what's offered
                # now). Reject rather than execute, so crossing the budget
                # reliably stops searching instead of merely discouraging it.
                for function_call in function_calls:
                    call_id = (
                        function_call.get("call_id")
                        if isinstance(function_call, dict)
                        else None
                    )
                    if isinstance(call_id, str) and call_id:
                        current_turn.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": _CONTEXT_BUDGET_REJECTED_SEARCH_OUTPUT,
                            }
                        )
                turns.append(current_turn)
                termination_reason = "context_budget_search_rejected"
                continue

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
                    turns.append(current_turn)
                    status = "completed"
                    termination_reason = "final_answer"
                    break
                if rejected_outputs:
                    # Give the model a chance to see the error and retry
                    # with a valid search call instead of treating this
                    # turn as a dead end.
                    turns.append(current_turn)
                    termination_reason = "invalid_tool_call_retry"
                    continue
                if (
                    normalized_output
                    and isinstance(normalized_output[-1], dict)
                    and normalized_output[-1].get("type") == "reasoning"
                ):
                    # Match the upstream OSS runner: do not feed a dangling
                    # reasoning-only item back as completed conversation state.
                    current_turn.pop()
                    turns.append(current_turn)
                    termination_reason = "reasoning_only_retry"
                    continue
                turns.append(current_turn)
                # Match the upstream OSS runner exactly: it returns
                # status "completed" as soon as a turn has no
                # type=="function_call" items, regardless of whether any
                # answer text was produced. This can end a run with no
                # final answer; evaluate_run.py-style scoring already
                # treats a missing answer as a parse failure downstream.
                status = "completed"
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
                    # Match upstream oss_client.py: any tool-execution failure
                    # is fed back to the model as the tool output and the loop
                    # continues. Only a call without a routable call_id ends
                    # the run, because its error cannot reach the model.
                    feedback_call_id = (
                        function_call.get("call_id")
                        if isinstance(function_call, dict)
                        else None
                    )
                    if not isinstance(feedback_call_id, str) or not feedback_call_id:
                        run_error = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        status = "error"
                        termination_reason = "invalid_tool_call"
                        break
                    invalid_call = {
                        "turn": iteration,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    invalid_function_calls.append(invalid_call)
                    self._progress("invalid_function_call", **invalid_call)
                    function_outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": feedback_call_id,
                            "output": (
                                f"Error executing {OSS_SEARCH_TOOL_NAME}: {exc}"
                            ),
                        }
                    )
                    continue
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

            current_turn.extend(function_outputs)
            turns.append(current_turn)
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
                "max_generation_retries": self.max_generation_retries,
                "context_budget_tokens": self.context_budget_tokens,
                "context_budget_triggered": context_budget_triggered,
                "generation_usage": generation_usage,
                "generation_steps": generation_steps,
                "search_steps": search_steps,
                "context_truncation_events": context_truncation_events,
                "invalid_function_calls": invalid_function_calls,
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
        raw_name = function_call.get("name")
        name = (
            _strip_leaked_channel_suffix(raw_name)
            if isinstance(raw_name, str)
            else raw_name
        )
        if name != OSS_SEARCH_TOOL_NAME:
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
        if not isinstance(arguments, dict) or "user_query" not in arguments:
            # Match upstream oss_client.py: arguments["user_query"] tolerates
            # extra keys; only a missing user_query is an error.
            keys = sorted(arguments) if isinstance(arguments, dict) else []
            raise ValueError(
                "search function requires a user_query argument; got keys "
                f"{keys}"
            )
        search_query = arguments["user_query"]
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError("search query must be a non-empty string")
        return call_id, search_query.strip()
