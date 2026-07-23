"""Single-pass RAG over a frozen ranking: one no-tool GPT-OSS generation.

The top-k documents of a frozen retrieval ranking are truncated with the
same 512-token Standard snippet contract, rendered in the same JSON hit
shape the agent sees from the search tool, and answered in one Responses
API call with no tools. The emitted record keeps the official run shape so
the same summarizer works on agent and single-pass run directories.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_system.contracts import BenchmarkQuery, CorpusDocument, SearchCandidate
from rag_system.retrieval.standard import (
    STANDARD_SNIPPET_MAX_TOKENS,
    SnippetTokenizer,
    prefix_snippet,
)
from rag_system.workflows.oss_standard_agent import (
    ResponsesClient,
    _message_text,
    _reasoning_output,
    _validate_final_answer,
)
from rag_system.workflows.standard_agent import _ranked_label_metrics


SINGLE_PASS_PROMPT = """You are a research assistant. I will give you a question and a set of retrieved documents. You need to reason and answer the question based on these retrieved documents, step by step. The documents may be truncated and may not all be relevant.

Question: {question}

Retrieved documents:
{documents}

Your response should be in the following format:
Explanation: {{your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


def load_trec_ranking(path: str | Path) -> dict[str, list[tuple[str, float]]]:
    """Load a TREC run file into rank-ordered ``(docid, score)`` lists."""

    ranked: dict[str, list[tuple[int, str, float]]] = {}
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{line_number}: not a 6-field TREC run line"
                )
            query_id, _, document_id, rank_text, score_text, _ = fields
            try:
                rank = int(rank_text)
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: not a TREC rank/score pair"
                ) from exc
            ranked.setdefault(query_id, []).append((rank, document_id, score))

    return {
        query_id: [(document_id, score) for _, document_id, score in sorted(hits)]
        for query_id, hits in ranked.items()
    }


def select_context_document_ids(
    ranking: dict[str, list[tuple[str, float]]],
    query_ids: Sequence[str],
    top_k: int,
) -> set[str]:
    """Return the union of top-k document IDs needed for the given queries."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    needed: set[str] = set()
    for query_id in query_ids:
        hits = ranking.get(query_id)
        if hits is None:
            raise ValueError(f"frozen ranking has no hits for query {query_id}")
        if len(hits) < top_k:
            raise ValueError(
                f"frozen ranking for query {query_id} has fewer than "
                f"{top_k} hits"
            )
        needed.update(document_id for document_id, _ in hits[:top_k])
    return needed


def build_document_lookup(
    documents: Iterable[CorpusDocument],
    needed_document_ids: set[str],
) -> dict[str, str]:
    """Stream the corpus once, keeping only the needed document texts."""

    lookup: dict[str, str] = {}
    remaining = set(needed_document_ids)
    for document in documents:
        if document.document_id in remaining:
            lookup[document.document_id] = document.text
            remaining.discard(document.document_id)
            if not remaining:
                break
    if remaining:
        raise ValueError(
            "corpus is missing ranked documents: "
            + ", ".join(sorted(remaining)[:5])
        )
    return lookup


@dataclass(frozen=True)
class SinglePassWorkflow:
    responses_client: ResponsesClient
    tokenizer: SnippetTokenizer
    snippet_max_tokens: int = STANDARD_SNIPPET_MAX_TOKENS
    max_generation_retries: int = 2
    progress_callback: Callable[[dict[str, Any]], None] | None = None

    def _progress(self, event: str, **details: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback({"event": event, **details})

    def run(
        self,
        query: BenchmarkQuery,
        candidates: Sequence[SearchCandidate],
    ) -> dict[str, Any]:
        if not candidates:
            raise ValueError("at least one context candidate is required")
        document_ids = [candidate.document_id for candidate in candidates]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("context candidates contain duplicate document IDs")

        hits = []
        snippet_token_counts = []
        for candidate in candidates:
            snippet, token_count = prefix_snippet(
                self.tokenizer, candidate.text, self.snippet_max_tokens
            )
            hits.append(
                {
                    "docid": candidate.document_id,
                    "score": candidate.score,
                    "snippet": snippet,
                }
            )
            snippet_token_counts.append(token_count)

        # Match the agent's tool-result rendering: whitespace is part of the
        # context seen by the model.
        prompt = SINGLE_PASS_PROMPT.format(
            question=query.question,
            documents=json.dumps(hits, indent=2, ensure_ascii=False),
        )
        input_items = [{"role": "user", "content": prompt}]
        self._progress(
            "context_built",
            context_document_count=len(hits),
            snippet_token_counts=snippet_token_counts,
        )

        results: list[dict[str, Any]] = []
        generation_usage: list[Any] = []
        status = "incomplete"
        termination_reason = "no_final_message"
        run_error: dict[str, Any] | None = None
        final_answer_validation: dict[str, Any] | None = None

        self._progress("generation_started")
        response = None
        generation_attempt = 0
        while True:
            try:
                response = self.responses_client.complete(input_items, [])
                break
            except Exception as exc:
                generation_attempt += 1
                error = {"type": type(exc).__name__, "message": str(exc)}
                if generation_attempt <= self.max_generation_retries:
                    self._progress(
                        "generation_retrying",
                        attempt=generation_attempt,
                        max_generation_retries=self.max_generation_retries,
                        error=error,
                    )
                    continue
                run_error = error
                status = "error"
                termination_reason = "generation_request_error"
                self._progress("generation_failed", error=run_error)
                break

        if response is not None:
            usage = response.get("usage")
            generation_usage.append(usage)
            output = response["output"]
            for item in output:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    results.append(
                        {
                            "type": "reasoning",
                            "tool_name": None,
                            "arguments": None,
                            "output": _reasoning_output(item),
                        }
                    )
            final_text = "\n".join(
                text
                for text in (
                    _message_text(item)
                    for item in output
                    if isinstance(item, dict) and item.get("type") == "message"
                )
                if text
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
            self._progress(
                "generation_completed",
                response_status=response.get("status"),
                usage=usage,
                final_answer_found=bool(final_text),
            )

        evidence_metrics = _ranked_label_metrics(
            document_ids, query.evidence_document_ids
        )
        gold_metrics = _ranked_label_metrics(document_ids, query.gold_document_ids)

        record: dict[str, Any] = {
            "schema_version": "1.0",
            "query_id": query.query_id,
            "tool_call_counts": {"search": 0},
            "status": status,
            "retrieved_docids": document_ids,
            "result": results,
            "diagnostics": {
                "workflow": "single_pass",
                "context_document_count": len(hits),
                "snippet_max_tokens": self.snippet_max_tokens,
                "snippet_token_counts": snippet_token_counts,
                "max_generation_retries": self.max_generation_retries,
                "evidence": evidence_metrics,
                "gold": gold_metrics,
                "generation_usage": generation_usage,
                "termination_reason": termination_reason,
                "final_answer_validation": final_answer_validation,
            },
        }
        if run_error is not None:
            record["error"] = run_error
        return record
