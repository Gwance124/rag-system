"""Tests for the no-tool single-pass RAG workflow over a frozen ranking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_system.contracts import BenchmarkQuery, SearchCandidate
from rag_system.workflows.single_pass import (
    SinglePassWorkflow,
    load_trec_ranking,
)


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True
        return "".join(chr(token_id) for token_id in token_ids)


class FakeResponsesClient:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def complete(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.requests.append((input_items, tools))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


VALID_ANSWER = (
    "Explanation: the answer is stated in the first document [d1].\n"
    "Exact Answer: blue\n"
    "Confidence: 90%"
)


def _response(text: str) -> dict[str, Any]:
    return {
        "id": "resp-1",
        "status": "completed",
        "usage": {"input_tokens": 400, "output_tokens": 50},
        "output": [
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "thinking"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
    }


def _query() -> BenchmarkQuery:
    return BenchmarkQuery(
        query_id="q1",
        question="What color is the sky?",
        reference_answer="blue",
        evidence_document_ids=("d1", "d9"),
        gold_document_ids=("d1",),
    )


def _candidates() -> list[SearchCandidate]:
    return [
        SearchCandidate(document_id="d1", score=0.9, text="the sky is blue"),
        SearchCandidate(document_id="d2", score=0.8, text="y" * 600),
    ]


def test_load_trec_ranking_orders_hits_by_rank(tmp_path: Path) -> None:
    trec = tmp_path / "top1000.trec"
    trec.write_text(
        "q1 Q0 d2 2 0.8 run\n"
        "q1 Q0 d1 1 0.9 run\n"
        "q2 Q0 d7 1 0.7 run\n",
        encoding="utf-8",
    )

    ranking = load_trec_ranking(trec)

    assert ranking["q1"] == [("d1", 0.9), ("d2", 0.8)]
    assert ranking["q2"] == [("d7", 0.7)]


def test_load_trec_ranking_rejects_malformed_lines(tmp_path: Path) -> None:
    trec = tmp_path / "bad.trec"
    trec.write_text("q1 d1 0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TREC"):
        load_trec_ranking(trec)


def test_single_pass_run_builds_context_and_official_record() -> None:
    client = FakeResponsesClient([_response(VALID_ANSWER)])
    workflow = SinglePassWorkflow(
        responses_client=client,
        tokenizer=CharacterTokenizer(),
        snippet_max_tokens=512,
    )

    record = workflow.run(_query(), _candidates())

    # The request is a single no-tool prompt containing question and snippets.
    assert len(client.requests) == 1
    input_items, tools = client.requests[0]
    assert tools == []
    prompt = input_items[0]["content"]
    assert "What color is the sky?" in prompt
    assert "the sky is blue" in prompt
    assert "y" * 512 in prompt
    assert "y" * 513 not in prompt

    assert record["query_id"] == "q1"
    assert record["status"] == "completed"
    assert record["tool_call_counts"] == {"search": 0}
    assert record["retrieved_docids"] == ["d1", "d2"]
    assert record["result"][-1] == {
        "type": "output_text",
        "tool_name": None,
        "arguments": None,
        "output": VALID_ANSWER,
    }

    diagnostics = record["diagnostics"]
    assert diagnostics["workflow"] == "single_pass"
    assert diagnostics["context_document_count"] == 2
    assert diagnostics["snippet_token_counts"] == [15, 512]
    assert diagnostics["evidence"]["recall"] == pytest.approx(0.5)
    assert diagnostics["gold"]["recall"] == pytest.approx(1.0)
    assert diagnostics["final_answer_validation"]["valid"] is True
    assert diagnostics["termination_reason"] == "final_answer"
    assert diagnostics["generation_usage"] == [
        {"input_tokens": 400, "output_tokens": 50}
    ]


def test_single_pass_marks_invalid_format_but_still_completes() -> None:
    client = FakeResponsesClient([_response("The sky is blue.")])
    workflow = SinglePassWorkflow(
        responses_client=client,
        tokenizer=CharacterTokenizer(),
    )

    record = workflow.run(_query(), _candidates())

    assert record["status"] == "completed"
    assert record["diagnostics"]["final_answer_validation"]["valid"] is False


def test_single_pass_retries_transient_errors_then_records_error() -> None:
    client = FakeResponsesClient(
        [RuntimeError("boom 1"), RuntimeError("boom 2"), RuntimeError("boom 3")]
    )
    workflow = SinglePassWorkflow(
        responses_client=client,
        tokenizer=CharacterTokenizer(),
        max_generation_retries=2,
    )

    record = workflow.run(_query(), _candidates())

    assert len(client.requests) == 3
    assert record["status"] == "error"
    assert record["error"]["message"] == "boom 3"
    assert record["diagnostics"]["termination_reason"] == "generation_request_error"
    # Context documents were still exposed to the model attempt.
    assert record["retrieved_docids"] == ["d1", "d2"]


def test_select_context_document_ids_covers_top_k_of_each_query() -> None:
    from rag_system.workflows.single_pass import select_context_document_ids

    ranking = {
        "q1": [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)],
        "q2": [("d2", 0.8), ("d1", 0.7)],
    }

    needed = select_context_document_ids(ranking, ["q1", "q2"], top_k=2)

    assert needed == {"d1", "d2"}


def test_select_context_document_ids_rejects_missing_query() -> None:
    from rag_system.workflows.single_pass import select_context_document_ids

    with pytest.raises(ValueError, match="q9"):
        select_context_document_ids({"q1": [("d1", 0.9)]}, ["q9"], top_k=1)


def test_select_context_document_ids_rejects_short_ranking() -> None:
    from rag_system.workflows.single_pass import select_context_document_ids

    with pytest.raises(ValueError, match="fewer than"):
        select_context_document_ids({"q1": [("d1", 0.9)]}, ["q1"], top_k=5)


def test_build_document_lookup_streams_only_needed_documents() -> None:
    from rag_system.contracts import CorpusDocument
    from rag_system.workflows.single_pass import build_document_lookup

    corpus = (
        CorpusDocument(document_id=f"d{index}", text=f"text {index}", url="u")
        for index in range(5)
    )

    lookup = build_document_lookup(corpus, {"d1", "d3"})

    assert lookup == {"d1": "text 1", "d3": "text 3"}


def test_build_document_lookup_reports_missing_documents() -> None:
    from rag_system.contracts import CorpusDocument
    from rag_system.workflows.single_pass import build_document_lookup

    corpus = iter([CorpusDocument(document_id="d1", text="text 1", url="u")])

    with pytest.raises(ValueError, match="d9"):
        build_document_lookup(corpus, {"d1", "d9"})


def test_single_pass_requires_candidates() -> None:
    workflow = SinglePassWorkflow(
        responses_client=FakeResponsesClient([]),
        tokenizer=CharacterTokenizer(),
    )

    with pytest.raises(ValueError, match="candidate"):
        workflow.run(_query(), [])
