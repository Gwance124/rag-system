import json
import pandas as pd
from chunking.tokenizer import FakeTokenizer
from chunking.types import ChunkRecord
from chunking.pipeline import filter_pilot_papers, run_chunking, write_chunks, write_failures


def test_filters_by_category_and_year():
    df = pd.DataFrame([
        {"id": "1", "title": "A", "abstract": "a", "categories": "cs.IR cs.CL",
         "yymm_id": "2103", "latex": "..."},
        {"id": "2", "title": "B", "abstract": "b", "categories": "cs.CL",
         "yymm_id": "2103", "latex": "..."},
        {"id": "3", "title": "C", "abstract": "c", "categories": "cs.IR",
         "yymm_id": "1907", "latex": "..."},
    ])
    result = filter_pilot_papers(df)
    assert result["id"].tolist() == ["1"]


def test_output_has_only_expected_columns():
    df = pd.DataFrame([
        {"id": "1", "title": "A", "abstract": "a", "categories": "cs.IR",
         "yymm_id": "2103", "latex": "...", "authors": "someone", "doi": "10.1/x"},
    ])
    result = filter_pilot_papers(df)
    assert list(result.columns) == ["id", "title", "abstract", "categories", "yymm_id", "latex"]


def test_run_chunking_produces_records_and_skips_failures():
    df = pd.DataFrame([
        {"id": "good.1", "title": "T", "abstract": "A",
         "latex": "# Intro\n\nSome text here.\n"},
        {"id": "bad.1", "title": "T2", "abstract": "A2", "latex": None},
    ])
    records, failures = run_chunking(df, FakeTokenizer(), max_tokens=100)
    assert len(records) == 1
    assert records[0].id == "good.1"
    assert len(failures) == 1
    assert failures[0]["id"] == "bad.1"


def test_run_chunking_skips_only_truly_pathological_latex_field():
    # ~55M chars - well past MAX_LATEX_CHARS, representative of corrupted/
    # concatenated dataset rows rather than a real single paper.
    huge_latex = "# Intro\n\n" + ("word " * 11_000_000)
    df = pd.DataFrame([
        {"id": "huge.1", "title": "T", "abstract": "A", "latex": huge_latex},
    ])
    records, failures = run_chunking(df, FakeTokenizer(), max_tokens=100)
    assert records == []
    assert len(failures) == 1
    assert failures[0]["id"] == "huge.1"
    assert "too large" in failures[0]["error"]


def test_run_chunking_does_not_skip_a_legitimately_long_paper():
    # A real long paper (e.g. one with a large appendix) can run several MB -
    # well above the old 2M-char threshold but nowhere near true corruption.
    # It should be chunked normally, not rejected for its size alone.
    section_body = "One sentence here. " * 20_000  # ~400k chars of real prose
    long_latex = f"# Intro\n\n{section_body}\n\n## Appendix\n\n{section_body}\n"
    df = pd.DataFrame([
        {"id": "long.1", "title": "T", "abstract": "A", "latex": long_latex},
    ])
    records, failures = run_chunking(df, FakeTokenizer(), max_tokens=100)
    assert failures == []
    assert len(records) > 1
    assert all(r.id == "long.1" for r in records)


def test_write_chunks_and_failures_roundtrip(tmp_path):
    records = [
        ChunkRecord(
            id="1", chunk_index=0, section_path="Intro",
            text_with_context="ctx", text_raw="raw",
        )
    ]
    chunks_path = tmp_path / "chunks.parquet"
    write_chunks(records, str(chunks_path))
    df = pd.read_parquet(chunks_path)
    assert df.iloc[0]["id"] == "1"

    failures_path = tmp_path / "failures.jsonl"
    write_failures([{"id": "2", "error": "boom"}], str(failures_path))
    with open(failures_path) as f:
        line = json.loads(f.readline())
    assert line == {"id": "2", "error": "boom"}
