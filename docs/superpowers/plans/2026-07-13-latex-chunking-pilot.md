# LaTeX Chunking Pipeline (Pilot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that parses the arxiv-latex `latex` field for a pilot corpus (cs.IR papers from 2020+, ~18,727 papers) into structure-aware ~512-token chunks with document-context prefixes, written to parquet for manual inspection.

All paths below are relative to `latex-parser/`; run the pilot commands from that directory.

**Architecture:** A small `chunking` Python package (src layout) with pure-function modules — tokenizer abstraction, markdown-tree parser, greedy chunker, and pandas-based pipeline glue — driven by thin CLI scripts for each pipeline stage (filter, chunk, inspect, spot-check). Unit tests use a fake whitespace tokenizer so the whole suite runs without network access or the real model; the real HF tokenizer is only loaded when the CLI scripts run against real data on the work laptop/server.

**Tech Stack:** Python 3.11, pandas + pyarrow (parquet I/O), transformers (`AutoTokenizer`, local-files-only), pytest.

## Global Constraints

- No new downloads of models or datasets in this session — all development and unit testing here uses synthetic fixtures and a fake tokenizer; the real tokenizer/data only get touched later on the work laptop/servers.
- The real `nv-embed-reason-3b` tokenizer must be loaded from its existing local vLLM model directory (`local_files_only=True`), never fetched from the network.
- Pilot corpus definition: `categories` contains `cs.IR` AND year (derived from `yymm_id`) >= 2020.
- Chunk target: ~512 tokens per chunk, measured on `text_with_context` (prefix + body) via the real tokenizer.
- `chunks.parquet` columns are exactly: `id, chunk_index, section_path, text_with_context, text_raw` — no `token_count` or `block_types` columns (dropped per spec review).
- Never split a code/algorithm block across a chunk boundary unless the block alone exceeds the token cap, in which case split at blank-line (code) or sentence (everything else) boundaries.
- Bibliography/references sections and residual LaTeX bloat (`\cite{}`, `\ref{}`, `\label{}`) must be stripped during parsing.
- Parse failures are logged to `parse_failures.jsonl` (`{id, error}`) and skipped, never crash the run.

---

## File Structure

```
rag-system/latex-parser/
  pyproject.toml
  src/
    chunking/
      __init__.py
      types.py           # Block, Section, ParsedPaper, ChunkRecord dataclasses
      tokenizer.py       # Tokenizer protocol, FakeTokenizer, HFTokenizer
      markdown_parse.py  # parse_sections(latex_text) -> list[Section]
      chunker.py         # chunk_paper(paper, tokenizer, max_tokens) -> list[ChunkRecord]
      pipeline.py        # filter_pilot_papers, parse_paper_row, run_chunking, write_chunks, write_failures
  scripts/
    filter_pilot.py      # Stage 0 CLI
    run_chunking.py      # Stage 1-3 CLI
    inspect_sample.py    # Stage 1 manual inspection helper
    spot_check.py        # Stage 4 manual spot-check helper
  tests/
    test_tokenizer.py
    test_markdown_parse.py
    test_chunker.py
    test_pipeline.py
```

---

### Task 1: Project scaffolding + tokenizer abstraction

**Files:**
- Create: `pyproject.toml`
- Create: `src/chunking/__init__.py`
- Create: `src/chunking/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Produces: `Tokenizer` protocol with `count_tokens(text: str) -> int`; `FakeTokenizer` (whitespace-based, used by all later tests); `HFTokenizer(model_path: str)` (wraps a real local `AutoTokenizer`, lazy-imports `transformers`).

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "rag-system-chunking"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "pyarrow>=14.0",
    "transformers>=4.40",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create the package skeleton**

```bash
mkdir -p src/chunking tests scripts
touch src/chunking/__init__.py
```

- [ ] **Step 3: Set up a virtualenv and install the package in editable mode**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: installs pandas, pyarrow, transformers, pytest without error.

- [ ] **Step 4: Write `src/chunking/tokenizer.py`**

```python
from typing import Protocol


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


class FakeTokenizer:
    """Whitespace-based token counter for tests, independent of any real model."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class HFTokenizer:
    """Wraps a real HF tokenizer loaded from a local model directory (e.g. vLLM's
    model path). Never touches the network."""

    def __init__(self, model_path: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text))
```

- [ ] **Step 5: Write the failing test**

```python
# tests/test_tokenizer.py
from chunking.tokenizer import FakeTokenizer, HFTokenizer


def test_fake_tokenizer_counts_words():
    tok = FakeTokenizer()
    assert tok.count_tokens("hello world foo") == 3


def test_hf_tokenizer_delegates_to_underlying_encode(monkeypatch):
    class StubTokenizer:
        def encode(self, text):
            return list(range(7))  # pretend 7 tokens

    def fake_from_pretrained(model_path, local_files_only=True):
        assert local_files_only is True
        return StubTokenizer()

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", fake_from_pretrained
    )

    tok = HFTokenizer("/fake/model/path")
    assert tok.count_tokens("anything") == 7
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_tokenizer.py -v`
Expected: both tests PASS (write Step 4's code first if you're following strict TDD ordering; here the interface is trivial enough that writing tokenizer.py then the test is fine, but confirm failure first if you want strict red-green: temporarily comment out the class bodies, confirm import errors, then restore).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/chunking/__init__.py src/chunking/tokenizer.py tests/test_tokenizer.py
git commit -m "Add project scaffolding and tokenizer abstraction"
```

---

### Task 2: Data types + Markdown-tree parser

**Files:**
- Create: `src/chunking/types.py`
- Create: `src/chunking/markdown_parse.py`
- Test: `tests/test_markdown_parse.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces: `Block(block_type: str, text: str)`, `Section(heading: str, level: int, path: str, blocks: list[Block])`, `ParsedPaper(id: str, title: str, abstract: str, sections: list[Section])` dataclasses; `parse_sections(latex_text: str) -> list[Section]` — used by Task 5's `parse_paper_row`.

- [ ] **Step 1: Write `src/chunking/types.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Block:
    block_type: str  # "paragraph" | "code" | "table" | "figure_caption" | "equation"
    text: str


@dataclass
class Section:
    heading: str
    level: int
    path: str
    blocks: list[Block] = field(default_factory=list)


@dataclass
class ParsedPaper:
    id: str
    title: str
    abstract: str
    sections: list[Section] = field(default_factory=list)


@dataclass
class ChunkRecord:
    id: str
    chunk_index: int
    section_path: str
    text_with_context: str
    text_raw: str
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_markdown_parse.py
from chunking.markdown_parse import parse_sections


def test_parses_headings_into_section_path():
    text = "# Title Section\n\nSome intro text.\n\n## Subsection\n\nMore text.\n"
    sections = parse_sections(text)
    assert [s.path for s in sections] == [
        "Title Section",
        "Title Section > Subsection",
    ]
    assert sections[0].blocks[0].text == "Some intro text."
    assert sections[1].blocks[0].text == "More text."


def test_drops_bibliography_section():
    text = (
        "# Method\n\nWe propose X.\n\n"
        "# References\n\n[1] Some citation.\n"
    )
    sections = parse_sections(text)
    assert [s.heading for s in sections] == ["Method"]


def test_strips_cite_ref_label_commands():
    text = "# Method\n\nWe build on prior work \\cite{smith2020} as shown in \\ref{fig:1}.\n"
    sections = parse_sections(text)
    assert "\\cite" not in sections[0].blocks[0].text
    assert "\\ref" not in sections[0].blocks[0].text


def test_code_block_kept_intact_and_classified():
    text = "# Method\n\n```python\ndef foo():\n    return 1\n```\n"
    sections = parse_sections(text)
    block = sections[0].blocks[0]
    assert block.block_type == "code"
    assert block.text == "def foo():\n    return 1"


def test_equation_block_classified():
    text = "# Method\n\n$$ x = y + z $$\n"
    sections = parse_sections(text)
    assert sections[0].blocks[0].block_type == "equation"


def test_table_block_classified():
    text = "# Results\n\n| A | B |\n| - | - |\n| 1 | 2 |\n"
    sections = parse_sections(text)
    assert sections[0].blocks[0].block_type == "table"


def test_figure_caption_classified():
    text = "# Results\n\n![diagram](fig1.png)\n"
    sections = parse_sections(text)
    assert sections[0].blocks[0].block_type == "figure_caption"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_markdown_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chunking.markdown_parse'`

- [ ] **Step 4: Write `src/chunking/markdown_parse.py`**

```python
import re
from chunking.types import Section, Block

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
_CODE_FENCE_RE = re.compile(r'^```')
_BIBLIOGRAPHY_RE = re.compile(r'^(references|bibliography)$', re.IGNORECASE)
_CITE_RE = re.compile(r'\\(?:cite|ref|label)\{[^}]*\}')
_EQUATION_RE = re.compile(r'^\$\$.*\$\$$', re.DOTALL)
_FIGURE_RE = re.compile(r'^(!\[.*\]\(.*\)|Figure\s+\d+)', re.IGNORECASE)
_TABLE_LINE_RE = re.compile(r'^\|.*\|$')


def _clean_latex_bloat(text: str) -> str:
    return _CITE_RE.sub('', text)


def _classify_block(raw: str) -> str:
    stripped = raw.strip()
    if _EQUATION_RE.match(stripped):
        return "equation"
    if _FIGURE_RE.match(stripped):
        return "figure_caption"
    lines = stripped.splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if non_empty and all(_TABLE_LINE_RE.match(line) for line in non_empty):
        return "table"
    return "paragraph"


def parse_sections(latex_text: str) -> list[Section]:
    lines = latex_text.splitlines()
    sections: list[Section] = []
    current_section: Section | None = None
    in_code_block = False
    code_lines: list[str] = []
    paragraph_lines: list[str] = []
    heading_stack: list[str] = []
    skip_section = False

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = _clean_latex_bloat("\n".join(paragraph_lines)).strip()
            if text and current_section is not None:
                current_section.blocks.append(Block(_classify_block(text), text))
            paragraph_lines = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            text = "\n".join(code_lines)
            if current_section is not None:
                current_section.blocks.append(Block("code", text))
            code_lines = []

    for line in lines:
        if _CODE_FENCE_RE.match(line.strip()):
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                flush_paragraph()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            if _BIBLIOGRAPHY_RE.match(heading_text):
                skip_section = True
                current_section = None
                continue
            skip_section = False

            heading_stack = heading_stack[: level - 1] + [heading_text]
            path = " > ".join(heading_stack)
            current_section = Section(heading=heading_text, level=level, path=path)
            sections.append(current_section)
            continue

        if skip_section or current_section is None:
            continue

        if line.strip() == "":
            flush_paragraph()
        else:
            paragraph_lines.append(line)

    flush_paragraph()
    flush_code()
    return [s for s in sections if s.blocks]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_markdown_parse.py -v`
Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/chunking/types.py src/chunking/markdown_parse.py tests/test_markdown_parse.py
git commit -m "Add data types and markdown-tree parser"
```

---

### Task 3: Chunker (greedy grouping with token cap)

**Files:**
- Create: `src/chunking/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `ParsedPaper`, `Section`, `Block`, `ChunkRecord` from `chunking.types`; `Tokenizer` protocol and `FakeTokenizer` from `chunking.tokenizer`.
- Produces: `chunk_paper(paper: ParsedPaper, tokenizer: Tokenizer, max_tokens: int = 512) -> list[ChunkRecord]` — used by Task 5's `run_chunking`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker.py
from chunking.types import ParsedPaper, Section, Block
from chunking.tokenizer import FakeTokenizer
from chunking.chunker import chunk_paper


def _paper(sections):
    return ParsedPaper(id="1234.5678", title="T", abstract="A", sections=sections)


def test_single_small_section_produces_one_chunk():
    section = Section(
        heading="Intro", level=1, path="Intro",
        blocks=[Block("paragraph", "one two three")],
    )
    paper = _paper([section])
    chunks = chunk_paper(paper, FakeTokenizer(), max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].id == "1234.5678"
    assert chunks[0].section_path == "Intro"
    assert chunks[0].text_raw == "one two three"
    assert chunks[0].text_with_context.startswith("T\nA\nIntro\n\n")


def test_splits_into_multiple_chunks_when_over_cap():
    blocks = [Block("paragraph", "aaa bbb ccc"), Block("paragraph", "ddd eee fff")]
    section = Section(heading="Intro", level=1, path="Intro", blocks=blocks)
    paper = _paper([section])
    chunks = chunk_paper(paper, FakeTokenizer(), max_tokens=7)
    assert len(chunks) == 2
    assert chunks[0].text_raw == "aaa bbb ccc"
    assert chunks[1].text_raw == "ddd eee fff"
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_never_splits_code_block_across_chunk_boundary():
    blocks = [
        Block("paragraph", "aaa bbb ccc"),
        Block("code", "def foo():\n    return 1"),
    ]
    section = Section(heading="Method", level=1, path="Method", blocks=blocks)
    paper = _paper([section])
    chunks = chunk_paper(paper, FakeTokenizer(), max_tokens=7)
    assert len(chunks) == 2
    assert chunks[1].text_raw.count("def foo") == 1


def test_oversized_single_block_is_split_at_sentence_boundaries():
    long_text = "One sentence here. Another sentence follows. A third one too."
    section = Section(
        heading="Method", level=1, path="Method",
        blocks=[Block("paragraph", long_text)],
    )
    paper = _paper([section])
    chunks = chunk_paper(paper, FakeTokenizer(), max_tokens=6)
    assert len(chunks) > 1
    rejoined = " ".join(c.text_raw for c in chunks)
    for fragment in [
        "One sentence here.",
        "Another sentence follows.",
        "A third one too.",
    ]:
        assert fragment in rejoined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chunking.chunker'`

- [ ] **Step 3: Write `src/chunking/chunker.py`**

```python
import re
from chunking.types import ParsedPaper, ChunkRecord, Block
from chunking.tokenizer import Tokenizer

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_oversized_block(block: Block, tokenizer: Tokenizer, max_tokens: int) -> list[Block]:
    if block.block_type == "code":
        pieces = block.text.split("\n\n")
        joiner = "\n\n"
    else:
        pieces = _SENTENCE_SPLIT_RE.split(block.text)
        joiner = " "

    sub_blocks: list[Block] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = tokenizer.count_tokens(piece)
        if current and current_tokens + piece_tokens > max_tokens:
            sub_blocks.append(Block(block.block_type, joiner.join(current)))
            current = []
            current_tokens = 0
        current.append(piece)
        current_tokens += piece_tokens

    if current:
        sub_blocks.append(Block(block.block_type, joiner.join(current)))

    return sub_blocks


def chunk_paper(paper: ParsedPaper, tokenizer: Tokenizer, max_tokens: int = 512) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    chunk_index = 0

    for section in paper.sections:
        prefix = f"{paper.title}\n{paper.abstract}\n{section.path}\n\n"
        prefix_tokens = tokenizer.count_tokens(prefix)

        current_blocks: list[Block] = []
        current_tokens = prefix_tokens

        def flush():
            nonlocal current_blocks, current_tokens, chunk_index
            if not current_blocks:
                return
            body = "\n\n".join(b.text for b in current_blocks)
            records.append(
                ChunkRecord(
                    id=paper.id,
                    chunk_index=chunk_index,
                    section_path=section.path,
                    text_with_context=prefix + body,
                    text_raw=body,
                )
            )
            chunk_index += 1
            current_blocks = []
            current_tokens = prefix_tokens

        for block in section.blocks:
            block_tokens = tokenizer.count_tokens(block.text)

            if prefix_tokens + block_tokens > max_tokens:
                flush()
                for sub in _split_oversized_block(block, tokenizer, max_tokens - prefix_tokens):
                    current_blocks = [sub]
                    current_tokens = prefix_tokens + tokenizer.count_tokens(sub.text)
                    flush()
                continue

            if current_tokens + block_tokens > max_tokens:
                flush()

            current_blocks.append(block)
            current_tokens += block_tokens

        flush()

    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chunker.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/chunking/chunker.py tests/test_chunker.py
git commit -m "Add greedy token-capped chunker"
```

---

### Task 4: Stage 0 — filter to pilot set

**Files:**
- Create: `src/chunking/pipeline.py`
- Create: `scripts/filter_pilot.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `filter_pilot_papers(df: pd.DataFrame, category: str = "cs.IR", min_year: int = 2020) -> pd.DataFrame` with columns `id, title, abstract, categories, yymm_id, latex`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
import pandas as pd
from chunking.pipeline import filter_pilot_papers


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chunking.pipeline'`

- [ ] **Step 3: Write `src/chunking/pipeline.py` (filter portion)**

```python
import re
import pandas as pd

_YEAR_RE = re.compile(r'^(\d{2})(\d{2})')


def _year_from_yymm_id(yymm_id: str) -> int:
    match = _YEAR_RE.match(yymm_id)
    if not match:
        raise ValueError(f"Cannot parse year from yymm_id: {yymm_id!r}")
    yy = int(match.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def filter_pilot_papers(
    df: pd.DataFrame, category: str = "cs.IR", min_year: int = 2020
) -> pd.DataFrame:
    years = df["yymm_id"].apply(_year_from_yymm_id)
    has_category = df["categories"].apply(lambda cats: category in cats.split())
    filtered = df[has_category & (years >= min_year)]
    return filtered[
        ["id", "title", "abstract", "categories", "yymm_id", "latex"]
    ].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: both tests PASS

- [ ] **Step 5: Write `scripts/filter_pilot.py`**

```python
import argparse
import glob
import pandas as pd
from chunking.pipeline import filter_pilot_papers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-glob", required=True,
        help="Glob pattern for source parquet files, e.g. 'data/part-*.parquet'",
    )
    parser.add_argument("--output", required=True, help="Output path for pilot_papers.parquet")
    parser.add_argument("--category", default="cs.IR")
    parser.add_argument("--min-year", type=int, default=2020)
    args = parser.parse_args()

    frames = [pd.read_parquet(path) for path in sorted(glob.glob(args.input_glob))]
    df = pd.concat(frames, ignore_index=True)
    pilot = filter_pilot_papers(df, category=args.category, min_year=args.min_year)
    pilot.to_parquet(args.output, index=False)
    print(f"Wrote {len(pilot)} papers to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/chunking/pipeline.py scripts/filter_pilot.py tests/test_pipeline.py
git commit -m "Add Stage 0 pilot-set filtering pipeline and CLI"
```

---

### Task 5: Stage 1-3 — parse, chunk, write output (with failure logging)

**Files:**
- Modify: `src/chunking/pipeline.py` (append parse/chunk/write functions)
- Create: `scripts/run_chunking.py`
- Test: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `parse_sections` (Task 2), `chunk_paper` (Task 3), `HFTokenizer`/`FakeTokenizer` (Task 1), `ParsedPaper`/`ChunkRecord` (Task 2).
- Produces: `parse_paper_row(row) -> ParsedPaper`, `run_chunking(pilot_df, tokenizer, max_tokens=512) -> tuple[list[ChunkRecord], list[dict]]`, `write_chunks(records, output_path)`, `write_failures(failures, output_path)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_pipeline.py
import json
from chunking.tokenizer import FakeTokenizer
from chunking.types import ChunkRecord
from chunking.pipeline import run_chunking, write_chunks, write_failures


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `run_chunking`, `write_chunks`, `write_failures` not defined

- [ ] **Step 3: Append to `src/chunking/pipeline.py`**

```python
import json
from chunking.types import ParsedPaper
from chunking.markdown_parse import parse_sections
from chunking.chunker import chunk_paper


def parse_paper_row(row) -> ParsedPaper:
    sections = parse_sections(row["latex"])
    return ParsedPaper(id=row["id"], title=row["title"], abstract=row["abstract"], sections=sections)


def run_chunking(pilot_df: pd.DataFrame, tokenizer, max_tokens: int = 512):
    records = []
    failures = []
    for _, row in pilot_df.iterrows():
        try:
            paper = parse_paper_row(row)
            records.extend(chunk_paper(paper, tokenizer, max_tokens=max_tokens))
        except Exception as exc:
            failures.append({"id": row["id"], "error": str(exc)})
    return records, failures


def write_chunks(records, output_path: str):
    df = pd.DataFrame([r.__dict__ for r in records])
    df.to_parquet(output_path, index=False)


def write_failures(failures, output_path: str):
    with open(output_path, "w") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: all 4 tests in this file PASS

- [ ] **Step 5: Write `scripts/run_chunking.py`**

```python
import argparse
import pandas as pd
from chunking.tokenizer import HFTokenizer
from chunking.pipeline import run_chunking, write_chunks, write_failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-papers", required=True, help="Path to pilot_papers.parquet")
    parser.add_argument(
        "--tokenizer-path", required=True,
        help="Local path to the nv-embed-reason-3b tokenizer/model dir (vLLM's model dir)",
    )
    parser.add_argument("--chunks-output", required=True)
    parser.add_argument("--failures-output", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    df = pd.read_parquet(args.pilot_papers)
    tokenizer = HFTokenizer(args.tokenizer_path)
    records, failures = run_chunking(df, tokenizer, max_tokens=args.max_tokens)

    write_chunks(records, args.chunks_output)
    write_failures(failures, args.failures_output)

    print(f"Wrote {len(records)} chunks to {args.chunks_output}")
    print(f"Logged {len(failures)} parse failures to {args.failures_output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/chunking/pipeline.py scripts/run_chunking.py tests/test_pipeline.py
git commit -m "Add Stage 1-3 parse/chunk/write pipeline and CLI"
```

---

### Task 6: Manual inspection helpers (Stage 1 sample eyeball, Stage 4 spot-check)

**Files:**
- Create: `scripts/inspect_sample.py`
- Create: `scripts/spot_check.py`

**Interfaces:**
- Consumes: `HFTokenizer` (Task 1); reads `pilot_papers.parquet` / `chunks.parquet` produced by Tasks 4-5.
- Produces: no importable interface — these are manual-run CLI tools, matching the spec's Testing Strategy which calls Stage 1/Stage 4 verification "a manual verification step, not an automated test."

- [ ] **Step 1: Write `scripts/inspect_sample.py`**

```python
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Print raw `latex` field samples for manual inspection (Stage 1)."
    )
    parser.add_argument("--pilot-papers", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--chars", type=int, default=3000, help="Max characters to print per sample")
    args = parser.parse_args()

    df = pd.read_parquet(args.pilot_papers)
    sample = df.sample(n=min(args.n, len(df)), random_state=0)
    for _, row in sample.iterrows():
        print("=" * 80)
        print(f"id: {row['id']}  title: {row['title']}")
        print("-" * 80)
        print(row["latex"][: args.chars])
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `scripts/spot_check.py`**

```python
import argparse
import pandas as pd
from chunking.tokenizer import HFTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Print full chunk sequences for a handful of papers (Stage 4)."
    )
    parser.add_argument("--chunks", required=True, help="Path to chunks.parquet")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--n-papers", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.chunks)
    tokenizer = HFTokenizer(args.tokenizer_path)

    sample_ids = df["id"].drop_duplicates().sample(
        n=min(args.n_papers, df["id"].nunique()), random_state=0
    )
    for paper_id in sample_ids:
        paper_chunks = df[df["id"] == paper_id].sort_values("chunk_index")
        print("=" * 80)
        print(f"paper: {paper_id}  ({len(paper_chunks)} chunks)")
        for _, chunk in paper_chunks.iterrows():
            tokens = tokenizer.count_tokens(chunk["text_with_context"])
            print("-" * 80)
            print(f"chunk {chunk['chunk_index']} | section: {chunk['section_path']} | tokens: {tokens}")
            print(chunk["text_raw"][:1000])
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite one more time to confirm nothing broke**

Run: `pytest -v`
Expected: all tests across `test_tokenizer.py`, `test_markdown_parse.py`, `test_chunker.py`, `test_pipeline.py` PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/inspect_sample.py scripts/spot_check.py
git commit -m "Add manual inspection and spot-check CLI helpers"
```

---

## After This Plan

These scripts are meant to be run on the work laptop/server against real data, in order:

```bash
python scripts/filter_pilot.py --input-glob "data/part-*.parquet" --output pilot_papers.parquet
python scripts/inspect_sample.py --pilot-papers pilot_papers.parquet --n 10
#   -> eyeball output; if the `latex` field looks meaningfully different from
#      assumed Markdown (e.g. raw \section{}/\cite{} everywhere with no # headings),
#      stop and adjust markdown_parse.py before proceeding.
python scripts/run_chunking.py --pilot-papers pilot_papers.parquet \
    --tokenizer-path /mnt/nvme2/mlee/tokenizer/models--nvidia--llama-nv-embed-reasoning-3b \
    --chunks-output chunks.parquet --failures-output parse_failures.jsonl
python scripts/spot_check.py --chunks chunks.parquet \
    --tokenizer-path /mnt/nvme2/mlee/tokenizer/models--nvidia--llama-nv-embed-reasoning-3b --n-papers 1 \
    --pilot-papers pilot_papers.parquet
```
