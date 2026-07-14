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
    assert chunks[0].text_with_context == "T\nIntro\nChunk 0\n\none two three"


def test_context_prefix_excludes_abstract():
    section = Section(
        heading="Intro", level=1, path="Intro",
        blocks=[Block("paragraph", "one two three")],
    )
    paper = ParsedPaper(id="1234.5678", title="T", abstract="ABSTRACT_CONTENT", sections=[section])
    chunks = chunk_paper(paper, FakeTokenizer(), max_tokens=100)
    assert "ABSTRACT_CONTENT" not in chunks[0].text_with_context


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


def test_pathological_block_with_no_boundaries_is_hard_split_by_words():
    # No periods, no blank lines - sentence/blank-line splitting finds nothing,
    # so this must fall back to a bounded word-window split rather than ever
    # tokenizing the whole 2000-word blob as one piece.
    long_text = " ".join(f"word{i}" for i in range(2000))
    section = Section(
        heading="Method", level=1, path="Method",
        blocks=[Block("paragraph", long_text)],
    )
    paper = _paper([section])
    chunks = chunk_paper(paper, FakeTokenizer(), max_tokens=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text_raw.split()) <= 50
    rejoined_words = " ".join(c.text_raw for c in chunks).split()
    assert rejoined_words == long_text.split()
