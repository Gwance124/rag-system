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


def test_drops_bibtex_entries_regardless_of_heading_name():
    text = (
        "# refs.bib\n\n"
        "@article{smith2020foo,\n"
        "  author = {Smith, J.},\n"
        "  title = {Some Paper},\n"
        "  year = {2020}\n"
        "}\n\n"
        "# Method\n\nWe propose X.\n"
    )
    sections = parse_sections(text)
    # the whole "refs.bib" section had nothing but a bibtex entry, so once
    # that block is filtered out the section itself has no blocks left
    assert [s.heading for s in sections] == ["Method"]


def test_strips_bibtex_entries_within_a_section_but_keeps_real_content():
    text = (
        "# Related Work\n\n"
        "We build on prior work.\n\n"
        "@article{jones2019bar,\n"
        "  author = {Jones, A.},\n"
        "  title = {Another Paper}\n"
        "}\n\n"
        "This concludes the section.\n"
    )
    sections = parse_sections(text)
    assert len(sections) == 1
    texts = [b.text for b in sections[0].blocks]
    assert texts == ["We build on prior work.", "This concludes the section."]


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
