from chunking.latex_parse import parse_sections


def test_parses_sections_and_subsections_into_paths():
    text = (
        "\\begin{document}\n"
        "\\section{Introduction}\n"
        "Some intro text.\n\n"
        "\\subsection{Motivation}\n"
        "More text.\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert [s.path for s in sections] == [
        "Introduction",
        "Introduction > Motivation",
    ]
    assert sections[0].blocks[0].text == "Some intro text."
    assert sections[1].blocks[0].text == "More text."


def test_drops_content_bundled_from_non_tex_files():
    text = (
        "================================================\n"
        "FILE: acmart.cls\n"
        "================================================\n"
        "\\ProvidesClass{acmart}\n"
        "\\RequirePackage{xkeyval}\n"
        "================================================\n"
        "FILE: main.tex\n"
        "================================================\n"
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We propose X.\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert [s.heading for s in sections] == ["Method"]
    assert sections[0].blocks[0].text == "We propose X."


def test_drops_preamble_before_begin_document():
    text = (
        "\\documentclass{article}\n"
        "\\usepackage{amsmath}\n"
        "\\newcommand{\\foo}{bar}\n"
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We propose X.\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert [s.heading for s in sections] == ["Method"]
    assert "newcommand" not in sections[0].blocks[0].text


def test_drops_thebibliography_environment():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We propose X.\n"
        "\\section{References}\n"
        "\\begin{thebibliography}{99}\n"
        "\\bibitem{smith2020} J. Smith, Some Paper, 2020.\n"
        "\\end{thebibliography}\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert [s.heading for s in sections] == ["Method"]


def test_drops_content_after_bibliography_command():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We propose X.\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{refs}\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert [s.heading for s in sections] == ["Method"]
    assert "refs" not in sections[0].blocks[0].text


def test_strips_cite_ref_label_commands():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We build on prior work \\cite{smith2020} as shown in \\ref{fig:1}.\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert "\\cite" not in sections[0].blocks[0].text
    assert "\\ref" not in sections[0].blocks[0].text


def test_strips_line_comments():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We propose X. % this is a private note\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert "private note" not in sections[0].blocks[0].text
    assert "We propose X." in sections[0].blocks[0].text


def test_equation_environment_classified():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "\\begin{equation}\n"
        "x = y + z\n"
        "\\end{equation}\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert sections[0].blocks[0].block_type == "equation"
    assert sections[0].blocks[0].text == "x = y + z"


def test_algorithm_environment_classified_as_code():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "\\begin{algorithm}\n"
        "for i in range(n): pass\n"
        "\\end{algorithm}\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert sections[0].blocks[0].block_type == "code"


def test_table_environment_classified():
    text = (
        "\\begin{document}\n"
        "\\section{Results}\n"
        "\\begin{table}\n"
        "\\begin{tabular}{ll}\n"
        "A & B \\\\\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    assert sections[0].blocks[0].block_type == "table"


def test_drops_raw_bibtex_entries_if_present_inline():
    text = (
        "\\begin{document}\n"
        "\\section{Method}\n"
        "We propose X.\n\n"
        "@article{jones2019bar,\n"
        "  author = {Jones, A.},\n"
        "  title = {Another Paper}\n"
        "}\n"
        "\\end{document}\n"
    )
    sections = parse_sections(text)
    texts = [b.text for b in sections[0].blocks]
    assert texts == ["We propose X."]
