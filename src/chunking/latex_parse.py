import re
from chunking.types import Section, Block

# The `latex` field bundles raw source files (.tex/.bib/.sty/.cls/...)
# concatenated together, each preceded by a delimiter like:
#   ================================================
#   FILE: acmart.cls
#   ================================================
# This is NOT converted Markdown - it's the original LaTeX/BibTeX source.
_FILE_HEADER_RE = re.compile(r'={10,}\s*FILE:\s*(?P<filename>\S+)\s*={10,}\s*')

_BEGIN_DOCUMENT_RE = re.compile(r'\\begin\{document\}')
_THEBIBLIOGRAPHY_RE = re.compile(r'\\begin\{thebibliography\}.*?(\\end\{thebibliography\}|$)', re.DOTALL)
_BIBLIOGRAPHY_CMD_RE = re.compile(r'\\bibliography\{[^}]*\}.*', re.DOTALL)
_END_DOCUMENT_RE = re.compile(r'\\end\{document\}.*', re.DOTALL)
_COMMENT_RE = re.compile(r'(?<!\\)%.*')
_INCLUDE_RE = re.compile(r'\\(?:input|include)\{(?P<name>[^}]*)\}')

_SECTION_RE = re.compile(r'\\(?P<subs>(?:sub){0,2})section\*?\{(?P<title>[^}]*)\}')

_CITE_RE = re.compile(r'\\(?:[Cc]ite\w*|ref|eqref|autoref|label)\{[^}]*\}')
_BIBTEX_ENTRY_RE = re.compile(r'^@[A-Za-z]+\s*\{')
_DISPLAY_MATH_RE = re.compile(r'^(\$\$.*\$\$|\\\[.*\\\])$', re.DOTALL)

_ENV_RE = re.compile(
    r'\\begin\{(?P<env>equation\*?|align\*?|algorithm\*?|algorithmic\*?'
    r'|lstlisting|verbatim|table\*?|tabular\*?|figure\*?)\}'
    r'(?P<body>.*?)'
    r'\\end\{(?P=env)\}',
    re.DOTALL,
)
_ENV_TYPE_MAP = {
    "equation": "equation", "equation*": "equation",
    "align": "equation", "align*": "equation",
    "algorithm": "code", "algorithm*": "code",
    "algorithmic": "code", "algorithmic*": "code",
    "lstlisting": "code", "verbatim": "code",
    "table": "table", "table*": "table",
    "tabular": "table", "tabular*": "table",
    "figure": "figure_caption", "figure*": "figure_caption",
}


def _split_into_files(latex_text: str) -> list[tuple[str, str]]:
    matches = list(_FILE_HEADER_RE.finditer(latex_text))
    if not matches:
        return [("", latex_text)]
    files = []
    for i, m in enumerate(matches):
        filename = m.group("filename")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(latex_text)
        files.append((filename, latex_text[start:end]))
    return files


def _strip_preamble(text: str) -> str:
    match = _BEGIN_DOCUMENT_RE.search(text)
    return text[match.end():] if match else text


def _strip_bibliography(text: str) -> str:
    text = _THEBIBLIOGRAPHY_RE.sub('', text)
    text = _BIBLIOGRAPHY_CMD_RE.sub('', text)
    text = _END_DOCUMENT_RE.sub('', text)
    return text


def _find_matching_file(name: str, tex_files: dict[str, str]) -> str | None:
    name = name.strip()
    basename = name.rsplit("/", 1)[-1]
    candidates = [name, name + ".tex", basename, basename + ".tex"]
    for candidate in candidates:
        for key in tex_files:
            if key.lower() == candidate.lower():
                return key
    return None


def _resolve_includes(text: str, tex_files: dict[str, str], active: set[str], used: set[str]) -> str:
    """Splices \\input{X}/\\include{X} with X's content (recursively), in
    place, so section order follows the paper's actual logical structure
    instead of the order files happen to appear in the raw bundle.
    Comments must already be stripped by this point, or a commented-out
    \\input would get wrongly resolved. `active` guards against include
    cycles (A includes B includes A); `used` accumulates every file that
    got spliced in, so callers can tell which files were never referenced."""

    def replace(m: re.Match) -> str:
        key = _find_matching_file(m.group("name"), tex_files)
        if key is None or key in active:
            # Unresolvable or a cycle - drop the command rather than leaving
            # useless raw "\input{...}" text as chunk content.
            return ""
        active.add(key)
        used.add(key)
        resolved = _resolve_includes(tex_files[key], tex_files, active, used)
        active.discard(key)
        return resolved

    return _INCLUDE_RE.sub(replace, text)


def _extract_tex_content(latex_text: str) -> str:
    files = _split_into_files(latex_text)
    if len(files) == 1 and files[0][0] == "":
        # No FILE: bundling at all - the whole text is the single document.
        # No other files exist to resolve \input/\include against, so any
        # such command is unresolvable and gets dropped.
        content = _COMMENT_RE.sub('', files[0][1])
        content = _resolve_includes(content, {}, set(), set())
        content = _strip_preamble(content)
        return _strip_bibliography(content)

    tex_files = {
        name: _COMMENT_RE.sub('', content)
        for name, content in files
        if name.lower().endswith(".tex")
    }
    main_filenames = [name for name, content in tex_files.items() if _BEGIN_DOCUMENT_RE.search(content)]

    used: set[str] = set()
    parts = []
    for name in main_filenames or list(tex_files):  # fallback: no \begin{document} anywhere
        used.add(name)
        resolved = _resolve_includes(tex_files[name], tex_files, {name}, used)
        content = _strip_preamble(resolved)
        content = _strip_bibliography(content)
        parts.append(content)

    # Files never pulled in by any \input/\include (e.g. a stray appendix) -
    # append them too, so their content isn't silently lost.
    for name, content in tex_files.items():
        if name in used:
            continue
        content = _strip_preamble(content)
        content = _strip_bibliography(content)
        parts.append(content)

    return "\n\n".join(parts)


def _clean_latex_bloat(text: str) -> str:
    return _CITE_RE.sub('', text)


def _looks_like_bibtex(text: str) -> bool:
    return bool(_BIBTEX_ENTRY_RE.match(text.strip()))


def _classify_paragraph(text: str) -> str:
    if _DISPLAY_MATH_RE.match(text.strip()):
        return "equation"
    return "paragraph"


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _extract_blocks_from_body(body: str) -> list[Block]:
    blocks: list[Block] = []
    pos = 0
    for m in _ENV_RE.finditer(body):
        for para in _split_paragraphs(_clean_latex_bloat(body[pos:m.start()])):
            if not _looks_like_bibtex(para):
                blocks.append(Block(_classify_paragraph(para), para))

        env_type = _ENV_TYPE_MAP.get(m.group("env"), "paragraph")
        env_body = m.group("body").strip()
        if env_body:
            blocks.append(Block(env_type, env_body))
        pos = m.end()

    for para in _split_paragraphs(_clean_latex_bloat(body[pos:])):
        if not _looks_like_bibtex(para):
            blocks.append(Block(_classify_paragraph(para), para))

    return blocks


def parse_sections(latex_text: str) -> list[Section]:
    tex_content = _extract_tex_content(latex_text)
    tex_content = _COMMENT_RE.sub('', tex_content)

    sections: list[Section] = []
    heading_stack: list[str] = []
    matches = list(_SECTION_RE.finditer(tex_content))

    for i, m in enumerate(matches):
        title = m.group("title").strip()
        level = m.group("subs").count("sub") + 1
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tex_content)
        body = tex_content[start:end]

        heading_stack = heading_stack[: level - 1] + [title]
        path = " > ".join(heading_stack)
        section = Section(heading=title, level=level, path=path, blocks=_extract_blocks_from_body(body))
        if section.blocks:
            sections.append(section)

    return sections
