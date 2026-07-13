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
