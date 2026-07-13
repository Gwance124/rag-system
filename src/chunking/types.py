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
