import re
from chunking.types import ParsedPaper, ChunkRecord, Block
from chunking.tokenizer import Tokenizer

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Word count is a cheap, near-lower-bound proxy for token count (BPE-style
# tokenizers rarely produce fewer tokens than whitespace-separated words).
# If a piece's word count alone already dwarfs the token budget by this much,
# treat it as pathologically oversized and hard-split it *before* ever
# calling the real tokenizer on it - this is what protects us from a single
# malformed block (e.g. no sentence/blank-line boundaries at all) turning
# into a multi-million-token tokenizer call.
_CHEAP_SIZE_MULTIPLIER = 20


def _hard_split_by_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), max_words)
    ]


def _looks_definitely_oversized(text: str, max_tokens: int) -> bool:
    return len(text.split()) > max_tokens * _CHEAP_SIZE_MULTIPLIER


def _bound_piece_to_max_tokens(piece: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Guarantees every returned piece's *real* token count is <= max_tokens
    (barring a single unsplittable word that alone exceeds it). The
    word-count heuristic above is a cheap proxy and can under-trigger for
    content where individual "words" are themselves token-heavy (e.g.
    \\pdfglyphtounicode{Aacute}{00A0}-style commands) - this verifies
    against the real tokenizer and keeps shrinking until it's actually true."""
    tokens = tokenizer.count_tokens(piece)
    if tokens <= max_tokens:
        return [piece]

    words = piece.split()
    if len(words) <= 1:
        return [piece]  # can't split a single "word" any further

    # Scale the word window by the observed token/word ratio (with a safety
    # margin) rather than guessing - this converges in a couple of passes
    # even when that ratio is wildly higher than the 1:1 prose assumption.
    words_per_piece = max(1, int(len(words) * max_tokens / tokens * 0.9))
    result = []
    for sub in _hard_split_by_words(piece, words_per_piece):
        result.extend(_bound_piece_to_max_tokens(sub, tokenizer, max_tokens))
    return result


def _split_oversized_block(block: Block, tokenizer: Tokenizer, max_tokens: int) -> list[Block]:
    if block.block_type == "code":
        pieces = block.text.split("\n\n")
        joiner = "\n\n"
    else:
        pieces = _SENTENCE_SPLIT_RE.split(block.text)
        joiner = " "

    # A piece with no natural split points (e.g. one giant sentence/line) can
    # still be pathologically large - hard-split it by words before it ever
    # reaches the tokenizer.
    expanded_pieces = []
    for piece in pieces:
        if _looks_definitely_oversized(piece, max_tokens):
            expanded_pieces.extend(_hard_split_by_words(piece, max_tokens))
        else:
            expanded_pieces.append(piece)
    pieces = expanded_pieces

    sub_blocks: list[Block] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = tokenizer.count_tokens(piece)

        if piece_tokens > max_tokens:
            if current:
                sub_blocks.append(Block(block.block_type, joiner.join(current)))
                current = []
                current_tokens = 0
            for bounded in _bound_piece_to_max_tokens(piece, tokenizer, max_tokens):
                sub_blocks.append(Block(block.block_type, bounded))
            continue

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
        # Header (title + section path) is fixed per section; the chunk number
        # is only known at flush time. It's excluded from prefix_tokens/budget
        # tracking since it's a couple tokens at most - negligible now that the
        # (often ~400-token) abstract is no longer repeated on every chunk.
        header = f"{paper.title}\n{section.path}\n"
        prefix_tokens = tokenizer.count_tokens(header)

        current_blocks: list[Block] = []
        current_tokens = prefix_tokens

        def flush():
            nonlocal current_blocks, current_tokens, chunk_index
            if not current_blocks:
                return
            body = "\n\n".join(b.text for b in current_blocks)
            prefix = f"{header}Chunk {chunk_index}\n\n"
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
            budget = max_tokens - prefix_tokens

            # Cheap pre-check: skip tokenizing a block's raw text at all if it's
            # already clearly oversized by word count, so a pathological block
            # (e.g. no paragraph/sentence boundaries at all) never reaches the
            # tokenizer as one unbounded string.
            if _looks_definitely_oversized(block.text, budget):
                flush()
                for sub in _split_oversized_block(block, tokenizer, budget):
                    current_blocks = [sub]
                    current_tokens = prefix_tokens + tokenizer.count_tokens(sub.text)
                    flush()
                continue

            block_tokens = tokenizer.count_tokens(block.text)

            if prefix_tokens + block_tokens > max_tokens:
                flush()
                for sub in _split_oversized_block(block, tokenizer, budget):
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
