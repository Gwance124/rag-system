import statistics


def estimate_tokens_from_chars(text: str) -> int:
    """Cheap token estimate for text too large to tokenize directly."""
    return len(text) // 4


def summarize_tokens(values: list[int]) -> dict:
    return {
        "total": sum(values),
        "avg": sum(values) / len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def paper_cleaned_token_count(chunk_texts: list[str], tokenizer) -> int:
    return sum(tokenizer.count_tokens(t) for t in chunk_texts)
