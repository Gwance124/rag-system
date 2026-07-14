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


def compute_pool_token_stats(paper_ids: list[str], texts: list[str], tokenizer) -> dict:
    """Tokenizes every chunk text in the pool (not just a sample) and returns
    both the chunk-level distribution (one value per chunk) and the
    paper-level distribution (each paper's chunks summed to one total)."""
    chunk_token_counts = [tokenizer.count_tokens(t) for t in texts]

    paper_totals: dict[str, int] = {}
    for paper_id, tokens in zip(paper_ids, chunk_token_counts):
        paper_totals[paper_id] = paper_totals.get(paper_id, 0) + tokens

    return {
        "chunk_stats": summarize_tokens(chunk_token_counts),
        "paper_stats": summarize_tokens(list(paper_totals.values())),
    }


def build_token_stats_report(
    paper_ids: list[str],
    raw_token_estimates: list[int],
    cleaned_token_counts: list[int],
    chunk_token_counts: list[int],
) -> dict:
    return {
        "per_paper": [
            {"id": paper_id, "raw_estimate": raw, "cleaned_actual": cleaned}
            for paper_id, raw, cleaned in zip(paper_ids, raw_token_estimates, cleaned_token_counts)
        ],
        "raw_summary": summarize_tokens(raw_token_estimates),
        "cleaned_summary": summarize_tokens(cleaned_token_counts),
        "chunk_summary": summarize_tokens(chunk_token_counts),
    }
