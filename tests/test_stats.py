from chunking.stats import estimate_tokens_from_chars, summarize_tokens, paper_cleaned_token_count
from chunking.tokenizer import FakeTokenizer


def test_estimate_tokens_from_chars_divides_length_by_four():
    assert estimate_tokens_from_chars("a" * 100) == 25


def test_summarize_tokens_returns_total_avg_median_min_max():
    stats = summarize_tokens([10, 20, 30, 40])
    assert stats == {
        "total": 100,
        "avg": 25.0,
        "median": 25.0,
        "min": 10,
        "max": 40,
    }


def test_paper_cleaned_token_count_sums_tokenizer_counts_across_chunks():
    tok = FakeTokenizer()
    total = paper_cleaned_token_count(["hello world", "foo bar baz"], tok)
    assert total == 5
