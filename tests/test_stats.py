from chunking.stats import (
    estimate_tokens_from_chars,
    summarize_tokens,
    paper_cleaned_token_count,
    build_token_stats_report,
)
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


def test_build_token_stats_report_includes_per_paper_values_and_summaries():
    report = build_token_stats_report(
        paper_ids=["p1", "p2"],
        raw_token_estimates=[100, 200],
        cleaned_token_counts=[40, 60],
        chunk_token_counts=[20, 20, 30, 30],
    )
    assert report["per_paper"] == [
        {"id": "p1", "raw_estimate": 100, "cleaned_actual": 40},
        {"id": "p2", "raw_estimate": 200, "cleaned_actual": 60},
    ]
    assert report["raw_summary"] == summarize_tokens([100, 200])
    assert report["cleaned_summary"] == summarize_tokens([40, 60])
    assert report["chunk_summary"] == summarize_tokens([20, 20, 30, 30])
