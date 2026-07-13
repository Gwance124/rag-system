import pandas as pd
from chunking.pipeline import filter_pilot_papers


def test_filters_by_category_and_year():
    df = pd.DataFrame([
        {"id": "1", "title": "A", "abstract": "a", "categories": "cs.IR cs.CL",
         "yymm_id": "2103", "latex": "..."},
        {"id": "2", "title": "B", "abstract": "b", "categories": "cs.CL",
         "yymm_id": "2103", "latex": "..."},
        {"id": "3", "title": "C", "abstract": "c", "categories": "cs.IR",
         "yymm_id": "1907", "latex": "..."},
    ])
    result = filter_pilot_papers(df)
    assert result["id"].tolist() == ["1"]


def test_output_has_only_expected_columns():
    df = pd.DataFrame([
        {"id": "1", "title": "A", "abstract": "a", "categories": "cs.IR",
         "yymm_id": "2103", "latex": "...", "authors": "someone", "doi": "10.1/x"},
    ])
    result = filter_pilot_papers(df)
    assert list(result.columns) == ["id", "title", "abstract", "categories", "yymm_id", "latex"]
