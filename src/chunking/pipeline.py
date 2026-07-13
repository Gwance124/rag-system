import re
import pandas as pd

_YEAR_RE = re.compile(r'^(\d{2})(\d{2})')


def _year_from_yymm_id(yymm_id: str) -> int:
    match = _YEAR_RE.match(yymm_id)
    if not match:
        raise ValueError(f"Cannot parse year from yymm_id: {yymm_id!r}")
    yy = int(match.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def filter_pilot_papers(
    df: pd.DataFrame, category: str = "cs.IR", min_year: int = 2020
) -> pd.DataFrame:
    years = df["yymm_id"].apply(_year_from_yymm_id)
    has_category = df["categories"].apply(lambda cats: category in cats.split())
    filtered = df[has_category & (years >= min_year)]
    return filtered[
        ["id", "title", "abstract", "categories", "yymm_id", "latex"]
    ].reset_index(drop=True)
