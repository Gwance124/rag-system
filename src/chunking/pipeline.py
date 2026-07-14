import json
import re
import pandas as pd
from chunking.types import ParsedPaper
from chunking.latex_parse import parse_sections
from chunking.chunker import chunk_paper

_YEAR_RE = re.compile(r'^(\d{2})(\d{2})')

# chunk_paper/chunker.py now bounds tokenizer calls at the block level
# regardless of a paper's total length, so this is only a last-resort
# circuit breaker against true dataset corruption (e.g. multiple papers'
# content accidentally concatenated into one row), not a filter on
# legitimately long papers (which can run several MB, e.g. large appendices).
MAX_LATEX_CHARS = 50_000_000


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


def parse_paper_row(row) -> ParsedPaper:
    sections = parse_sections(row["latex"])
    return ParsedPaper(id=row["id"], title=row["title"], abstract=row["abstract"], sections=sections)


def run_chunking(pilot_df: pd.DataFrame, tokenizer, max_tokens: int = 512, progress_every: int = 0):
    records = []
    failures = []
    total = len(pilot_df)
    for i, (_, row) in enumerate(pilot_df.iterrows(), 1):
        latex_text = row["latex"]
        if isinstance(latex_text, str) and len(latex_text) > MAX_LATEX_CHARS:
            failures.append({
                "id": row["id"],
                "error": f"latex field too large ({len(latex_text)} chars > {MAX_LATEX_CHARS}), skipped",
            })
        else:
            try:
                paper = parse_paper_row(row)
                records.extend(chunk_paper(paper, tokenizer, max_tokens=max_tokens))
            except Exception as exc:
                failures.append({"id": row["id"], "error": str(exc)})

        if progress_every and i % progress_every == 0:
            print(f"[{i}/{total}] papers processed, {len(records)} chunks so far, {len(failures)} failures")

    return records, failures


def write_chunks(records, output_path: str):
    df = pd.DataFrame([r.__dict__ for r in records])
    df.to_parquet(output_path, index=False)


def write_failures(failures, output_path: str):
    with open(output_path, "w") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")
