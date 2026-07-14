import pandas as pd
from chunking.latex_parse import _split_into_files, _BEGIN_DOCUMENT_RE

pilot_df = pd.read_parquet("pilot_papers.parquet").set_index("id")
chunks_df = pd.read_parquet("chunks.parquet")

# --- worst paper: check for multiple \begin{document} files in its bundle ---
print("=" * 80)
print("worst paper bundle structure: 2008.05180")
latex = pilot_df.loc["2008.05180", "latex"]
for name, content in _split_into_files(latex):
    has_doc = bool(_BEGIN_DOCUMENT_RE.search(content))
    print(f"{name!r:40} begin{{document}}={has_doc}  len={len(content)}")

# --- worst chunk: dump its actual raw content ---
print("=" * 80)
print("worst chunk content: 2305.12058 chunk 81")
row = chunks_df[(chunks_df["id"] == "2305.12058") & (chunks_df["chunk_index"] == 81)].iloc[0]
print("section_path:", row["section_path"])
print(row["text_raw"][:2000])
print("... [total len:", len(row["text_raw"]), "chars]")
