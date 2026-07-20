import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from retrieval.benchmarks import (
    DEFAULT_MTEB_DATASET,
    DEFAULT_QASPER_DATASET,
    DEFAULT_QASPER_RAW_DATASET,
    Benchmark,
    load_bright_hf,
    load_jsonl_benchmark,
    load_litsearch_hf,
    load_mteb_hf,
    load_qasper_hf,
    load_qasper_paper_benchmark_hf,
    load_qasper_paper_documents_hf,
    load_scholargym_benchmark,
    mteb_dataset_id,
    qasper_chunk_candidates,
    qasper_raw_dataset_path,
    scholargym_paths,
)
from retrieval.dense import QdrantIndex
from retrieval.fusion import aggregate_to_papers, rrf_fuse
from retrieval.metrics import (
    capped_recall_at_k,
    evaluate_capped_recall,
    evaluate_litsearch_comparison,
    evaluate_run,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.pipeline import HybridRetriever
from retrieval.sparse import BM25Index
from retrieval.types import Document, RetrievalConfig, SearchHit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_checkpoint1 import build_report, to_markdown
from plot_benchmark_results import load_results
import run_public_bench


def test_bm25_prefers_matching_document():
    index = BM25Index([
        Document("a", "cache eviction and memory"),
        Document("b", "unrelated compiler paper"),
        Document("c", "cache cache memory"),
    ])
    assert index.search("cache memory", top_n=1)[0].doc_id == "c"


def test_bm25_candidate_scope_ranks_only_allowed_documents():
    index = BM25Index([
        Document("global-best", "cache cache cache"),
        Document("paper-a", "cache"),
        Document("paper-b", "compiler"),
    ])
    hits = index.search("cache", top_n=1, allowed_ids={"paper-a", "paper-b"})
    assert [hit.doc_id for hit in hits] == ["paper-a"]


def test_rrf_fuses_rankings_and_aggregates_chunks_to_papers():
    fused = rrf_fuse(
        {
            "sparse": [SearchHit("chunk-1", 5.0, "paper-a"), SearchHit("chunk-2", 4.0, "paper-b")],
            "dense": [SearchHit("chunk-2", 0.9, "paper-b"), SearchHit("chunk-3", 0.8, "paper-a")],
        },
        rrf_k=1,
    )
    documents = {
        "chunk-1": Document("chunk-1", "", "paper-a"),
        "chunk-2": Document("chunk-2", "", "paper-b"),
        "chunk-3": Document("chunk-3", "", "paper-a"),
    }
    papers = aggregate_to_papers(fused, documents)
    assert {hit.doc_id for hit in papers} == {"paper-a", "paper-b"}
    assert papers[0].score >= papers[1].score


def test_metrics_match_small_hand_computed_run():
    ranking = ["wrong", "right-1", "right-2"]
    relevant = {"right-1", "right-2"}
    assert recall_at_k(ranking, relevant, 2) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    assert round(ndcg_at_k(ranking, relevant, 2), 6) == round(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3)), 6
    )
    assert evaluate_run({"q": ranking}, {"q": relevant}, ks=(2,)) == {
        "recall@2": 0.5,
        "ndcg@10": ndcg_at_k(ranking, relevant, 10),
        "mrr": 0.5,
    }


def test_lmeb_capped_recall_caps_relevant_count_at_cutoff():
    relevant = {f"d{i}" for i in range(20)}
    ranking = [f"d{i}" for i in range(5)] + ["wrong"] * 5
    assert recall_at_k(ranking, relevant, 10) == 0.25
    assert capped_recall_at_k(ranking, relevant, 10) == 0.5
    assert evaluate_capped_recall({"q": ranking}, {"q": relevant}, 10) == 0.5


def test_hybrid_retriever_records_rewrite_embed_search_and_fuse_timings():
    class Dense:
        def embed_query(self, query):
            assert query == "rewritten"
            return [1.0]

        def search_vector(self, vector, top_n):
            assert vector == [1.0]
            return [SearchHit("dense", 1.0)]

    retriever = HybridRetriever(
        sparse_index=BM25Index([Document("sparse", "rewritten query")]),
        dense_index=Dense(),
        query_rewriter=lambda query: "rewritten",
        config=RetrievalConfig(rewrite=True, top_k=2, sparse_weight=0.6, dense_weight=0.4),
    )
    result = retriever.search("original")
    assert [hit.doc_id for hit in result.hits] == ["sparse", "dense"]
    assert {"rewrite_ms", "embed_ms", "dense_search_ms", "sparse_search_ms", "fuse_ms"} <= result.timings_ms.keys()


def test_dense_index_applies_model_specific_query_prefix():
    class Embedder:
        query_prefix = "Instruct: retrieve papers\nQuery: "
        passage_prefix = ""
        timeout = 1

        def embed(self, texts):
            self.texts = texts
            return [[1.0]]

    embedder = Embedder()
    index = QdrantIndex("test", embedder, "http://localhost:6333")
    assert index.embed_query("find papers") == [1.0]
    assert embedder.texts == ["Instruct: retrieve papers\nQuery: find papers"]


def test_dense_index_reports_upsert_progress():
    class Embedder:
        passage_prefix = ""
        timeout = 1

        def embed(self, texts):
            return [[1.0] for _ in texts]

    index = QdrantIndex("test", Embedder(), "http://localhost:6333")
    index._request = lambda *args, **kwargs: {}
    progress = []
    index.create(
        [Document(str(i), "text") for i in range(3)],
        batch_size=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(2, 3), (3, 3)]


def test_dense_index_sends_candidate_filter_to_qdrant():
    class Embedder:
        timeout = 1

    index = QdrantIndex("test", Embedder(), "http://localhost:6333")
    requests = []

    def request(method, path, body):
        requests.append((method, path, body))
        return {"result": []}

    index._request = request
    index.search_vector([1.0], top_n=10, allowed_ids={"chunk-b", "chunk-a"})

    body = requests[0][2]
    assert body["limit"] == 2
    assert body["filter"]["must"][0]["match"]["any"] == ["chunk-a", "chunk-b"]


def test_jsonl_benchmark_keeps_exclusions_and_qrels(tmp_path):
    documents = tmp_path / "documents.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    documents.write_text(json.dumps({"doc_id": "d1", "text": "find it"}) + "\n")
    queries.write_text(json.dumps({"query_id": "q1", "query": "find it", "excluded_ids": ["d2"]}) + "\n")
    qrels.write_text(json.dumps({"query_id": "q1", "doc_id": "d1"}) + "\n")
    benchmark = load_jsonl_benchmark(documents, queries, qrels)
    assert benchmark.queries == {"q1": "find it"}
    assert benchmark.qrels == {"q1": {"d1"}}
    assert benchmark.excluded_ids == {"q1": {"d2"}}


def test_jsonl_cli_runs_end_to_end(tmp_path):
    documents = tmp_path / "documents.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    documents.write_text(json.dumps({"doc_id": "d1", "text": "matching phrase"}) + "\n")
    queries.write_text(json.dumps({"query_id": "q1", "query": "matching phrase"}) + "\n")
    qrels.write_text(json.dumps({"query_id": "q1", "doc_id": "d1"}) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "run_public_bench.py"),
            "--benchmark",
            "jsonl",
            "--documents",
            str(documents),
            "--queries",
            str(queries),
            "--qrels",
            str(qrels),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(result.stdout)
    assert output["config"]["benchmark"] == "jsonl"
    assert output["metrics"]["recall@5"] == 1.0
    assert output["queries"] == 1


def test_scholargym_loader_reads_released_and_readme_schemas(tmp_path):
    paper_db = tmp_path / "scholargym_paper_db.json"
    paper_db.write_text(json.dumps({
        "2101.00001v2": {"title": "Paper", "abstract": "Abstract", "year": 2021},
        "2101.00002": {"title": "Other", "abstract": "Text"},
    }))
    benchmark = tmp_path / "scholargym_bench.jsonl"
    benchmark.write_text("\n".join([
        json.dumps({
            "qid": "q1",
            "query": "find paper",
            "cited_paper": [{"arxiv_id": "arxiv:2101.00001"}, {"arxiv_id": "2101.00002"}],
            "gt_label": [1, "0"],
            "source": "PASA_AutoScholar",
        }),
        json.dumps({"query_id": "q2", "query": "readme schema", "gt_arxiv_ids": ["2101.00002"]}),
    ]) + "\n")

    loaded = load_scholargym_benchmark(paper_db, benchmark)
    assert {document.doc_id for document in loaded.documents} == {"2101.00001", "2101.00002"}
    assert loaded.queries == {"q1": "find paper", "q2": "readme schema"}
    assert loaded.qrels == {"q1": {"2101.00001"}, "q2": {"2101.00002"}}
    assert loaded.query_metadata["q1"]["source"] == "PASA_AutoScholar"


def test_scholargym_defaults_to_the_shared_dataset_cache(tmp_path):
    paper_db, benchmark = scholargym_paths(tmp_path / "huggingface")
    root = tmp_path / "huggingface" / "datasets" / "datasets--shenhao--ScholarGym"
    assert paper_db == root / "scholargym_paper_db.json"
    assert benchmark == root / "scholargym_bench.jsonl"


def test_scholargym_resolves_huggingface_snapshot(tmp_path):
    root = tmp_path / "huggingface" / "datasets" / "datasets--shenhao--ScholarGym"
    snapshot = root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("abc123\n")
    (snapshot / "scholargym_paper_db.json").write_text("{}")
    (snapshot / "scholargym_bench.jsonl").write_text("\n")

    paper_db, benchmark = scholargym_paths(dataset_dir=root)
    assert paper_db == snapshot / "scholargym_paper_db.json"
    assert benchmark == snapshot / "scholargym_bench.jsonl"


def test_checkpoint1_report_selects_litsearch_first(tmp_path):
    root = tmp_path / "results" / "public"
    sparse = root / "sparse"
    model = root / "qwen3-embedding-4b"
    sparse.mkdir(parents=True)
    model.mkdir()

    def write(path, metrics, comparison=None):
        path.write_text(json.dumps({"metrics": metrics, **({"litsearch_paper_comparison": comparison} if comparison else {})}))

    write(sparse / "litsearch-sparse.json", {"recall@5": 0.3, "recall@20": 0.4, "recall@100": 0.7})
    write(sparse / "mteb-scifact-sparse.json", {"ndcg@10": 0.2, "recall@100": 0.8})
    write(model / "litsearch-dense.json", {"recall@5": 0.5, "recall@20": 0.6, "recall@100": 0.9}, {
        "delta_vs_paper_bm25": {"average": {"specific": {"recall@5": 0.1}}}
    })
    write(model / "mteb-scifact-dense.json", {"ndcg@10": 0.1, "recall@100": 0.9})

    report = build_report(root)
    assert report["winner_by_litsearch"] == "qwen3-embedding-4b / dense"
    assert "LitSearch R@5" in to_markdown(report)


def test_plotter_groups_results_by_dataset_and_model_pipeline(tmp_path):
    root = tmp_path / "results" / "public"
    (root / "sparse").mkdir(parents=True)
    (root / "qwen3-embedding-4b").mkdir()
    (root / "sparse" / "litsearch-sparse.json").write_text(json.dumps({
        "config": {"benchmark": "litsearch", "mode": "sparse"},
        "metrics": {"recall@20": 0.4},
        "litsearch_paper_comparison": {
            "ours": {
                "average": {
                    "broad": {"queries": 155, "recall@20": 0.35},
                    "specific": {"queries": 442, "recall@5": 0.45, "recall@20": 0.55},
                }
            }
        },
    }))
    (root / "qwen3-embedding-4b" / "mteb-scifact-dense.json").write_text(json.dumps({
        "config": {"benchmark": "mteb", "dataset": "scifact", "mode": "dense", "embedding_model": "Qwen/Qwen3-Embedding-4B"},
        "metrics": {"ndcg@10": 0.5},
    }))
    for scope, score in (("global", 0.2), ("paper", 0.6), ("two-stage", 0.4)):
        (root / "qwen3-embedding-4b" / f"qasper-{scope}-dense.json").write_text(json.dumps({
            "config": {
                "benchmark": "qasper",
                "qasper_scope": scope,
                "qasper_paper_top_k": 20 if scope == "two-stage" else None,
                "mode": "dense",
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
            },
            "metrics": {"ndcg@10": score},
            **({
                "qasper": {
                    "paper_retrieval": {
                        "metrics": {"recall@5": 0.7, "recall@20": 0.9}
                    }
                }
            } if scope == "two-stage" else {}),
        }))
    (root / "qwen3-embedding-4b" / "qasper-two-stage-k100-dense.json").write_text(json.dumps({
        "config": {
            "benchmark": "qasper",
            "qasper_scope": "two-stage",
            "qasper_paper_top_k": 100,
            "mode": "dense",
            "embedding_model": "Qwen/Qwen3-Embedding-4B",
        },
        "metrics": {"ndcg@10": 0.45},
        "qasper": {
            "paper_retrieval": {
                "metrics": {"recall@5": 0.7, "recall@20": 0.9}
            }
        },
    }))

    results = load_results(root)
    assert [row["label"] for row in results["litsearch"]] == ["BM25 / sparse"]
    assert results["litsearch-average-broad"][0]["metrics"]["recall@20"] == 0.35
    assert results["litsearch-average-specific"][0]["metrics"]["recall@5"] == 0.45
    assert results["mteb-scifact"][0]["label"] == "Qwen3-Embedding-4B / dense"
    assert results["qasper-global"][0]["metrics"]["ndcg@10"] == 0.2
    assert results["qasper-paper"][0]["metrics"]["ndcg@10"] == 0.6
    two_stage = {row["label"]: row for row in results["qasper-two-stage"]}
    assert two_stage["Qwen3-Embedding-4B / dense / paper-k=20"]["metrics"]["ndcg@10"] == 0.4
    assert two_stage["Qwen3-Embedding-4B / dense / paper-k=100"]["metrics"]["ndcg@10"] == 0.45
    assert len(results["qasper-two-stage-papers"]) == 2


def test_bright_loader_is_strictly_local(monkeypatch, tmp_path):
    calls = []

    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DownloadMode:
        REUSE_DATASET_IF_EXISTS = "reuse"

    def load_dataset(dataset_id, config, **kwargs):
        calls.append((dataset_id, config, kwargs))
        assert kwargs["split"] == "biology"
        if config == "examples":
            return [{"id": "q1", "query": "find", "gold_ids": ["d1"], "excluded_ids": []}]
        return [{"id": "d1", "content": "answer"}]

    fake_datasets = SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    benchmark = load_bright_hf("biology", cache_dir=tmp_path / "hf")

    assert len(benchmark.documents) == 1
    assert len(calls) == 2
    for _, _, kwargs in calls:
        assert kwargs["cache_dir"] == str(tmp_path / "hf" / "datasets")
        assert kwargs["download_config"].local_files_only is True
        assert kwargs["download_mode"] == "reuse"


def test_litsearch_loader_reads_hf_schema(monkeypatch, tmp_path):
    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DownloadMode:
        REUSE_DATASET_IF_EXISTS = "reuse"

    def load_dataset(dataset_id, config, **kwargs):
        assert dataset_id == "princeton-nlp/LitSearch"
        if config == "query":
            return [{"query": "find papers", "corpusids": [7], "query_set": "inline-citation", "specificity": 1}]
        return [{"corpusid": 7, "title": "Title", "abstract": "Abstract"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    ))
    benchmark = load_litsearch_hf(cache_dir=tmp_path)
    assert benchmark.queries == {"q1": "find papers"}
    assert benchmark.qrels == {"q1": {"d7"}}
    assert benchmark.query_metadata == {"q1": {"query_set": "inline-citation", "specificity": 1}}
    assert benchmark.documents[0].text == "Title Abstract"


def test_litsearch_comparison_reports_paper_cutoffs():
    benchmark = SimpleNamespace(
        qrels={"q1": {"d1"}, "q2": {"d2"}},
        query_metadata={
            "q1": {"query_set": "inline-citation", "specificity": 0},
            "q2": {"query_set": "manual-acl", "specificity": 1},
        },
    )
    report = evaluate_litsearch_comparison(benchmark, {"q1": ["d1"], "q2": ["d2"]})
    assert report["ours"]["inline-citation"]["broad"]["recall@20"] == 1.0
    assert report["ours"]["author-written"]["specific"]["recall@5"] == 1.0
    assert "recall@20" in report["paper_bm25"]["inline-citation"]["broad"]
    assert report["paper_ndcg@10"]["E5-large-v2"]["specific"] == 0.453


def test_mteb_loader_reads_beir_schema(monkeypatch, tmp_path):
    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DownloadMode:
        REUSE_DATASET_IF_EXISTS = "reuse"

    def load_dataset(dataset_id, config, **kwargs):
        assert dataset_id == "mteb/scifact"
        if config == "corpus":
            assert kwargs["split"] == "corpus"
            return [{"_id": "d1", "title": "Paper", "text": "Evidence"}]
        if config == "queries":
            assert kwargs["split"] == "queries"
            return [
                {"_id": "q1", "text": "claim"},
                {"_id": "train-only", "text": "not in test qrels"},
            ]
        assert kwargs["split"] == "test"
        return [{"query-id": "q1", "corpus-id": "d1", "score": 0.5}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    ))
    benchmark = load_mteb_hf("mteb/scifact", cache_dir=tmp_path)
    assert benchmark.queries == {"q1": "claim"}
    assert benchmark.qrels == {"q1": {"d1"}}
    assert benchmark.documents[0].text == "Paper Evidence"


def test_mteb_defaults_to_scidocs_and_accepts_any_dataset_id():
    assert DEFAULT_MTEB_DATASET == "scidocs"
    assert mteb_dataset_id("scidocs") == "mteb/scidocs"
    assert mteb_dataset_id("cqadupstack-android") == "mteb/cqadupstack-android"
    assert mteb_dataset_id("my-org/custom-beir") == "my-org/custom-beir"


def test_mteb_loader_excludes_identical_query_document_ids(monkeypatch):
    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DownloadMode:
        REUSE_DATASET_IF_EXISTS = "reuse"

    def load_dataset(dataset_id, config, **kwargs):
        if config == "corpus":
            return [{"_id": "same", "text": "document"}]
        if config == "queries":
            return [{"_id": "same", "text": "query"}]
        return [{"query-id": "same", "corpus-id": "other", "score": 1}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    ))
    benchmark = load_mteb_hf("mteb/arguana")
    assert benchmark.excluded_ids == {"same": {"same"}}


def test_qasper_loader_supports_global_and_paper_scoped_retrieval(monkeypatch):
    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DownloadMode:
        REUSE_DATASET_IF_EXISTS = "reuse"

    def load_dataset(dataset_id, config, **kwargs):
        assert dataset_id == DEFAULT_QASPER_DATASET
        assert kwargs["split"] == "test"
        if config == "QASPER-corpus":
            return [
                {"id": "1911.00001_0", "title": "Paper One", "text": "Abstract"},
                {"id": "1911.00001_1", "title": "Methods", "text": "Gold evidence"},
                {"id": "1911.00002_0", "title": "Paper Two", "text": "Distractor"},
            ]
        if config == "QASPER-queries":
            return [{"id": "q1", "text": "What method was used?"}]
        if config == "QASPER-qrels":
            return [{"query-id": "q1", "corpus-id": "1911.00001_1", "score": 1}]
        assert config == "QASPER-top_ranked"
        return [{"query-id": "q1", "corpus-ids": ["1911.00001_0", "1911.00001_1"]}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    ))

    global_benchmark = load_qasper_hf(scope="global")
    scoped_benchmark = load_qasper_hf(scope="paper")

    assert len(global_benchmark.documents) == 3
    assert global_benchmark.documents[1].paper_id == "1911.00001"
    assert global_benchmark.documents[1].text == "Methods Gold evidence"
    assert global_benchmark.candidate_ids is None
    assert scoped_benchmark.candidate_ids == {
        "q1": {"1911.00001_0", "1911.00001_1"}
    }
    assert scoped_benchmark.qrels == {"q1": {"1911.00001_1"}}


def test_qasper_paper_benchmark_derives_target_from_gold_chunks(monkeypatch):
    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DownloadMode:
        REUSE_DATASET_IF_EXISTS = "reuse"

    def load_dataset(dataset_id, config, **kwargs):
        assert dataset_id == DEFAULT_QASPER_RAW_DATASET
        assert config == "qasper"
        assert kwargs["split"] == "train+validation+test"
        return [
            {"id": "1911.00001", "title": "Target Paper", "abstract": "Caching method"},
            {"id": "1911.00002", "title": "Distractor", "abstract": "Parsing method"},
        ]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    ))
    chunks = Benchmark(
        documents=[
            Document("1911.00001_0", "target chunk", "1911.00001"),
            Document("1911.00002_0", "other chunk", "1911.00002"),
        ],
        queries={"q1": "What caching method is used?"},
        qrels={"q1": {"1911.00001_0"}},
        excluded_ids={"q1": set()},
    )

    papers = load_qasper_paper_benchmark_hf(chunk_benchmark=chunks)

    assert papers.qrels == {"q1": {"1911.00001"}}
    assert papers.documents[0].text == "Target Paper Caching method"
    assert papers.query_metadata["q1"]["target_paper_id"] == "1911.00001"
    assert qasper_chunk_candidates(
        chunks.documents,
        {"q1": ["1911.00002", "1911.00001"]},
    ) == {"q1": {"1911.00001_0", "1911.00002_0"}}


def test_qasper_raw_dataset_resolves_cached_snapshot(tmp_path):
    repository = tmp_path / "datasets" / "datasets--allenai--qasper"
    snapshot = repository / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("abc123\n")

    assert qasper_raw_dataset_path(cache_dir=tmp_path) == str(snapshot)


def test_qasper_paper_documents_load_staged_parquet_without_hub(monkeypatch, tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.parquet").write_bytes(b"fixture")

    class DownloadConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def load_dataset(dataset_id, **kwargs):
        assert dataset_id == "parquet"
        assert kwargs["split"] == "data"
        assert len(kwargs["data_files"]["data"]) == 3
        return [{"id": "1911.00001", "title": "Paper", "abstract": "Abstract"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        load_dataset=load_dataset,
    ))

    documents = load_qasper_paper_documents_hf(dataset_id=tmp_path)

    assert documents == [
        Document("1911.00001", "Paper Abstract", "1911.00001", {"title": "Paper"})
    ]


def test_qasper_two_stage_restricts_chunks_to_retrieved_papers(monkeypatch):
    chunks = Benchmark(
        documents=[
            Document("target_0", "cache evidence", "target"),
            Document("other_0", "compiler evidence", "other"),
        ],
        queries={"q1": "cache method"},
        qrels={"q1": {"target_0"}},
        excluded_ids={"q1": set()},
    )
    papers = Benchmark(
        documents=[
            Document("target", "cache method paper", "target"),
            Document("other", "compiler paper", "other"),
        ],
        queries=dict(chunks.queries),
        qrels={"q1": {"target"}},
        excluded_ids={"q1": set()},
    )
    monkeypatch.setattr(
        run_public_bench,
        "load_qasper_paper_benchmark_hf",
        lambda **kwargs: papers,
    )
    args = SimpleNamespace(
        benchmark="qasper",
        mode="sparse",
        qasper_paper_top_k=1,
        qasper_paper_collection=None,
        dataset_id=None,
        qasper_raw_dataset_id=DEFAULT_QASPER_RAW_DATASET,
        split="test",
        cache_dir=None,
        qasper_query_limit=None,
        top_n=10,
        top_k=10,
        collection=None,
    )

    restricted, run, _, details = run_public_bench._run_qasper_two_stage(
        args,
        argparse.ArgumentParser(),
        chunks,
    )

    assert restricted.candidate_ids == {"q1": {"target_0"}}
    assert run == {"q1": ["target_0"]}
    assert details["paper_metrics"]["recall@1"] == 1.0
    assert details["conditional_evidence_metrics"]["queries"] == 1


def test_retriever_passes_same_candidate_scope_to_sparse_and_dense():
    class Sparse:
        def search(self, query, top_n, allowed_ids):
            assert allowed_ids == {"allowed"}
            return [SearchHit("allowed", 1.0), SearchHit("blocked", 0.9)]

    class Dense:
        def embed_query(self, query):
            return [1.0]

        def search_vector(self, vector, top_n, allowed_ids):
            assert allowed_ids == {"allowed"}
            return [SearchHit("allowed", 1.0), SearchHit("blocked", 0.9)]

    retriever = HybridRetriever(
        sparse_index=Sparse(),
        dense_index=Dense(),
        config=RetrievalConfig(top_k=10),
    )
    result = retriever.search("question", allowed_ids={"allowed"})
    assert [hit.doc_id for hit in result.hits] == ["allowed"]
