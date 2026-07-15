import json
import math
import sys
from types import SimpleNamespace

from retrieval.benchmarks import (
    DEFAULT_MTEB_DATASET,
    load_bright_hf,
    load_jsonl_benchmark,
    load_litsearch_hf,
    load_mteb_hf,
    mteb_dataset_id,
)
from retrieval.dense import QdrantIndex
from retrieval.fusion import aggregate_to_papers, rrf_fuse
from retrieval.metrics import evaluate_litsearch_comparison, evaluate_run, ndcg_at_k, recall_at_k, reciprocal_rank
from retrieval.pipeline import HybridRetriever
from retrieval.sparse import BM25Index
from retrieval.types import Document, RetrievalConfig, SearchHit


def test_bm25_prefers_matching_document():
    index = BM25Index([
        Document("a", "cache eviction and memory"),
        Document("b", "unrelated compiler paper"),
        Document("c", "cache cache memory"),
    ])
    assert index.search("cache memory", top_n=1)[0].doc_id == "c"


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
            return [{"id": "d1", "title": "Paper", "text": "Evidence"}]
        if config == "queries":
            assert kwargs["split"] == "queries"
            return [
                {"id": "q1", "text": "claim"},
                {"id": "train-only", "text": "not in test qrels"},
            ]
        assert kwargs["split"] == "test"
        return [{"query-id": "q1", "corpus-id": "d1", "score": 1}]

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
            return [{"id": "same", "text": "document"}]
        if config == "queries":
            return [{"id": "same", "text": "query"}]
        return [{"query-id": "same", "corpus-id": "other", "score": 1}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        DownloadConfig=DownloadConfig,
        DownloadMode=DownloadMode,
        load_dataset=load_dataset,
    ))
    benchmark = load_mteb_hf("mteb/arguana")
    assert benchmark.excluded_ids == {"same": {"same"}}
