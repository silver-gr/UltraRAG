"""Federated merge: sources with incomparable native scores share the candidate list by rank."""
import logging
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import Field
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode

from config import RetrievalConfig
from embeddings import TrackedVoyageReranker
from eval_harness import RetrievalProbe
from federated_query import FederatedQueryEngine, FederatedRetriever, IndexSource
from main import UltraRAG
from query_engine import HybridQueryEngine, RAGQueryEngine, RerankScoreThreshold
from research_mode import ResearchRetriever

pytestmark = pytest.mark.unit

QUERY = QueryBundle(query_str="dopamine and ADHD")


class StubRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._stub_nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._stub_nodes


class StubReranker(BaseNodePostprocessor):
    """Scores the first node above a 0.1 threshold and the rest below it."""

    def _postprocess_nodes(self, nodes, query_bundle=None):
        return [NodeWithScore(node=n.node, score=0.9 if i == 0 else 0.05) for i, n in enumerate(nodes)]


class RecordingReranker(BaseNodePostprocessor):
    """Reverses the candidates, so its order can't be mistaken for the merge order."""

    top_n: int = 1000
    calls: list = Field(default_factory=list)

    def _postprocess_nodes(self, nodes, query_bundle=None):
        self.calls.append([n.node.node_id for n in nodes])
        ranked = list(reversed(nodes))[: self.top_n]
        return [NodeWithScore(node=n.node, score=1.0 - i / 1000) for i, n in enumerate(ranked)]


class KeepOnlyRrf(BaseNodePostprocessor):
    """Stands in for the date filter."""

    def _postprocess_nodes(self, nodes, query_bundle=None):
        return [n for n in nodes if _origin(n) == "rrf_source"]


def _nodes(prefix, scores, same_file=False):
    return [
        NodeWithScore(
            node=TextNode(
                id_=f"{prefix}-{i}",
                text=f"{prefix} chunk {i}",
                metadata={"file_path": prefix if same_file else f"{prefix}/{i}.md"},
            ),
            score=score,
        )
        for i, score in enumerate(scores)
    ]


def _rrf_scores(n):
    return [1 / (60 + i) for i in range(n)]


def _cosine_scores(n):
    return [0.9 - 0.02 * i for i in range(n)]


def _source(name, nodes, weight):
    return IndexSource(
        name=name, source_type="vault", weight=weight, custom_retriever=StubRetriever(nodes)
    )


def _config(**retrieval):
    return SimpleNamespace(retrieval=RetrievalConfig(**retrieval))


def _retriever(sources, **retrieval):
    return FederatedRetriever(sources, _config(**retrieval), parallel=False)


def _two_scale_sources():
    return [
        _source("rrf_source", _nodes("rrf", _rrf_scores(10)), 1.0),
        _source("cosine_source", _nodes("cosine", _cosine_scores(10)), 0.5),
    ]


def _origin(node):
    return node.node.metadata["retrieval_source"]


def _ids(nodes):
    return [n.node.node_id for n in nodes]


def _federated_rag(monkeypatch, reranker, sources=None, **retrieval):
    """An UltraRAG with only what query_federated needs."""
    # MockLLM echoes the prompt, so the answer shows what reached the synthesiser.
    monkeypatch.setattr(Settings, "_llm", MockLLM())
    rag = object.__new__(UltraRAG)
    rag.config = _config(use_self_correction=False, **retrieval)
    rag.federated_engine = FederatedQueryEngine(
        sources or _two_scale_sources(), rag.config, reranker=reranker, use_cache=False
    )
    return rag


class TestRerankerWiring:
    def test_query_federated_sends_the_reranked_order_to_the_synthesiser(self, monkeypatch):
        reranker = RecordingReranker()
        rag = _federated_rag(monkeypatch, reranker)

        result = rag.query_federated("q", max_sources=2)

        assert len(reranker.calls) == 1
        merged = reranker.calls[0]
        assert len(merged) == 20, "every candidate reaches the reranker, before the max_sources cut"
        first, second = (i.replace("-", " chunk ") for i in reversed(merged[-2:]))
        assert [s["excerpt"] for s in result["sources"]] == [first, second]
        assert [s["score"] for s in result["sources"]] == [pytest.approx(1.0), pytest.approx(0.999)]
        answer = result["answer"]
        assert answer.index(first) < answer.index(second)
        assert merged[0].replace("-", " chunk ") not in answer

    def test_date_filter_runs_before_the_reranker(self, monkeypatch):
        reranker = RecordingReranker(top_n=2)
        rag = _federated_rag(monkeypatch, reranker)
        monkeypatch.setattr("main.create_temporal_filter", lambda preset: KeepOnlyRrf())

        result = rag.query_federated("q", date_filter="last_week")

        assert all(i.startswith("rrf") for i in reranker.calls[0])
        assert len(result["sources"]) == 2

    @pytest.mark.parametrize("max_sources, expected", [(None, 30), (25, 25), (40, 30)])
    def test_source_count_is_capped_by_rerank_top_n(self, monkeypatch, max_sources, expected):
        sources = [
            _source("a", _nodes("a", _rrf_scores(25)), 1.0),
            _source("b", _nodes("b", _rrf_scores(25)), 1.0),
        ]
        rag = _federated_rag(monkeypatch, RecordingReranker(top_n=30), sources, rerank_top_n=30)

        result = rag.query_federated("q", max_sources=max_sources)

        assert len(result["sources"]) == expected

    def test_final_source_mix_is_logged(self, monkeypatch, caplog):
        rag = _federated_rag(monkeypatch, RecordingReranker(top_n=2))

        with caplog.at_level(logging.INFO, logger="main"):
            rag.query_federated("q")

        assert "Federated final per-source counts" in caplog.text
        assert "'vault': 2" in caplog.text

    def test_eval_probe_measures_the_production_path(self, monkeypatch):
        reranker = RecordingReranker()
        engine = _federated_rag(monkeypatch, reranker).federated_engine
        probe = object.__new__(RetrievalProbe)
        probe._vault_engine = object()
        probe._federated_engine = engine

        result = probe.run("q", scope="federated")

        assert _ids(result.pre_nodes) == reranker.calls[0]
        assert _ids(result.post_nodes) == _ids(engine.retrieve("q"))

    def test_federated_research_reranks_every_retrieval(self):
        reranker = RecordingReranker()
        retriever = FederatedRetriever(
            _two_scale_sources(), _config(), reranker=reranker, parallel=False
        )

        first = retriever.retrieve(QUERY)
        retriever.retrieve(QUERY)

        assert retriever.rerank_calls == 2
        assert _ids(first) == list(reversed(reranker.calls[0]))

    def test_federated_research_ignores_the_fallback_reranker(self):
        retriever = FederatedRetriever(
            _two_scale_sources(),
            _config(),
            reranker=SimilarityPostprocessor(similarity_cutoff=0.7),
            parallel=False,
        )

        assert retriever.retrieve(QUERY)
        assert retriever.rerank_calls == 0

    @pytest.mark.parametrize("reranker, calls", [(RecordingReranker(), 1), (None, 0)], ids=["on", "off"])
    def test_rerank_call_count_is_logged(self, monkeypatch, caplog, reranker, calls):
        engine = _federated_rag(monkeypatch, reranker).federated_engine

        with caplog.at_level(logging.INFO, logger="federated_query"):
            engine.retrieve("q")

        assert f"rerank calls for this query: {calls}" in caplog.text


class TestRankNormalisedMerge:
    def test_top_k_contains_both_sources(self):
        merged = _retriever(_two_scale_sources(), federated_final_top_k=10).retrieve(QUERY)

        assert len(merged) == 10
        assert {_origin(n) for n in merged} == {"rrf_source", "cosine_source"}

    def test_higher_weight_wins_at_equal_rank(self):
        # The lower-weight source goes first, so a tie would not put rrf_source on top.
        sources = list(reversed(_two_scale_sources()))

        merged = _retriever(sources, federated_final_top_k=10).retrieve(QUERY)

        assert _origin(merged[0]) == "rrf_source"

    def test_native_score_is_preserved(self):
        merged = _retriever(_two_scale_sources(), federated_final_top_k=20).retrieve(QUERY)

        expected = dict(
            zip((f"rrf-{i}" for i in range(10)), _rrf_scores(10))
        ) | dict(zip((f"cosine-{i}" for i in range(10)), _cosine_scores(10)))
        for node in merged:
            assert node.node.metadata["native_score"] == pytest.approx(expected[node.node.node_id])

    def test_rank_follows_native_order_not_retriever_order(self):
        shuffled = _nodes("rrf", [0.01, 0.03, 0.02])
        merged = _retriever([_source("only", shuffled, 1.0)]).retrieve(QUERY)

        assert [n.node.node_id for n in merged] == ["rrf-1", "rrf-2", "rrf-0"]

    def test_slots_are_shared_in_proportion_to_weight(self):
        """The default four-source setup: no source monopolises or vanishes from the cut."""
        sources = [
            _source("vault", _nodes("vault", _rrf_scores(100)), 1.0),
            _source("books", _nodes("books", _rrf_scores(100)), 0.9),
            _source("conversations", _nodes("conv", _rrf_scores(100)), 0.8),
            _source("saved_items", _nodes("saved", [0.9 - 0.004 * i for i in range(100)]), 0.5),
        ]

        merged = _retriever(sources, federated_final_top_k=120).retrieve(QUERY)
        counts = Counter(_origin(n) for n in merged)

        total_weight = 3.2
        for source in sources:
            assert counts[source.name] == pytest.approx(120 * source.weight / total_weight, abs=2)

    def test_zero_weight_source_ranks_last_without_error(self):
        sources = [
            _source("muted", _nodes("muted", [0.9, 0.8]), 0.0),
            _source("normal", _nodes("normal", _rrf_scores(2)), 1.0),
        ]

        merged = _retriever(sources).retrieve(QUERY)

        assert [_origin(n) for n in merged] == ["normal", "normal", "muted", "muted"]
        assert merged[-1].score == 0.0


class TestParallelRetrieval:
    def test_a_slow_source_is_not_dropped(self, monkeypatch):
        """Waiting on one future at a time with a timeout dropped whichever source came first."""

        class AlwaysTooSlow:
            def __init__(self, value):
                self.value = value

            def result(self, timeout=None):
                if timeout is not None:
                    raise TimeoutError
                return self.value

        class Executor:
            def __init__(self, max_workers):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, *args):
                return AlwaysTooSlow(fn(*args))

        monkeypatch.setattr("federated_query.ThreadPoolExecutor", Executor)
        retriever = FederatedRetriever(_two_scale_sources(), _config(), parallel=True)

        assert {_origin(n) for n in retriever.retrieve(QUERY)} == {"rrf_source", "cosine_source"}


class TestMergeStillDedupesAndCaps:
    def test_per_document_cap_applies_after_the_merge_sort(self):
        sources = [
            _source("one_doc", _nodes("doc", _cosine_scores(10), same_file=True), 1.0),
            _source("many_docs", _nodes("other", _rrf_scores(10)), 1.0),
        ]

        merged = _retriever(
            sources, federated_max_chunks_per_document=3, federated_final_top_k=10
        ).retrieve(QUERY)

        counts = Counter(_origin(n) for n in merged)
        assert counts == {"one_doc": 3, "many_docs": 7}
        # The capped document keeps its best-ranked chunks.
        kept = [n.node.node_id for n in merged if _origin(n) == "one_doc"]
        assert kept == ["doc-0", "doc-1", "doc-2"]

    def test_duplicate_text_keeps_the_higher_scored_copy(self):
        low = _nodes("low", [0.9, 0.8])
        high = _nodes("high", [0.02, 0.01])
        low[0].node.text = high[0].node.text = "same text in both sources"

        merged = _retriever(
            [_source("low", low, 0.5), _source("high", high, 1.0)]
        ).retrieve(QUERY)

        twins = [n for n in merged if n.node.text == "same text in both sources"]
        assert [n.node.node_id for n in twins] == ["high-0"]


class TestDownstreamConsumers:
    def test_native_score_stays_out_of_reranker_and_llm_text(self):
        retriever = _retriever([_source("only", _nodes("a", [0.5]), 1.0)])
        retriever.retrieve(QUERY)
        node = retriever.retrieve(QUERY)[0].node  # second pass: the node object is reused

        assert "native_score" not in node.get_content(metadata_mode=MetadataMode.EMBED)
        assert "native_score" not in node.get_content(metadata_mode=MetadataMode.LLM)
        assert node.excluded_llm_metadata_keys.count("native_score") == 1

    @pytest.mark.parametrize(
        "reranker", [None, SimilarityPostprocessor(similarity_cutoff=0.7)], ids=["none", "fallback"]
    )
    def test_engine_without_real_reranker_returns_results(self, monkeypatch, reranker):
        """A cosine cutoff on rank-based scores would remove every node."""
        monkeypatch.setattr(Settings, "_llm", MockLLM())
        engine = FederatedQueryEngine(
            _two_scale_sources(),
            _config(similarity_threshold=0.3, rerank_score_threshold=0.1),
            reranker=reranker,
            use_cache=False,
        )

        assert engine.query_engine.retrieve(QUERY)

    def test_real_reranker_and_its_threshold_still_apply(self, monkeypatch):
        monkeypatch.setattr(Settings, "_llm", MockLLM())
        engine = FederatedQueryEngine(
            _two_scale_sources(),
            _config(rerank_score_threshold=0.1),
            reranker=StubReranker(),
            use_cache=False,
        )

        kept = engine.query_engine.retrieve(QUERY)

        assert [n.score for n in kept] == [0.9]

    def test_research_score_floor_skips_mixed_federated_scores(self):
        research = ResearchRetriever(base_retriever=StubRetriever([]), llm=Mock())
        federated = _retriever(_two_scale_sources()).retrieve(QUERY)

        assert research._check_score_floor(federated) == (False, 0.0)

    def test_research_score_floor_still_applies_to_plain_cosine_scores(self):
        research = ResearchRetriever(base_retriever=StubRetriever([]), llm=Mock())

        assert research._check_score_floor(_nodes("low", [0.1])) == (True, pytest.approx(0.1))
        assert research._check_score_floor(_nodes("high", [0.8])) == (False, pytest.approx(0.8))


class TestQuotaFallbackThreshold:
    """RERANK_SCORE_THRESHOLD must not remove every node when the reranker fell back."""

    @pytest.fixture
    def tracked(self, monkeypatch):
        """A TrackedVoyageReranker whose quota answer the test controls."""
        tracker = Mock(estimate_tokens=Mock(return_value=1))
        monkeypatch.setattr("embeddings.get_tracker", lambda: tracker)
        return TrackedVoyageReranker(StubReranker(), "rerank-3"), tracker

    def test_threshold_is_ignored_and_logged_when_quota_is_exceeded(self, monkeypatch, caplog, tracked):
        reranker, tracker = tracked
        tracker.check_rerank_quota.return_value = (False, "over quota")
        engine = _federated_rag(monkeypatch, reranker, rerank_score_threshold=0.1).federated_engine

        with caplog.at_level(logging.WARNING, logger="query_engine"):
            nodes = engine.retrieve("q")

        assert len(nodes) == 20
        assert "ignoring score threshold 0.1" in caplog.text

    def test_threshold_applies_again_once_the_reranker_runs(self, monkeypatch, tracked):
        reranker, tracker = tracked
        tracker.check_rerank_quota.return_value = (False, "over quota")
        engine = _federated_rag(monkeypatch, reranker, rerank_score_threshold=0.1).federated_engine
        engine.retrieve("q")

        tracker.check_rerank_quota.return_value = (True, "")

        assert [n.score for n in engine.retrieve("q")] == [0.9]

    def test_a_failed_rerank_call_falls_back_instead_of_failing_the_query(
        self, monkeypatch, caplog, tracked
    ):
        _, tracker = tracked
        tracker.check_rerank_quota.return_value = (True, "")
        broken = Mock(top_n=5)
        broken.postprocess_nodes.side_effect = RuntimeError("503 from Voyage")
        reranker = TrackedVoyageReranker(broken, "rerank-3")
        engine = _federated_rag(monkeypatch, reranker, rerank_score_threshold=0.1).federated_engine

        with caplog.at_level(logging.ERROR, logger="embeddings"):
            nodes = engine.retrieve("q")

        assert len(nodes) == 5, "the threshold must not remove the unranked nodes"
        assert reranker.fell_back is True
        assert "Rerank failed" in caplog.text
        tracker.record_rerank_usage.assert_not_called()

    @pytest.mark.parametrize("engine_class", [RAGQueryEngine, HybridQueryEngine])
    def test_vault_engines_use_the_guarded_threshold(self, monkeypatch, engine_class):
        monkeypatch.setattr(Settings, "_llm", MockLLM())
        reranker = StubReranker()
        built = {}

        def capture(**kwargs):
            built.update(kwargs)
            return Mock()

        monkeypatch.setattr("query_engine.RetrieverQueryEngine", capture)
        monkeypatch.setattr("query_engine.VectorIndexRetriever", Mock())
        config = SimpleNamespace(
            retrieval=RetrievalConfig(rerank_score_threshold=0.1, use_self_correction=False),
        )

        engine_class(index=Mock(), config=config, reranker=reranker, use_cache=False)

        thresholds = [p for p in built["node_postprocessors"] if isinstance(p, RerankScoreThreshold)]
        assert len(thresholds) == 1
        assert thresholds[0].reranker is reranker

    def test_research_path_has_the_same_guard(self, tracked):
        reranker, tracker = tracked
        tracker.check_rerank_quota.return_value = (False, "over quota")
        retriever = FederatedRetriever(
            _two_scale_sources(), _config(rerank_score_threshold=0.1), reranker=reranker, parallel=False
        )

        assert retriever.retrieve(QUERY)
