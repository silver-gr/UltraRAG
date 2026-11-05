"""Federated merge: sources with incomparable native scores share the candidate list by rank."""
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode

from config import RetrievalConfig
from federated_query import FederatedQueryEngine, FederatedRetriever, IndexSource
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
