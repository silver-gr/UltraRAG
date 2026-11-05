"""SavedItemsRetriever pre-filters the kNN with the same quality rule as x3lixi's own search."""
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from llama_index.core.schema import QueryBundle

from config import SavedItemsConfig, load_config
from saved_items_retriever import SavedItemsRetriever

pytestmark = pytest.mark.unit

QUALITY = "(usefulness_score > 0 OR relevance_score > 0)"
ROW = {
    "text": "body",
    "title": "Title",
    "url": "https://example.com/a",
    "domain": "example.com",
    "source": "web",
    "tags": None,
    "item_id": "item-1",
    "chunk_index": 2,
    "usefulness_score": 50,
    "_distance": 0.25,
}


class StubQuery:
    """Records the LanceDB query chain."""

    def __init__(self):
        self.where_calls = []
        self.selected = None

    def metric(self, _metric):
        return self

    def where(self, expr, prefilter=None):
        self.where_calls.append((expr, prefilter))
        return self

    def limit(self, _limit):
        return self

    def select(self, columns):
        self.selected = columns
        return self

    def to_pandas(self):
        return pd.DataFrame([ROW])


@pytest.fixture
def run_query(monkeypatch):
    """Run _retrieve against a stub table; returns the recorded query and the nodes."""
    embed_model = Mock(get_query_embedding=Mock(return_value=[0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        "saved_items_retriever.Settings", SimpleNamespace(embed_model=embed_model)
    )

    def _run(config, allowed_item_ids=None):
        query = StubQuery()
        retriever = SavedItemsRetriever(config)
        retriever.allowed_item_ids = allowed_item_ids
        retriever._table = SimpleNamespace(search=lambda _vector: query)
        retriever._validated = True
        return query, retriever._retrieve(QueryBundle(query_str="dopamine"))

    return _run


def _config_from_env(monkeypatch, temp_vault, **env):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-3.5-lite")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.delenv("SAVED_ITEMS_MIN_QUALITY_FILTER", raising=False)
    monkeypatch.delenv("SAVED_ITEMS_WHERE", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return load_config().saved_items


class TestQualityPredicate:
    def test_default_applies_quality_filter_as_prefilter(self, monkeypatch, temp_vault, run_query):
        query, _ = run_query(_config_from_env(monkeypatch, temp_vault))

        assert query.where_calls == [(QUALITY, True)]

    def test_filter_can_be_disabled(self, monkeypatch, temp_vault, run_query):
        config = _config_from_env(monkeypatch, temp_vault, SAVED_ITEMS_MIN_QUALITY_FILTER="false")

        query, _ = run_query(config)

        assert query.where_calls == []

    def test_operator_predicate_is_anded_in(self, monkeypatch, temp_vault, run_query):
        config = _config_from_env(
            monkeypatch, temp_vault, SAVED_ITEMS_WHERE="source = 'web' OR source = 'rss'"
        )

        query, _ = run_query(config)

        assert query.where_calls == [(f"{QUALITY} AND (source = 'web' OR source = 'rss')", True)]

    def test_operator_predicate_works_without_quality_filter(
        self, monkeypatch, temp_vault, run_query
    ):
        config = _config_from_env(
            monkeypatch,
            temp_vault,
            SAVED_ITEMS_MIN_QUALITY_FILTER="false",
            SAVED_ITEMS_WHERE="source = 'web'",
        )

        query, _ = run_query(config)

        assert query.where_calls == [("(source = 'web')", True)]

    def test_empty_operator_predicate_is_ignored(self, monkeypatch, temp_vault, run_query):
        query, _ = run_query(_config_from_env(monkeypatch, temp_vault, SAVED_ITEMS_WHERE=""))

        assert query.where_calls == [(QUALITY, True)]


class TestViewerScope:
    def test_scope_prefilters_by_item_id_and_skips_quality_filter(self, run_query):
        query, nodes = run_query(SavedItemsConfig(), allowed_item_ids={"item-2", "item-1"})

        assert query.where_calls == [("item_id IN ('item-1', 'item-2')", True)]
        assert nodes[0].node.metadata["file_path"] == "item-1"

    def test_unscoped_query_keeps_quality_filter(self, run_query):
        query, _ = run_query(SavedItemsConfig(), allowed_item_ids=None)

        assert query.where_calls == [(QUALITY, True)]

    def test_scope_is_anded_with_operator_predicate(self, run_query):
        query, _ = run_query(SavedItemsConfig(where="source = 'web'"), allowed_item_ids={"a"})

        assert query.where_calls == [("item_id IN ('a') AND (source = 'web')", True)]

    def test_item_ids_are_escaped(self, run_query):
        query, _ = run_query(SavedItemsConfig(), allowed_item_ids={"o'brien"})

        assert query.where_calls == [("item_id IN ('o''brien')", True)]

    def test_empty_scope_returns_nothing_without_querying(self, run_query):
        query, nodes = run_query(SavedItemsConfig(), allowed_item_ids=set())

        assert nodes == []
        assert query.selected is None


class TestX3lixiContract:
    def test_select_and_node_metadata_shape_are_unchanged(self, run_query):
        """ultrarag_service.py wraps _retrieve and post-filters on metadata['file_path']."""
        query, nodes = run_query(SavedItemsConfig())

        assert query.selected == [
            "text", "title", "url", "domain", "source", "tags",
            "item_id", "chunk_index", "usefulness_score", "_distance",
        ]
        assert nodes[0].node.metadata["file_path"] == "item-1"
        assert nodes[0].node.node_id == "saved_items:item-1:2"
        assert nodes[0].score == pytest.approx(0.75)

    @pytest.mark.usefixtures("run_query")  # stubs the embedding model
    def test_wrapping_retrieve_still_works_during_the_transition(self):
        """x3lixi's current monkey-patch replaces _retrieve on the instance."""
        retriever = SavedItemsRetriever(SavedItemsConfig())
        retriever._table = SimpleNamespace(search=lambda _vector: StubQuery())
        retriever._validated = True
        original = retriever._retrieve
        retriever._retrieve = lambda bundle: [
            n for n in original(bundle) if n.node.metadata["file_path"] == "someone-else"
        ]

        assert retriever.retrieve(QueryBundle(query_str="dopamine")) == []


class TestValidateLogging:
    def test_predicate_is_logged_once(self, monkeypatch, caplog):
        table = SimpleNamespace(
            schema=[SimpleNamespace(name="vector", type=SimpleNamespace(list_size=3))]
        )
        db = SimpleNamespace(table_names=lambda: ["saved_items"], open_table=lambda _name: table)
        monkeypatch.setitem(sys.modules, "lancedb", SimpleNamespace(connect=lambda _path: db))
        monkeypatch.setattr(
            "saved_items_retriever.Settings",
            SimpleNamespace(embed_model=Mock(get_query_embedding=Mock(return_value=[0.0] * 3))),
        )
        retriever = SavedItemsRetriever(SavedItemsConfig())

        with caplog.at_level(logging.INFO, logger="saved_items_retriever"):
            assert retriever.validate() is True
            assert retriever.validate() is True

        assert [r.getMessage() for r in caplog.records if QUALITY in r.getMessage()] == [
            f"saved_items predicate: {QUALITY}"
        ]
