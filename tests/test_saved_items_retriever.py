"""SavedItemsRetriever pre-filters the kNN with the same quality rule as x3lixi's own search."""
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import lancedb
import pandas as pd
import pytest
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode

import content_research_engine
from config import EmbeddingConfig, RetrievalConfig, SavedItemsConfig, load_config
from federated_query import FederatedQueryEngine, FederatedRetriever, IndexSource
from main import UltraRAG, _source_excerpt
from query_engine import SourceNumberingPostprocessor
from saved_items_retriever import SavedItemsRetriever
from self_correction import SelfCorrectingRetriever

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
    monkeypatch.delenv("SAVED_ITEMS_STRICT_MODEL_CHECK", raising=False)
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
    def test_scope_prefilters_by_item_id_and_skips_quality_filter(self, run_query, caplog):
        with caplog.at_level(logging.INFO, logger="saved_items_retriever"):
            query, nodes = run_query(SavedItemsConfig(), allowed_item_ids={"item-2", "item-1"})

        assert query.where_calls == [("item_id IN ('item-1', 'item-2')", True)]
        assert "scoped to 2 items" in caplog.text
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
        db = SimpleNamespace(
            list_tables=lambda: SimpleNamespace(tables=["saved_items"]),
            open_table=lambda _name: table,
        )
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


BODY = "the chunk body"
HEADER = "[web · example.com · post] Title\nTopics: habits"
TABLE_ROW = {
    "vector": [1.0, 0.0, 0.0],
    "text": BODY,
    "title": "Title",
    "url": "https://example.com/a",
    "domain": "example.com",
    "source": "web",
    "tags": "",
    "item_id": "item-1",
    "chunk_index": 0,
    "usefulness_score": 50,
    "relevance_score": 50,
}


@pytest.fixture
def make_retriever(monkeypatch, tmp_path):
    """A retriever over a real LanceDB table holding the given row overrides."""
    embed_model = Mock(get_query_embedding=Mock(return_value=[1.0, 0.0, 0.0]))
    monkeypatch.setattr("saved_items_retriever.Settings", SimpleNamespace(embed_model=embed_model))

    def _make(rows, embedding_model=None, **config):
        lancedb.connect(str(tmp_path)).create_table(
            "saved_items", data=[{**TABLE_ROW, **row} for row in rows]
        )
        return SavedItemsRetriever(
            SavedItemsConfig(shared_lancedb_path=tmp_path, **config),
            embedding_model=embedding_model,
        )

    return _make


def _retrieve(retriever):
    return retriever.retrieve(QueryBundle(query_str="dopamine"))


class TestTableLookup:
    @staticmethod
    def _crowd_store(path):
        db = lancedb.connect(str(path))
        for index in range(12):
            db.create_table(f"a{index:02d}", data=[{"x": index}])

    def test_validate_finds_a_table_past_the_first_listing_page(self, make_retriever, tmp_path):
        self._crowd_store(tmp_path)

        assert make_retriever([{}]).validate() is True

    def test_research_search_finds_a_table_past_the_first_listing_page(
        self, monkeypatch, tmp_path
    ):
        self._crowd_store(tmp_path)
        lancedb.connect(str(tmp_path)).create_table("saved_items", data=[TABLE_ROW])
        monkeypatch.setattr(content_research_engine, "_get_lancedb_path", lambda _source: tmp_path)

        results = content_research_engine._search_source(
            "saved_items", [1.0, 0.0, 0.0], weight=1.0, top_k=5, threshold=0.0
        )

        assert len(results) == 1

    def test_research_tools_read_the_configured_table_name(self):
        probe = (
            "import content_research_engine as c, obsession_radar as o\n"
            "print(c.TABLE_NAMES['saved_items'], o.TABLE_NAMES['saved_items'])\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "SAVED_ITEMS_TABLE_NAME": "saved_items_v2"},
        )

        assert result.stdout.split() == ["saved_items_v2", "saved_items_v2"]


class TestSelfCorrectionSeesTheChunk:
    LONG_HEADER = "[web] Title\nSummary: " + "item level summary " * 40

    def _retriever(self, llm):
        node = NodeWithScore(
            node=TextNode(
                text=f"{self.LONG_HEADER}\n---\n{BODY}",
                metadata={"title": "Title", "chunk_body": BODY},
            ),
            score=1.0,
        )
        base = Mock(spec=BaseRetriever)
        base.retrieve.return_value = [node]
        return SelfCorrectingRetriever(base, llm, max_retries=1), node

    def test_grading_reads_the_body_not_the_header(self):
        llm = Mock()
        llm.complete.return_value = SimpleNamespace(text="CORRECT")
        retriever, node = self._retriever(llm)

        retriever.grade_relevance("dopamine", [node])

        prompt = llm.complete.call_args.args[0]
        assert BODY in prompt
        assert "item level summary" not in prompt

    def test_refinement_sample_reads_the_body_not_the_header(self):
        llm = Mock()
        llm.complete.return_value = SimpleNamespace(text="INCORRECT")
        retriever, _node = self._retriever(llm)
        refine = Mock(return_value="better query")
        retriever.refine_query = refine

        retriever.retrieve(QueryBundle(query_str="dopamine"))

        assert refine.call_args.args[1] == f"Title: {BODY}"

    def test_node_without_a_chunk_body_still_grades_on_its_text(self):
        llm = Mock()
        llm.complete.return_value = SimpleNamespace(text="CORRECT")
        retriever, _node = self._retriever(llm)
        plain = NodeWithScore(node=TextNode(text=BODY, metadata={"title": "Title"}), score=1.0)

        retriever.grade_relevance("dopamine", [plain])

        assert BODY in llm.complete.call_args.args[0]


class TestChunkContext:
    def test_table_without_new_columns_still_returns_plain_nodes(self, make_retriever):
        retriever = make_retriever([{}])

        assert retriever.validate() is True
        nodes = _retrieve(retriever)

        assert [n.node.text for n in nodes] == [BODY]
        assert "chunk_body" not in nodes[0].node.metadata

    def test_context_is_prepended_and_body_kept_for_display(self, make_retriever):
        nodes = _retrieve(make_retriever([{"context": HEADER}]))

        assert nodes[0].node.text == f"{HEADER}\n---\n{BODY}"
        assert nodes[0].node.metadata["chunk_body"] == BODY

    def test_empty_or_null_context_leaves_text_unchanged(self, make_retriever):
        rows = [
            {"item_id": "empty", "context": ""},
            {"item_id": "null", "context": None},
            {"item_id": "set", "context": HEADER},
        ]

        nodes = {n.node.metadata["file_path"]: n.node for n in _retrieve(make_retriever(rows))}

        for item_id in ("empty", "null"):
            assert nodes[item_id].text == BODY
            assert "chunk_body" not in nodes[item_id].metadata
        assert nodes["set"].text.startswith(HEADER)

    @pytest.mark.parametrize("mode", [MetadataMode.LLM, MetadataMode.EMBED])
    def test_chunk_is_not_sent_twice(self, make_retriever, mode):
        node = _retrieve(make_retriever([{"context": HEADER}]))[0].node

        content = node.get_content(metadata_mode=mode)

        assert content.count(BODY) == 1
        assert HEADER in content
        assert "chunk_body" not in content

    def test_scope_and_item_identity_hold_on_the_new_schema(self, make_retriever):
        rows = [
            {"item_id": "mine", "context": HEADER},
            {"item_id": "theirs", "context": HEADER, "vector": [0.9, 0.1, 0.0]},
        ]
        retriever = make_retriever(rows)
        retriever.allowed_item_ids = {"mine"}

        nodes = _retrieve(retriever)

        assert [n.node.metadata["file_path"] for n in nodes] == ["mine"]


def _headed_node(item_id="item-1", header=HEADER, body=BODY):
    return TextNode(
        id_=f"saved_items:{item_id}:0",
        text=f"{header}\n---\n{body}",
        metadata={"file_path": item_id, "chunk_body": body, "file_name": "f", "title": "T"},
    )


class TestSourceExcerpt:
    def test_prefers_chunk_body(self):
        assert _source_excerpt(_headed_node()) == BODY

    def test_falls_back_to_original_text_then_text(self):
        assert _source_excerpt(TextNode(text="[1] x", metadata={"_original_text": "x"})) == "x"
        assert _source_excerpt(TextNode(text="plain")) == "plain"

    def test_accepts_node_with_score_and_truncates(self):
        node = NodeWithScore(node=TextNode(text="a" * 1501), score=1.0)

        assert _source_excerpt(node) == "a" * 1500 + "..."

    def test_source_numbering_keeps_header_for_llm_and_body_for_display(self):
        node = NodeWithScore(node=_headed_node(), score=1.0)

        SourceNumberingPostprocessor().postprocess_nodes([node])

        assert HEADER in node.node.text
        assert _source_excerpt(node) == BODY

    def test_format_sources_shows_body(self):
        sources = UltraRAG._format_sources(None, [NodeWithScore(node=_headed_node(), score=1.0)])

        assert sources[0]["excerpt"] == BODY


class _StubRetriever(BaseRetriever):
    def __init__(self, nodes):
        self._stub_nodes = nodes
        super().__init__()

    def _retrieve(self, query_bundle):
        return self._stub_nodes


class TestQueryFederated:
    def test_llm_sees_the_header_and_sources_show_the_body(self, monkeypatch):
        # MockLLM echoes the prompt, so the answer shows what reached the synthesiser.
        monkeypatch.setattr(Settings, "_llm", MockLLM())
        source = IndexSource(
            name="saved_items",
            source_type="saved_items",
            weight=1.0,
            custom_retriever=_StubRetriever([NodeWithScore(node=_headed_node(), score=0.9)]),
        )
        rag = object.__new__(UltraRAG)
        rag.config = SimpleNamespace(retrieval=RetrievalConfig(use_self_correction=False))
        rag.federated_engine = FederatedQueryEngine([source], rag.config, use_cache=False)

        result = rag.query_federated("q")

        assert HEADER in result["answer"]
        assert result["answer"].count(BODY) == 1
        assert [s["excerpt"] for s in result["sources"]] == [BODY]


class TestFederatedDedupe:
    def test_identical_bodies_in_two_items_dedupe_despite_different_headers(self):
        nodes = [
            NodeWithScore(node=_headed_node("item-1", header="[a] One"), score=0.9),
            NodeWithScore(node=_headed_node("item-2", header="[b] Two"), score=0.8),
        ]
        source = IndexSource(
            name="saved_items",
            source_type="saved_items",
            weight=1.0,
            custom_retriever=_StubRetriever(nodes),
        )
        config = SimpleNamespace(retrieval=RetrievalConfig())

        merged = FederatedRetriever([source], config, parallel=False).retrieve(
            QueryBundle(query_str="dopamine")
        )

        assert [n.node.metadata["file_path"] for n in merged] == ["item-1"]

    def test_same_item_dedupe_key_ignores_the_header(self):
        first = NodeWithScore(node=_headed_node(header="[a] One"), score=0.9)
        second = NodeWithScore(node=_headed_node(header="[a] Refreshed"), score=0.8)

        assert FederatedRetriever._content_dedupe_key(first) == (
            FederatedRetriever._content_dedupe_key(second)
        )


class TestEmbeddingModelCheck:
    @pytest.mark.parametrize(
        ("table_model", "configured"),
        [
            ("voyage-3.5-lite", "voyage-3.5-lite"),
            ("voyage-4-lite", "voyage-4-large"),
            ("voyage-4", "voyage-4-lite"),
            ("", "voyage-4-large"),
        ],
    )
    def test_compatible_model_is_silent(self, make_retriever, caplog, table_model, configured):
        retriever = make_retriever(
            [{"embedding_model": table_model}], embedding_model=configured, strict_model_check=True
        )

        with caplog.at_level(logging.WARNING, logger="saved_items_retriever"):
            assert retriever.validate() is True

        assert "embedding model" not in caplog.text

    def test_mismatch_warns_but_keeps_the_source(self, make_retriever, caplog):
        retriever = make_retriever(
            [{"embedding_model": "voyage-3.5-lite"}], embedding_model="voyage-4-large"
        )

        with caplog.at_level(logging.WARNING, logger="saved_items_retriever"):
            assert retriever.validate() is True

        assert "voyage-3.5-lite" in caplog.text
        assert [n.node.text for n in _retrieve(retriever)] == [BODY]

    def test_strict_mismatch_disables_the_source(self, make_retriever, caplog):
        retriever = make_retriever(
            [{"embedding_model": "voyage-3.5-lite"}],
            embedding_model="voyage-4-large",
            strict_model_check=True,
        )

        with caplog.at_level(logging.WARNING, logger="saved_items_retriever"):
            assert retriever.validate() is False

        assert "voyage-3.5-lite" in caplog.text
        assert _retrieve(retriever) == []

    def test_strict_mismatch_disables_even_when_the_dim_check_cannot_run(
        self, make_retriever, monkeypatch
    ):
        retriever = make_retriever(
            [{"embedding_model": "voyage-3.5-lite"}],
            embedding_model="voyage-4-large",
            strict_model_check=True,
        )
        broken = Mock(get_query_embedding=Mock(side_effect=RuntimeError("no network")))
        monkeypatch.setattr("saved_items_retriever.Settings", SimpleNamespace(embed_model=broken))

        assert retriever.validate() is False

    def test_table_without_the_column_is_fine_even_when_strict(self, make_retriever, caplog):
        retriever = make_retriever([{}], embedding_model="voyage-4-large", strict_model_check=True)

        with caplog.at_level(logging.WARNING, logger="saved_items_retriever"):
            assert retriever.validate() is True

        assert "embedding model" not in caplog.text

    def test_strict_setting_comes_from_env_and_defaults_off(self, monkeypatch, temp_vault):
        assert _config_from_env(monkeypatch, temp_vault).strict_model_check is False
        config = _config_from_env(monkeypatch, temp_vault, SAVED_ITEMS_STRICT_MODEL_CHECK="true")
        assert config.strict_model_check is True

    def test_pipeline_passes_the_configured_model(self, monkeypatch):
        built = {}

        def fake_retriever(config, top_k, embedding_model, max_chunks_per_item):
            built["embedding_model"] = embedding_model
            built["max_chunks_per_item"] = max_chunks_per_item
            return Mock(validate=Mock(return_value=True))

        monkeypatch.setattr("saved_items_retriever.SavedItemsRetriever", fake_retriever)
        rag = SimpleNamespace(
            config=SimpleNamespace(
                saved_items=SavedItemsConfig(enabled=True),
                embedding=EmbeddingConfig(model="voyage-4-lite"),
                retrieval=RetrievalConfig(),
            ),
            index=None,
        )

        UltraRAG._auto_load_saved_items(rag)

        assert built["embedding_model"] == "voyage-4-lite"
        assert built["max_chunks_per_item"] == RetrievalConfig().federated_max_chunks_per_document


class TestLongItemCrowding:
    """Chunks of one item share a context header, so a long item can fill the whole window."""

    @staticmethod
    def _retrieve(monkeypatch, rows, top_k=4, max_per_item=2):
        limits = []

        class Query(StubQuery):
            def limit(self, value):
                limits.append(value)
                return self

            def to_pandas(self):
                return pd.DataFrame(rows[: limits[-1]])

        table = Mock(search=Mock(return_value=Query()))
        embed_model = Mock(get_query_embedding=Mock(return_value=[0.0, 0.0, 0.0]))
        monkeypatch.setattr(
            "saved_items_retriever.Settings", SimpleNamespace(embed_model=embed_model)
        )
        retriever = SavedItemsRetriever(
            SavedItemsConfig(enabled=True), top_k=top_k, max_chunks_per_item=max_per_item
        )
        retriever._table = table
        retriever._validated = True
        return retriever._retrieve(QueryBundle("q")), limits

    def test_caps_chunks_per_item_and_fills_top_k_from_other_items(self, monkeypatch):
        rows = [{**ROW, "item_id": "long", "chunk_index": i} for i in range(9)]
        rows += [{**ROW, "item_id": f"other{i}", "chunk_index": 0} for i in range(5)]

        nodes, limits = self._retrieve(monkeypatch, rows)

        assert limits[-1] > 4, "must fetch beyond top_k to reach other items"
        assert [n.node.metadata["document_id"] for n in nodes] == [
            "long",
            "long",
            "other0",
            "other1",
        ]

    def test_keeps_rank_order_and_returns_all_when_few_items(self, monkeypatch):
        rows = [{**ROW, "item_id": "a", "chunk_index": i} for i in range(3)]
        rows.append({**ROW, "item_id": "b", "chunk_index": 0})

        nodes, _ = self._retrieve(monkeypatch, rows)

        assert [n.node.node_id for n in nodes] == [
            "saved_items:a:0",
            "saved_items:a:1",
            "saved_items:b:0",
        ]
