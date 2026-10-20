"""
Tests for retrieval.py module.

Agent note:
- query() is the main entry point for all retrieval
- federated_query() queries multiple indexes
- research() does iterative retrieval with gap analysis
"""

import pytest
from unittest.mock import patch, MagicMock


class TestQuery:
    """
    Tests for query() function.

    Agent context:
    - query() takes a query string and index, returns QueryResult
    - Modes: simple (vector), hybrid (vector+BM25), research (iterative)
    - Uses caching by default
    """

    @pytest.mark.unit
    def test_query_returns_result(self, mock_llm):
        """
        GIVEN: A loaded index and query
        WHEN: query() is called
        THEN: Returns QueryResult with answer and sources
        """
        from models import QueryResult

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Test answer about RAG"
        mock_response.source_nodes = []
        mock_response.metadata = {}
        mock_engine.query.return_value = mock_response

        with patch('retrieval._build_query_engine', return_value=mock_engine):
            from retrieval import query

            mock_index = MagicMock()
            result = query("What is RAG?", mock_index, mode="simple")

            assert isinstance(result, QueryResult)
            assert len(result.answer) > 0

    @pytest.mark.unit
    def test_query_with_none_index_raises(self):
        """
        GIVEN: No index (None)
        WHEN: query() is called
        THEN: Raises IndexNotFoundError with fix hint
        """
        from retrieval import query
        from models import IndexNotFoundError

        with pytest.raises(IndexNotFoundError) as exc_info:
            query("test", None)

        assert "fix" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_query_simple_mode(self, mock_llm):
        """
        GIVEN: An index and mode="simple"
        WHEN: query() is called
        THEN: Uses simple vector retrieval (not hybrid)
        """
        from models import QueryResult

        mock_engine = MagicMock()
        mock_response = MagicMock()
        mock_response.response = "Simple response"
        mock_response.source_nodes = []
        mock_response.metadata = {}
        mock_engine.query.return_value = mock_response

        with patch('retrieval._build_query_engine', return_value=mock_engine):
            from retrieval import query

            mock_index = MagicMock()
            result = query("test", mock_index, mode="simple")

            assert result.mode == "simple"


class TestFederatedQuery:
    """Tests for federated_query() function."""

    @pytest.mark.unit
    def test_federated_query_single_source(self, mock_llm):
        """
        GIVEN: A single index source
        WHEN: federated_query() is called
        THEN: Works like regular query
        """
        from models import QueryResult, IndexSource, SourceType

        with patch('retrieval.get_llm', return_value=mock_llm):
            # Mock FederatedQueryEngine since federated_query now delegates to it
            mock_response = MagicMock()
            mock_response.response = "test answer"
            mock_response.source_nodes = []
            mock_response.metadata = {}

            with patch('federated_query.FederatedQueryEngine') as MockEngine:
                MockEngine.return_value.query.return_value = mock_response

                from retrieval import federated_query

                mock_index = MagicMock()
                sources = {
                    "vault": IndexSource(
                        name="vault",
                        index=mock_index,
                        source_type=SourceType.VAULT,
                    )
                }

                result = federated_query("test", sources)

                assert isinstance(result, QueryResult)

    @pytest.mark.unit
    def test_federated_query_empty_sources_raises(self):
        """
        GIVEN: Empty sources dict
        WHEN: federated_query() is called
        THEN: Raises IndexNotFoundError
        """
        from retrieval import federated_query
        from models import IndexNotFoundError

        with pytest.raises(IndexNotFoundError):
            federated_query("test", {})


class TestResearch:
    """
    Tests for research() function (iterative retrieval).

    Agent context:
    - research() performs multiple iterations with gap analysis
    - Returns ResearchResult with iterations and confidence
    - exhaustive=True forces all iterations
    """

    @pytest.mark.unit
    def test_research_returns_research_result(self, mock_llm):
        """
        GIVEN: An index and research query
        WHEN: research() is called
        THEN: Returns ResearchResult with iterations data
        """
        from models import ResearchResult

        with patch('retrieval.get_llm', return_value=mock_llm):
            with patch('retrieval._do_research') as mock_research:
                mock_research.return_value = ResearchResult(
                    answer="Test research answer",
                    sources=[],
                    tokens_used=100,
                    exec_time=1.0,
                    iterations=2,
                    confidence=0.9,
                )

                from retrieval import research

                result = research("find all about RAG", MagicMock())

                assert isinstance(result, ResearchResult)
                assert result.iterations > 0

    @pytest.mark.unit
    def test_research_with_none_index_raises(self):
        """
        GIVEN: No index (None)
        WHEN: research() is called
        THEN: Raises IndexNotFoundError
        """
        from retrieval import research
        from models import IndexNotFoundError

        with pytest.raises(IndexNotFoundError):
            research("test", None)


class TestEngineWiring:
    """
    The stateless API must wire the reranker and BM25 nodes like main.UltraRAG.

    Without a reranker the engines apply a cosine similarity cutoff to rank-fused
    scores, which removed every result ("Empty Response" from the agent CLI).
    """

    @pytest.mark.unit
    def test_query_engines_get_reranker_and_docstore_nodes(self, test_config):
        from retrieval import _build_query_engine

        index = MagicMock()
        index.docstore.docs = {"a": "node-a", "b": "node-b"}
        reranker = object()

        with patch('embeddings.get_reranker', return_value=reranker), \
             patch('query_engine.HybridQueryEngine') as hybrid, \
             patch('query_engine.RAGQueryEngine') as simple:
            _build_query_engine(index, test_config, "hybrid")
            _build_query_engine(index, test_config, "simple")

        assert hybrid.call_args.kwargs["reranker"] is reranker
        assert hybrid.call_args.kwargs["nodes"] == ["node-a", "node-b"]
        assert simple.call_args.kwargs["reranker"] is reranker

    @pytest.mark.unit
    def test_federated_query_gets_reranker_and_docstore_nodes(self, test_config):
        from models import IndexSource, SourceType
        from retrieval import federated_query

        index = MagicMock()
        index.docstore.docs = {"a": "node-a"}
        reranker = object()
        sources = {"vault": IndexSource(name="vault", index=index, source_type=SourceType.VAULT)}

        with patch('embeddings.get_reranker', return_value=reranker), \
             patch('federated_query.FederatedQueryEngine') as engine:
            engine.return_value.query.return_value = MagicMock(
                response="answer", source_nodes=[], metadata={}
            )
            federated_query("wiring test", sources, test_config)

        kwargs = engine.call_args.kwargs
        assert kwargs["reranker"] is reranker
        assert kwargs["sources"][0].nodes == ["node-a"]

    @pytest.mark.unit
    def test_research_retriever_gets_reranker_and_docstore_nodes(self, test_config):
        from models import IndexSource, SourceType
        from retrieval import _do_research

        index = MagicMock()
        index.docstore.docs = {"a": "node-a"}
        reranker = object()
        sources = {"vault": IndexSource(name="vault", index=index, source_type=SourceType.VAULT)}

        with patch('embeddings.get_reranker', return_value=reranker), \
             patch('retrieval.get_llm'), \
             patch('federated_query.FederatedRetriever') as retriever, \
             patch('research_mode.ResearchRetriever') as research_retriever:
            research_retriever.return_value.research.side_effect = RuntimeError("stop after wiring")
            _do_research("wiring test", sources, test_config, False)

        kwargs = retriever.call_args.kwargs
        assert kwargs["reranker"] is reranker
        assert kwargs["sources"][0].nodes == ["node-a"]
