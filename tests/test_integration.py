"""
Integration tests for full UltraRAG workflow.

Agent note: These tests verify the new modules work together.
Run with: pytest tests/test_integration.py -v
"""

import pytest
from unittest.mock import patch


@pytest.mark.integration
class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_index_then_query(self, test_config, mock_embedding_model):
        """
        GIVEN: A test vault
        WHEN: index_vault() then query() is called
        THEN: Full pipeline works end-to-end
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault
            from retrieval import query
            from models import QueryResult

            # Index
            index = index_vault(test_config)
            assert index is not None, "index_vault() should return an index"

            # Query (simple mode to avoid LLM dependency)
            with patch('retrieval.get_llm') as mock_llm:
                mock_response = type('Response', (), {
                    'response': 'RAG combines retrieval with generation.',
                    'source_nodes': [],
                    'metadata': {}
                })()
                mock_llm.return_value.complete.return_value = mock_response

                # Mock the query engine
                with patch.object(index, 'as_query_engine') as mock_engine:
                    mock_engine.return_value.query.return_value = mock_response

                    result = query("What is RAG?", index, test_config, mode="simple")

                    assert isinstance(result, QueryResult)
                    assert len(result.answer) > 0

    def test_load_and_query(self, test_config, mock_embedding_model):
        """
        GIVEN: A previously indexed vault
        WHEN: load_index() then query() is called
        THEN: Can query without re-indexing
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault, load_index
            from retrieval import query
            from models import QueryResult

            # First index
            index_vault(test_config)

            # Then load
            loaded = load_index(test_config, source="vault")
            assert loaded is not None, "load_index() should return the index"

            # Then query
            with patch('retrieval.get_llm') as mock_llm:
                mock_response = type('Response', (), {
                    'response': 'Test response',
                    'source_nodes': [],
                    'metadata': {}
                })()

                with patch.object(loaded, 'as_query_engine') as mock_engine:
                    mock_engine.return_value.query.return_value = mock_response

                    result = query("test", loaded, test_config, mode="simple")

                    assert isinstance(result, QueryResult)

    def test_index_exists_workflow(self, test_config, mock_embedding_model):
        """
        GIVEN: No index exists
        WHEN: index_exists() is called, then index_vault(), then index_exists() again
        THEN: Returns False, then True
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault, index_exists

            # Initially no index
            assert index_exists(test_config, source="vault") is False

            # Create index
            index_vault(test_config)

            # Now exists
            assert index_exists(test_config, source="vault") is True


@pytest.mark.integration
class TestModelsIntegration:
    """Test that models.py types work with other modules."""

    def test_query_result_from_retrieval(self, mock_llm):
        """
        GIVEN: A mock index
        WHEN: retrieval.query() is called
        THEN: Returns models.QueryResult type
        """
        from retrieval import query
        from models import QueryResult, IndexNotFoundError
        from unittest.mock import MagicMock

        with patch('retrieval.get_llm', return_value=mock_llm):
            mock_index = MagicMock()
            mock_response = MagicMock()
            mock_response.response = "Test answer"
            mock_response.source_nodes = []
            mock_response.metadata = {}
            mock_index.as_query_engine.return_value.query.return_value = mock_response

            result = query("test", mock_index, mode="simple")

            # Verify it's the correct type from models.py
            assert isinstance(result, QueryResult)
            assert hasattr(result, 'answer')
            assert hasattr(result, 'sources')
            assert hasattr(result, 'tokens_used')
            assert hasattr(result, 'exec_time')

    def test_index_not_found_error(self):
        """
        GIVEN: None passed as index
        WHEN: retrieval.query() is called
        THEN: Raises models.IndexNotFoundError with fix hint
        """
        from retrieval import query
        from models import IndexNotFoundError

        with pytest.raises(IndexNotFoundError) as exc_info:
            query("test", None)

        # Verify it's our custom exception with fix hint
        error = exc_info.value
        assert hasattr(error, 'fix_hint')
        assert 'fix' in str(error).lower()


@pytest.mark.integration
class TestBackwardCompatibility:
    """Verify existing imports still work."""

    def test_old_imports_still_work(self):
        """
        GIVEN: Old import patterns from main.py
        WHEN: Imports are executed
        THEN: No import errors
        """
        # These are the imports from main.py that should still work
        from config import load_config, RAGConfig
        from loader import ObsidianLoader
        from embeddings import get_embedding_model
        from chunking import ObsidianChunker
        from vector_store import get_vector_store, create_vector_index, load_vector_index, index_exists
        from query_engine import RAGQueryEngine, HybridQueryEngine

        # Just verify imports work
        assert RAGConfig is not None
        assert ObsidianLoader is not None
        assert get_embedding_model is not None

    def test_new_imports_work(self):
        """
        GIVEN: New module import patterns
        WHEN: Imports are executed
        THEN: No import errors
        """
        from indexing import index_vault, index_conversations, load_index, index_exists
        from retrieval import query, federated_query, research
        from models import (
            QueryResult, ResearchResult, SourceNode,
            IndexNotFoundError, ConfigurationError, LLMRateLimitError,
            IndexSource, SourceType
        )

        # Verify imports work
        assert index_vault is not None
        assert query is not None
        assert QueryResult is not None
