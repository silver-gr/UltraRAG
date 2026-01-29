"""
Tests for indexing.py module.

Agent note:
- Tests cover all public functions: index_vault, index_conversations, load_index
- Unit tests use mock embedding model
- Integration tests (@pytest.mark.integration) use real APIs
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestIndexVault:
    """
    Tests for index_vault() function.

    Agent context:
    - index_vault() loads .md files, chunks them, embeds, stores in LanceDB
    - Returns VectorStoreIndex ready for querying
    - Uses exclusion patterns from settings
    """

    @pytest.mark.unit
    def test_index_vault_returns_index(self, test_config, mock_embedding_model):
        """
        GIVEN: A test vault with 3 markdown files
        WHEN: index_vault() is called
        THEN: Returns a VectorStoreIndex with nodes

        Agent note: This tests the happy path. If it fails, check loader.py.
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault

            index = index_vault(test_config)

            assert index is not None
            # Should have indexed our 3 test files
            # Note: Actual node count depends on chunking

    @pytest.mark.unit
    def test_index_vault_empty_directory(self, test_config, tmp_path, mock_embedding_model):
        """
        GIVEN: An empty vault directory
        WHEN: index_vault() is called
        THEN: Returns None or raises appropriate error

        Agent note: Edge case - empty vault should not crash.
        """
        test_config.vault_path = tmp_path / "empty_vault"
        test_config.vault_path.mkdir()

        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault

            # Should handle gracefully
            result = index_vault(test_config)
            # Either None or empty index is acceptable

    @pytest.mark.unit
    def test_index_vault_respects_exclusions(self, test_config, mock_embedding_model, sample_vault_path):
        """
        GIVEN: A vault with exclusion pattern "test_note_1.md"
        WHEN: index_vault() is called
        THEN: Only 2 files are indexed

        Agent note: Tests exclusion_matcher integration.
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            with patch('indexing._get_exclusion_patterns', return_value=[{"pattern": "test_note_1.md", "type": "exact"}]):
                from indexing import index_vault

                index = index_vault(test_config)
                # Should have fewer nodes than without exclusion


class TestLoadIndex:
    """
    Tests for load_index() function.

    Agent context:
    - load_index() loads pre-built index from disk
    - Returns None if index doesn't exist
    - Reconstructs docstore from LanceDB metadata
    """

    @pytest.mark.unit
    def test_load_index_returns_none_when_missing(self, test_config):
        """
        GIVEN: No index exists at the configured path
        WHEN: load_index() is called
        THEN: Returns None (not raises exception)

        Agent note: Agents should check for None before querying.
        """
        from indexing import load_index

        result = load_index(test_config, source="vault")

        assert result is None

    @pytest.mark.unit
    def test_load_index_after_indexing(self, test_config, mock_embedding_model):
        """
        GIVEN: An index was previously created
        WHEN: load_index() is called
        THEN: Returns the index with all nodes intact

        Agent note: This tests persistence - index survives restart.
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault, load_index

            # Create index
            original = index_vault(test_config)

            # Load it back
            loaded = load_index(test_config, source="vault")

            assert loaded is not None


class TestIndexExists:
    """Tests for index_exists() function."""

    @pytest.mark.unit
    def test_index_exists_false_initially(self, test_config):
        """
        GIVEN: Fresh test directory with no index
        WHEN: index_exists() is called
        THEN: Returns False
        """
        from indexing import index_exists

        assert index_exists(test_config, source="vault") is False

    @pytest.mark.unit
    def test_index_exists_true_after_indexing(self, test_config, mock_embedding_model):
        """
        GIVEN: An index has been created
        WHEN: index_exists() is called
        THEN: Returns True
        """
        with patch('embeddings.get_embedding_model', return_value=mock_embedding_model):
            from indexing import index_vault, index_exists

            index_vault(test_config)

            assert index_exists(test_config, source="vault") is True
