"""Tests for configuration validation."""
import pytest
import os
from pathlib import Path
from pydantic import ValidationError
from config import (
    EmbeddingConfig,
    VectorDBConfig,
    GraphDBConfig,
    RetrievalConfig,
    LLMConfig,
    RAGConfig,
    load_config
)


class TestEmbeddingConfig:
    """Test EmbeddingConfig validation."""

    def test_valid_embedding_config(self):
        """Test creating a valid embedding config."""
        config = EmbeddingConfig(
            model="voyage-3-large",
            dimension=1024,
            chunk_size=512,
            chunk_overlap=75
        )
        assert config.model == "voyage-3-large"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 75

    def test_chunk_overlap_validation_success(self):
        """Test that valid chunk_overlap passes validation."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=100)
        assert config.chunk_overlap == 100

    def test_chunk_overlap_validation_failure(self):
        """Test that chunk_overlap >= chunk_size raises error."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(chunk_size=512, chunk_overlap=512)

        error_msg = str(exc_info.value)
        assert "chunk_overlap" in error_msg
        assert "must be less than chunk_size" in error_msg

    def test_chunk_overlap_greater_than_chunk_size(self):
        """Test that chunk_overlap > chunk_size raises error."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(chunk_size=100, chunk_overlap=200)

        error_msg = str(exc_info.value)
        assert "must be less than chunk_size" in error_msg

    def test_chunk_overlap_equal_chunk_size(self):
        """Test that chunk_overlap == chunk_size raises error."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(chunk_size=512, chunk_overlap=512)

        error_msg = str(exc_info.value)
        assert "must be less than chunk_size" in error_msg

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.model == "voyage-3.5-lite"
        assert config.dimension == 1024
        assert config.chunk_size == 512
        assert config.chunk_overlap == 75


class TestLLMConfig:
    """Test LLMConfig validation."""

    def test_valid_llm_config(self):
        """Test creating a valid LLM config."""
        config = LLMConfig(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            max_tokens=8192
        )
        assert config.temperature == 0.1

    def test_temperature_validation_success(self):
        """Test that valid temperatures pass validation."""
        # Test boundary values
        config_min = LLMConfig(temperature=0.0)
        assert config_min.temperature == 0.0

        config_max = LLMConfig(temperature=2.0)
        assert config_max.temperature == 2.0

        config_mid = LLMConfig(temperature=1.0)
        assert config_mid.temperature == 1.0

    def test_temperature_validation_below_zero(self):
        """Test that temperature < 0 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(temperature=-0.1)

        error_msg = str(exc_info.value)
        assert "temperature must be between 0 and 2" in error_msg

    def test_temperature_validation_above_two(self):
        """Test that temperature > 2 raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(temperature=2.5)

        error_msg = str(exc_info.value)
        assert "temperature must be between 0 and 2" in error_msg

    def test_temperature_validation_way_out_of_range(self):
        """Test temperature far outside valid range."""
        with pytest.raises(ValidationError):
            LLMConfig(temperature=10.0)

        with pytest.raises(ValidationError):
            LLMConfig(temperature=-5.0)

    def test_default_llm_config(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert config.model == "gemini-2.0-flash-exp"
        assert config.temperature == 0.1
        assert config.max_tokens == 8192
        assert config.enable_thinking_mode is True


class TestVectorDBConfig:
    """Test VectorDBConfig validation."""

    def test_valid_vector_db_config(self):
        """Test creating a valid vector DB config."""
        config = VectorDBConfig(
            db_type="lancedb",
            lancedb_path=Path("./data/lancedb")
        )
        assert config.db_type == "lancedb"

    def test_default_vector_db_config(self):
        """Test default vector DB configuration."""
        config = VectorDBConfig()
        assert config.db_type == "lancedb"
        assert config.lancedb_path == Path("./data/lancedb")
        assert config.qdrant_host == "localhost"
        assert config.qdrant_port == 6333


class TestGraphDBConfig:
    """Test GraphDBConfig validation."""

    def test_valid_graph_db_config(self):
        """Test creating a valid graph DB config."""
        config = GraphDBConfig(
            enabled=True,
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password"
        )
        assert config.enabled is True

    def test_default_graph_db_config(self):
        """Test default graph DB configuration."""
        config = GraphDBConfig()
        assert config.enabled is False
        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.neo4j_username == "neo4j"


class TestRetrievalConfig:
    """Test RetrievalConfig validation."""

    def test_valid_retrieval_config(self):
        """Test creating a valid retrieval config."""
        config = RetrievalConfig(
            top_k=75,
            rerank_top_n=10,
            similarity_threshold=0.7
        )
        assert config.top_k == 75
        assert config.rerank_top_n == 10

    def test_default_retrieval_config(self):
        """Test default retrieval configuration."""
        config = RetrievalConfig()
        assert config.top_k == 75
        assert config.rerank_top_n == 10
        assert config.reranker_model == "rerank-2.5"  # No "voyage-" prefix!
        assert config.enable_hybrid_search is True
        assert config.similarity_threshold == 0.3  # Lowered from 0.7


class TestRAGConfig:
    """Test RAGConfig validation."""

    def test_valid_rag_config(self, temp_vault):
        """Test creating a valid RAG config."""
        config = RAGConfig(
            vault_path=temp_vault,
            voyage_api_key="test-key",
            google_api_key="test-key"
        )
        assert config.vault_path == temp_vault

    def test_vault_path_validation_nonexistent(self):
        """Test that nonexistent vault path raises error."""
        with pytest.raises(ValidationError) as exc_info:
            RAGConfig(
                vault_path=Path("/nonexistent/vault"),
                voyage_api_key="test-key"
            )

        error_msg = str(exc_info.value)
        assert "vault path does not exist" in error_msg.lower()

    def test_vault_path_validation_not_directory(self, temp_vault):
        """Test that file path instead of directory raises error."""
        # Create a file
        file_path = temp_vault / "file.txt"
        file_path.write_text("content")

        with pytest.raises(ValidationError) as exc_info:
            RAGConfig(
                vault_path=file_path,
                voyage_api_key="test-key"
            )

        error_msg = str(exc_info.value)
        assert "not a directory" in error_msg.lower()

    def test_voyage_api_key_validation_with_voyage_model(self, temp_vault):
        """Test that Voyage API key is required when using Voyage models."""
        with pytest.raises(ValidationError) as exc_info:
            RAGConfig(
                vault_path=temp_vault,
                embedding=EmbeddingConfig(model="voyage-3-large"),
                voyage_api_key=""  # Empty API key
            )

        error_msg = str(exc_info.value)
        assert "VOYAGE_API_KEY is required" in error_msg

    def test_voyage_api_key_not_required_without_voyage_model(self, temp_vault):
        """Test that Voyage API key is not required for non-Voyage models."""
        config = RAGConfig(
            vault_path=temp_vault,
            embedding=EmbeddingConfig(model="qwen3-8b"),  # Non-Voyage, non-OpenAI model
            retrieval=RetrievalConfig(reranker_model="jina-reranker-v2"),  # Non-Voyage reranker
            voyage_api_key=""
        )
        assert config.voyage_api_key.get_secret_value() == ""

    def test_voyage_api_key_validation_with_reranker(self, temp_vault):
        """Test that Voyage API key is required when using Voyage reranker."""
        with pytest.raises(ValidationError) as exc_info:
            RAGConfig(
                vault_path=temp_vault,
                embedding=EmbeddingConfig(model="other-model"),
                retrieval=RetrievalConfig(reranker_model="voyage-rerank-2"),
                voyage_api_key=""
            )

        error_msg = str(exc_info.value)
        assert "VOYAGE_API_KEY is required" in error_msg

    def test_openai_api_key_validation(self, temp_vault):
        """Test that OpenAI API key is validated when using OpenAI models."""
        with pytest.raises(ValidationError) as exc_info:
            RAGConfig(
                vault_path=temp_vault,
                embedding=EmbeddingConfig(model="openai-3-large"),  # OpenAI model
                retrieval=RetrievalConfig(reranker_model="jina-reranker-v2"),  # Non-Voyage reranker
                voyage_api_key="",
                openai_api_key=""
            )

        error_msg = str(exc_info.value)
        assert "OPENAI_API_KEY is required" in error_msg

    def test_nested_config_objects(self, temp_vault):
        """Test that nested config objects are properly initialized."""
        config = RAGConfig(
            vault_path=temp_vault,
            embedding=EmbeddingConfig(chunk_size=1024),
            vector_db=VectorDBConfig(db_type="qdrant"),
            llm=LLMConfig(temperature=0.5),
            voyage_api_key="test-key"
        )

        assert config.embedding.chunk_size == 1024
        assert config.vector_db.db_type == "qdrant"
        assert config.llm.temperature == 0.5

    def test_default_factory_configs(self, temp_vault):
        """Test that default factory configs are created."""
        config = RAGConfig(
            vault_path=temp_vault,
            voyage_api_key="test-key"
        )

        # Verify default configs were created
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.vector_db, VectorDBConfig)
        assert isinstance(config.graph_db, GraphDBConfig)
        assert isinstance(config.retrieval, RetrievalConfig)
        assert isinstance(config.llm, LLMConfig)


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_missing_vault_path(self, monkeypatch):
        """Test that load_config raises error when OBSIDIAN_VAULT_PATH is missing."""
        # Remove the environment variable
        monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)

        with pytest.raises(ValueError) as exc_info:
            load_config()

        assert "OBSIDIAN_VAULT_PATH must be set" in str(exc_info.value)

    def test_load_config_with_env_vars(self, temp_vault, monkeypatch):
        """Test loading config from environment variables."""
        # Set environment variables
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
        monkeypatch.setenv("EMBEDDING_MODEL", "voyage-3.5-lite")
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
        monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")

        config = load_config()

        assert config.vault_path == temp_vault
        assert config.embedding.model == "voyage-3.5-lite"
        assert config.embedding.chunk_size == 1024
        assert config.embedding.chunk_overlap == 100
        assert config.llm.temperature == 0.5
        assert config.voyage_api_key.get_secret_value() == "test-voyage-key"

    def test_load_config_default_values(self, temp_vault, monkeypatch):
        """Test that load_config uses default values when env vars not set."""
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

        config = load_config()

        # Should use defaults
        assert config.embedding.model == "voyage-3.5-lite"
        assert config.embedding.chunk_size == 512
        assert config.llm.temperature == 0.1
        assert config.vector_db.db_type == "lancedb"

    def test_load_config_type_conversion(self, temp_vault, monkeypatch):
        """Test that load_config properly converts string env vars to correct types."""
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(temp_vault))
        monkeypatch.setenv("CHUNK_SIZE", "2048")
        monkeypatch.setenv("TOP_K", "100")
        monkeypatch.setenv("LLM_TEMPERATURE", "1.5")
        monkeypatch.setenv("ENABLE_HYBRID_SEARCH", "false")
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

        config = load_config()

        # Verify types were converted correctly
        assert isinstance(config.embedding.chunk_size, int)
        assert config.embedding.chunk_size == 2048
        assert isinstance(config.retrieval.top_k, int)
        assert config.retrieval.top_k == 100
        assert isinstance(config.llm.temperature, float)
        assert config.llm.temperature == 1.5
        assert isinstance(config.retrieval.enable_hybrid_search, bool)
        assert config.retrieval.enable_hybrid_search is False


class TestConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_chunk_size(self, temp_vault):
        """Test config with very large chunk size."""
        config = RAGConfig(
            vault_path=temp_vault,
            embedding=EmbeddingConfig(chunk_size=10000, chunk_overlap=1000),
            voyage_api_key="test-key"
        )
        assert config.embedding.chunk_size == 10000

    def test_very_small_chunk_size(self, temp_vault):
        """Test config with very small chunk size."""
        config = RAGConfig(
            vault_path=temp_vault,
            embedding=EmbeddingConfig(chunk_size=10, chunk_overlap=2),
            voyage_api_key="test-key"
        )
        assert config.embedding.chunk_size == 10

    def test_zero_chunk_overlap(self, temp_vault):
        """Test config with zero chunk overlap."""
        config = RAGConfig(
            vault_path=temp_vault,
            embedding=EmbeddingConfig(chunk_size=512, chunk_overlap=0),
            voyage_api_key="test-key"
        )
        assert config.embedding.chunk_overlap == 0

    def test_temperature_boundary_zero(self, temp_vault):
        """Test temperature at boundary value 0."""
        config = RAGConfig(
            vault_path=temp_vault,
            llm=LLMConfig(temperature=0.0),
            voyage_api_key="test-key"
        )
        assert config.llm.temperature == 0.0

    def test_temperature_boundary_two(self, temp_vault):
        """Test temperature at boundary value 2."""
        config = RAGConfig(
            vault_path=temp_vault,
            llm=LLMConfig(temperature=2.0),
            voyage_api_key="test-key"
        )
        assert config.llm.temperature == 2.0

    def test_path_with_special_characters(self, temp_vault):
        """Test vault path with special characters."""
        special_dir = temp_vault / "vault-with-special_chars.123"
        special_dir.mkdir()

        config = RAGConfig(
            vault_path=special_dir,
            voyage_api_key="test-key"
        )
        assert config.vault_path == special_dir

    def test_unicode_in_path(self, temp_vault):
        """Test vault path with unicode characters."""
        unicode_dir = temp_vault / "测试文件夹"
        unicode_dir.mkdir()

        config = RAGConfig(
            vault_path=unicode_dir,
            voyage_api_key="test-key"
        )
        assert config.vault_path == unicode_dir
