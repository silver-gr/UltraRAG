"""Configuration management for UltraRAG system."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

load_dotenv()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model: str = Field(default="voyage-3.5-lite")  # Free tier: 200M tokens
    dimension: int = Field(default=1024)
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=75)
    batch_size: int = Field(default=100)
    chunking_strategy: str = Field(default="obsidian_aware")
    late_chunking_alpha: float = Field(default=0.7)
    use_contextual_retrieval: bool = Field(default=False)  # Calls LLM per chunk - costs tokens!
    token_limit: int = Field(default=200_000_000)  # 200M free tier limit

    @field_validator('chunking_strategy')
    @classmethod
    def validate_chunking_strategy(cls, v: str) -> str:
        """Validate chunking strategy."""
        valid_strategies = ["obsidian_aware", "markdown_semantic", "late_chunking", "semantic", "markdown", "simple"]
        if v.lower() not in valid_strategies:
            raise ValueError(
                f"chunking_strategy must be one of {valid_strategies}, got '{v}'."
            )
        return v.lower()

    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        """Validate that chunk_overlap is less than chunk_size."""
        chunk_size = info.data.get('chunk_size')
        if chunk_size and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size}). "
                f"Recommended: chunk_overlap should be 10-25% of chunk_size."
            )
        return v

    @field_validator('late_chunking_alpha')
    @classmethod
    def validate_late_chunking_alpha(cls, v: float) -> float:
        """Validate that late_chunking_alpha is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(
                f"late_chunking_alpha must be between 0 and 1, got {v}. "
                f"Recommended: 0.7 (70% local context, 30% global context)."
            )
        return v


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    db_type: str = Field(default="lancedb")
    lancedb_path: Path = Field(default=Path("./data/lancedb"))
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="obsidian_notes")


class GraphDBConfig(BaseModel):
    """Graph database configuration."""
    enabled: bool = Field(default=False)
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="")


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = Field(default=75)
    rerank_top_n: int = Field(default=10)
    reranker_model: str = Field(default="rerank-2.5")  # Voyage model (no "voyage-" prefix!)
    reranker_token_limit: int = Field(default=200_000_000)  # 200M free tier limit
    enable_hybrid_search: bool = Field(default=True)
    enable_graph_search: bool = Field(default=False)
    enable_graph_retrieval: bool = Field(default=True)
    graph_retrieval_depth: int = Field(default=1)
    graph_retrieval_max_links: int = Field(default=3)
    similarity_threshold: float = Field(default=0.3)  # Lowered from 0.7 - cosine similarity often < 0.5

    # Query transformation settings
    query_transform_method: str = Field(default="hyde")
    query_transform_num_queries: int = Field(default=3)

    # Self-correction settings (Self-RAG and CRAG)
    use_self_correction: bool = Field(default=True)
    self_correction_max_retries: int = Field(default=2)

    @field_validator('query_transform_method')
    @classmethod
    def validate_query_transform_method(cls, v: str) -> str:
        """Validate query transformation method."""
        valid_methods = ["none", "disabled", "hyde", "multi_query", "both"]
        if v.lower() not in valid_methods:
            raise ValueError(
                f"query_transform_method must be one of {valid_methods}, got '{v}'. "
                f"Options: 'none'/'disabled' (no transformation), 'hyde' (hypothetical document embeddings), "
                f"'multi_query' (query expansion), 'both' (HyDE + multi-query)"
            )
        return v.lower()


class LLMConfig(BaseModel):
    """LLM configuration."""
    model: str = Field(default="gemini-3-flash-preview")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=8192)
    enable_thinking_mode: bool = Field(default=True)
    backend: str = Field(default="api")  # "api" or "cli" (uses gemini CLI for separate quota)

    @field_validator('backend')
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate LLM backend choice."""
        valid_backends = ["api", "cli"]
        if v.lower() not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got '{v}'. "
                f"'api' uses Google Gemini API directly. "
                f"'cli' uses Gemini CLI with separate free tier quota (1000/day)."
            )
        return v.lower()

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError(
                f"temperature must be between 0 and 2, got {v}. "
                f"Lower values (0-0.3) are more deterministic, higher values (0.7-2.0) are more creative."
            )
        return v


class RAGConfig(BaseModel):
    """Main RAG system configuration."""
    vault_path: Path
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    graph_db: GraphDBConfig = Field(default_factory=GraphDBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Indexing options
    enable_checkpointing: bool = Field(default=True)

    # Security: API Keys stored as SecretStr to prevent accidental logging/exposure
    voyage_api_key: SecretStr = Field(default=SecretStr(""))
    google_api_key: SecretStr = Field(default=SecretStr(""))
    openai_api_key: SecretStr = Field(default=SecretStr(""))

    @field_validator('vault_path')
    @classmethod
    def validate_vault_path(cls, v: Path) -> Path:
        """Validate that vault_path exists and is a directory."""
        if not v.exists():
            raise ValueError(
                f"Obsidian vault path does not exist: {v}\n"
                f"Please ensure OBSIDIAN_VAULT_PATH in .env points to a valid directory."
            )
        if not v.is_dir():
            raise ValueError(
                f"Obsidian vault path is not a directory: {v}\n"
                f"OBSIDIAN_VAULT_PATH must point to a directory containing your notes."
            )
        return v

    @field_validator('voyage_api_key')
    @classmethod
    def validate_voyage_api_key(cls, v: SecretStr, info: Any) -> SecretStr:
        """Validate Voyage API key if using Voyage models."""
        embedding_model = info.data.get('embedding', {})
        if hasattr(embedding_model, 'model'):
            model_name = embedding_model.model
        else:
            model_name = embedding_model.get('model', '') if isinstance(embedding_model, dict) else ''

        reranker_model = info.data.get('retrieval', {})
        if hasattr(reranker_model, 'reranker_model'):
            reranker_name = reranker_model.reranker_model
        else:
            reranker_name = reranker_model.get('reranker_model', '') if isinstance(reranker_model, dict) else ''

        # Check if Voyage API key is needed
        needs_voyage = (
            (model_name and 'voyage' in model_name.lower()) or
            (reranker_name and 'voyage' in reranker_name.lower())
        )

        # Check if key is empty (SecretStr)
        key_value = v.get_secret_value() if v else ""
        if needs_voyage and not key_value:
            raise ValueError(
                f"VOYAGE_API_KEY is required when using Voyage models.\n"
                f"Current embedding model: {model_name}\n"
                f"Current reranker model: {reranker_name}\n"
                f"Get a free API key at: https://www.voyageai.com/"
            )
        return v

    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_api_key(cls, v: SecretStr, info: Any) -> SecretStr:
        """Validate OpenAI API key if using OpenAI models."""
        embedding_model = info.data.get('embedding', {})
        if hasattr(embedding_model, 'model'):
            model_name = embedding_model.model
        else:
            model_name = embedding_model.get('model', '') if isinstance(embedding_model, dict) else ''

        # Check if key is empty (SecretStr)
        key_value = v.get_secret_value() if v else ""
        if model_name and 'openai' in model_name.lower() and not key_value:
            raise ValueError(
                f"OPENAI_API_KEY is required when using OpenAI embedding model: {model_name}\n"
                f"Get an API key at: https://platform.openai.com/"
            )
        return v


def load_config() -> RAGConfig:
    """Load configuration from environment variables."""
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", "")
    if not vault_path:
        raise ValueError("OBSIDIAN_VAULT_PATH must be set in .env file")

    return RAGConfig(
        vault_path=Path(vault_path),
        embedding=EmbeddingConfig(
            model=os.getenv("EMBEDDING_MODEL", "voyage-3.5-lite"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "75")),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", "obsidian_aware"),
            late_chunking_alpha=float(os.getenv("LATE_CHUNKING_ALPHA", "0.7")),
            use_contextual_retrieval=os.getenv("USE_CONTEXTUAL_RETRIEVAL", "false").lower() == "true",
            token_limit=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "200000000"))
        ),
        vector_db=VectorDBConfig(
            db_type=os.getenv("VECTOR_DB", "lancedb"),
            lancedb_path=Path(os.getenv("LANCEDB_PATH", "./data/lancedb")),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "obsidian_notes")
        ),
        graph_db=GraphDBConfig(
            enabled=os.getenv("ENABLE_GRAPH_SEARCH", "false").lower() == "true",
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "")
        ),
        retrieval=RetrievalConfig(
            top_k=int(os.getenv("TOP_K", "75")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "10")),
            reranker_model=os.getenv("RERANKER_MODEL", "rerank-2.5"),
            reranker_token_limit=int(os.getenv("RERANKER_TOKEN_LIMIT", "200000000")),
            enable_hybrid_search=os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
            enable_graph_search=os.getenv("ENABLE_GRAPH_SEARCH", "false").lower() == "true",
            enable_graph_retrieval=os.getenv("ENABLE_GRAPH_RETRIEVAL", "true").lower() == "true",
            graph_retrieval_depth=int(os.getenv("GRAPH_RETRIEVAL_DEPTH", "1")),
            graph_retrieval_max_links=int(os.getenv("GRAPH_RETRIEVAL_MAX_LINKS", "3")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
            query_transform_method=os.getenv("QUERY_TRANSFORM_METHOD", "hyde"),
            query_transform_num_queries=int(os.getenv("QUERY_TRANSFORM_NUM_QUERIES", "3")),
            use_self_correction=os.getenv("USE_SELF_CORRECTION", "true").lower() == "true",
            self_correction_max_retries=int(os.getenv("SELF_CORRECTION_MAX_RETRIES", "2"))
        ),
        llm=LLMConfig(
            model=os.getenv("LLM_MODEL", "gemini-3-flash-preview"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
            backend=os.getenv("LLM_BACKEND", "api")  # "api" or "cli"
        ),
        enable_checkpointing=os.getenv("ENABLE_CHECKPOINTING", "true").lower() == "true",
        voyage_api_key=SecretStr(os.getenv("VOYAGE_API_KEY", "")),
        google_api_key=SecretStr(os.getenv("GOOGLE_API_KEY", "")),
        openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", ""))
    )
