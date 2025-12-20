"""Embedding model initialization and management with token tracking."""
from __future__ import annotations

import os
import logging
from typing import Optional, Any, Union, List
from pydantic import SecretStr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from config import EmbeddingConfig
from token_tracker import get_tracker, QuotaExhaustedError

logger = logging.getLogger(__name__)


class TrackedVoyageEmbedding(BaseEmbedding):
    """Voyage embedding wrapper with token usage tracking."""

    def __init__(self, base_embedding: BaseEmbedding, model_name: str):
        super().__init__()
        self._base = base_embedding
        self._model_name = model_name
        self._tracker = get_tracker()

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text with tracking."""
        estimated_tokens = self._tracker.estimate_tokens([text])

        can_proceed, msg = self._tracker.check_embedding_quota(estimated_tokens)
        if not can_proceed:
            raise QuotaExhaustedError(msg)

        result = self._base._get_text_embedding(text)
        self._tracker.record_embedding_usage(estimated_tokens, self._model_name)
        return result

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with tracking."""
        estimated_tokens = self._tracker.estimate_tokens(texts)

        can_proceed, msg = self._tracker.check_embedding_quota(estimated_tokens)
        if not can_proceed:
            raise QuotaExhaustedError(msg)

        result = self._base._get_text_embeddings(texts)
        self._tracker.record_embedding_usage(estimated_tokens, self._model_name)
        return result

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async get embedding with tracking."""
        estimated_tokens = self._tracker.estimate_tokens([text])

        can_proceed, msg = self._tracker.check_embedding_quota(estimated_tokens)
        if not can_proceed:
            raise QuotaExhaustedError(msg)

        result = await self._base._aget_text_embedding(text)
        self._tracker.record_embedding_usage(estimated_tokens, self._model_name)
        return result

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async get embeddings with tracking."""
        estimated_tokens = self._tracker.estimate_tokens(texts)

        can_proceed, msg = self._tracker.check_embedding_quota(estimated_tokens)
        if not can_proceed:
            raise QuotaExhaustedError(msg)

        result = await self._base._aget_text_embeddings(texts)
        self._tracker.record_embedding_usage(estimated_tokens, self._model_name)
        return result

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding with tracking."""
        estimated_tokens = self._tracker.estimate_tokens([query])

        can_proceed, msg = self._tracker.check_embedding_quota(estimated_tokens)
        if not can_proceed:
            raise QuotaExhaustedError(msg)

        result = self._base._get_query_embedding(query)
        self._tracker.record_embedding_usage(estimated_tokens, self._model_name)
        return result

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async get query embedding with tracking."""
        estimated_tokens = self._tracker.estimate_tokens([query])

        can_proceed, msg = self._tracker.check_embedding_quota(estimated_tokens)
        if not can_proceed:
            raise QuotaExhaustedError(msg)

        result = await self._base._aget_query_embedding(query)
        self._tracker.record_embedding_usage(estimated_tokens, self._model_name)
        return result


class TrackedVoyageReranker:
    """Voyage reranker wrapper with token usage tracking."""

    def __init__(self, base_reranker: Any, model_name: str):
        self._base = base_reranker
        self._model_name = model_name
        self._tracker = get_tracker()

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Any = None,
        **kwargs
    ) -> List[NodeWithScore]:
        """Rerank nodes with token tracking."""
        # Estimate tokens: query + all document texts
        texts = [node.node.get_content() for node in nodes]
        query_text = query_bundle.query_str if query_bundle else ""
        all_texts = [query_text] + texts

        estimated_tokens = self._tracker.estimate_tokens(all_texts)

        can_proceed, msg = self._tracker.check_rerank_quota(estimated_tokens)
        if not can_proceed:
            logger.warning(f"Rerank quota exceeded, returning unranked results: {msg}")
            # Don't fail, just return original order
            return nodes[:self._base.top_n] if hasattr(self._base, 'top_n') else nodes

        result = self._base.postprocess_nodes(nodes, query_bundle, **kwargs)
        self._tracker.record_rerank_usage(estimated_tokens, self._model_name)
        return result

    def __getattr__(self, name):
        """Delegate other attributes to base reranker."""
        return getattr(self._base, name)


def get_embedding_model(
    config: EmbeddingConfig,
    api_key: Optional[Union[str, SecretStr]] = None,
    enable_tracking: bool = True
) -> BaseEmbedding:
    """Initialize embedding model based on configuration.

    Args:
        config: Embedding configuration
        api_key: Optional API key (will use env var if not provided)
        enable_tracking: Enable token usage tracking for Voyage models (default: True)
    """
    model_name = config.model.lower()
    logger.info(f"Initializing embedding model: {config.model}")

    try:
        if "voyage" in model_name:
            from llama_index.embeddings.voyageai import VoyageEmbedding

            # Security: Extract secret value if SecretStr
            key_value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key

            if not key_value:
                key_value = os.getenv("VOYAGE_API_KEY")
                if not key_value:
                    logger.error("VOYAGE_API_KEY not found in environment variables")
                    raise ValueError(
                        "VOYAGE_API_KEY not found. Please set it in your .env file.\n"
                        "Get your API key from: https://www.voyageai.com/"
                    )

            logger.debug(f"Creating VoyageEmbedding with model: {config.model}")
            base_embedding = VoyageEmbedding(
                model_name=config.model,
                voyage_api_key=key_value,
                truncation=True
            )

            # Wrap with token tracking for Voyage models
            if enable_tracking:
                logger.info(f"Enabling token tracking for {config.model}")
                tracker = get_tracker()
                tracker.print_status()
                return TrackedVoyageEmbedding(base_embedding, config.model)

            return base_embedding

        elif "qwen" in model_name or "8b" in model_name:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.debug("Using HuggingFace embedding: Alibaba-NLP/gte-Qwen2-1.5B-instruct")
            return HuggingFaceEmbedding(
                model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                trust_remote_code=True,
                embed_batch_size=32
            )

        elif "openai" in model_name:
            from llama_index.embeddings.openai import OpenAIEmbedding

            # Security: Extract secret value if SecretStr
            key_value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key

            if not key_value:
                key_value = os.getenv("OPENAI_API_KEY")
                if not key_value:
                    logger.error("OPENAI_API_KEY not found in environment variables")
                    raise ValueError(
                        "OPENAI_API_KEY not found. Please set it in your .env file.\n"
                        "Get your API key from: https://platform.openai.com/api-keys"
                    )

            logger.debug(f"Creating OpenAIEmbedding with dimensions: {config.dimension}")
            return OpenAIEmbedding(
                model="text-embedding-3-large",
                api_key=key_value,
                dimensions=config.dimension
            )

        else:
            # Default to a good open-source model
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            logger.warning(f"Unknown model {model_name}, defaulting to nomic-embed-text-v1.5")
            return HuggingFaceEmbedding(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                trust_remote_code=True
            )

    except ImportError as e:
        logger.error(f"Failed to import required embedding library: {e}")
        raise ImportError(
            f"Missing required library for {model_name}. "
            f"Install it with: pip install llama-index-embeddings-{model_name.split('-')[0]}"
        ) from e
    except Exception as e:
        logger.error(f"Failed to initialize embedding model {config.model}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize embedding model: {e}") from e


def get_reranker(
    model_name: str = "voyage-rerank-2.5",
    api_key: Optional[Union[str, SecretStr]] = None,
    top_n: int = 10,
    enable_tracking: bool = True
) -> Any:
    """Initialize reranker model.

    Args:
        model_name: Reranker model name (default: voyage-rerank-2.5)
        api_key: Optional API key
        top_n: Number of top results to return
        enable_tracking: Enable token usage tracking for Voyage models (default: True)
    """
    logger.info(f"Initializing reranker: {model_name}")

    try:
        # Voyage reranker models: voyage-rerank-X or just rerank-X
        is_voyage_reranker = "voyage" in model_name.lower() or model_name.lower().startswith("rerank-")

        if is_voyage_reranker:
            try:
                # Package: llama-index-postprocessor-voyageai-rerank
                from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank
            except ImportError:
                # Fallback to similarity-based reranking
                logger.warning("VoyageAI reranker not available. Install with: pip install llama-index-postprocessor-voyageai-rerank")
                from llama_index.core.postprocessor import SimilarityPostprocessor
                logger.info("Falling back to similarity-based reranking")
                return SimilarityPostprocessor(similarity_cutoff=0.7)

            # Security: Extract secret value if SecretStr
            key_value = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key

            if not key_value:
                key_value = os.getenv("VOYAGE_API_KEY")
                if not key_value:
                    logger.warning("VOYAGE_API_KEY not found, reranker may not work")

            base_reranker = VoyageAIRerank(
                model=model_name,
                api_key=key_value,
                top_n=top_n
            )

            # Wrap with token tracking
            if enable_tracking:
                logger.info(f"Enabling token tracking for {model_name}")
                return TrackedVoyageReranker(base_reranker, model_name)

            return base_reranker

        elif "jina" in model_name.lower():
            try:
                from llama_index.postprocessor.jinaai_rerank import JinaRerank
            except ImportError:
                logger.warning("Jina reranker not available. Install with: pip install llama-index-postprocessor-jinaai-rerank")
                from llama_index.core.postprocessor import SimilarityPostprocessor
                return SimilarityPostprocessor(similarity_cutoff=0.7)

            return JinaRerank(
                model=model_name,
                top_n=10
            )

        elif "cohere" in model_name.lower():
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
            except ImportError:
                logger.warning("Cohere reranker not available. Install with: pip install llama-index-postprocessor-cohere-rerank")
                from llama_index.core.postprocessor import SimilarityPostprocessor
                return SimilarityPostprocessor(similarity_cutoff=0.7)

            return CohereRerank(
                model=model_name,
                top_n=10
            )

        else:
            # Default: use similarity-based reranking
            from llama_index.core.postprocessor import SimilarityPostprocessor

            logger.info("Using default similarity-based reranking")
            return SimilarityPostprocessor(similarity_cutoff=0.7)

    except ImportError as e:
        logger.error(f"Failed to import reranker library: {e}")
        logger.info("Falling back to similarity-based reranking")
        from llama_index.core.postprocessor import SimilarityPostprocessor
        return SimilarityPostprocessor(similarity_cutoff=0.7)
    except Exception as e:
        logger.error(f"Failed to initialize reranker {model_name}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize reranker: {e}") from e
