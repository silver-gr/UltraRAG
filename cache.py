"""Caching layers for embeddings and queries to improve performance."""
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for embedding vectors to avoid recomputing identical texts."""

    def __init__(self, cache_dir: Path = Path("./data/embedding_cache")):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Embedding cache initialized at: {self.cache_dir}")

    def get_cache_key(self, text: str) -> str:
        """
        Generate a cache key from text using SHA256 hash.

        Args:
            text: Input text to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[Any]:
        """
        Retrieve cached embedding for given text.

        Args:
            text: Input text to lookup

        Returns:
            Cached embedding vector or None if not found
        """
        cache_file = self.cache_dir / f"{self.get_cache_key(text)}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return embedding
            except Exception as e:
                logger.warning(f"Cache read error for {cache_file}: {e}")
                # If cache is corrupted, remove it
                try:
                    cache_file.unlink()
                except Exception:
                    pass

        logger.debug(f"Cache miss for text: {text[:50]}...")
        return None

    def set(self, text: str, embedding: Any) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text (key)
            embedding: Embedding vector to cache
        """
        cache_file = self.cache_dir / f"{self.get_cache_key(text)}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            logger.debug(f"Cached embedding for text: {text[:50]}...")
        except Exception as e:
            logger.warning(f"Cache write error for {cache_file}: {e}")

    def clear(self) -> None:
        """Clear all cached embeddings."""
        try:
            count = 0
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cached embeddings")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and count
        """
        try:
            files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in files)
            return {
                "count": len(files),
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"count": 0, "size_bytes": 0, "size_mb": 0}
