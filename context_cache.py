"""Gemini context caching for repeated LLM calls with shared context.

Reduces cost and latency when the same large context (>32K tokens) is used
across multiple LLM calls within a session (e.g., research mode iterations).

Context caching stores pre-processed content on Gemini's servers, allowing
subsequent calls to reference it without re-sending. Cached tokens are billed
at a reduced rate (typically 75% less than regular input tokens).

Requirements:
- Minimum 32,768 tokens of cached content
- TTL-based expiration (default: 5 minutes for session-scoped caches)
- Supported on gemini-2.0-flash-001, gemini-2.5-flash, and later models
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum tokens required for context caching (Gemini API requirement)
MIN_CACHE_TOKENS = 32_768

# Default TTL for session caches (5 minutes, sufficient for research sessions)
DEFAULT_TTL = "300s"


class GeminiContextCache:
    """Manages Gemini context caches for a research session.

    Creates a cache with the accumulated retrieval context and system instruction,
    then provides cached completions for subsequent LLM calls.

    Usage:
        cache = GeminiContextCache(model="gemini-3-flash-preview")
        cache.create_cache(context_text, system_instruction="Analyze gaps...")

        # Subsequent calls use the cache:
        response = cache.cached_complete("What gaps exist?")

        # Clean up when done:
        cache.delete_cache()
    """

    def __init__(self, model: str = "gemini-3-flash-preview", ttl: str = DEFAULT_TTL):
        self.model = model
        self.ttl = ttl
        self._client = None
        self._cache_name: Optional[str] = None
        self._cached_token_count: int = 0

    def _get_client(self):
        """Lazy-init the google.genai client."""
        if self._client is None:
            from google import genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY required for context caching")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token, conservative for mixed content)."""
        return max(1, len(text) // 4)

    @property
    def is_active(self) -> bool:
        """Whether a cache is currently active."""
        return self._cache_name is not None

    @property
    def cached_tokens(self) -> int:
        """Number of tokens currently cached."""
        return self._cached_token_count

    def create_cache(
        self,
        context_text: str,
        system_instruction: str = "",
        display_name: str = "ultrarag_research"
    ) -> bool:
        """Create a context cache with the provided content.

        Args:
            context_text: The context to cache (retrieved nodes, etc.)
            system_instruction: System instruction for the cached context
            display_name: Human-readable name for the cache

        Returns:
            True if cache was created, False if content too small or error
        """
        estimated_tokens = self.estimate_tokens(context_text + system_instruction)

        if estimated_tokens < MIN_CACHE_TOKENS:
            logger.debug(
                f"Context too small for caching: ~{estimated_tokens} tokens "
                f"(minimum: {MIN_CACHE_TOKENS})"
            )
            return False

        try:
            from google.genai import types

            client = self._get_client()

            # Delete existing cache if present
            if self._cache_name:
                self.delete_cache()

            cached_content = client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    contents=[
                        types.Content(
                            role='user',
                            parts=[types.Part.from_text(text=context_text)]
                        )
                    ],
                    system_instruction=system_instruction,
                    display_name=display_name,
                    ttl=self.ttl,
                )
            )

            self._cache_name = cached_content.name
            self._cached_token_count = estimated_tokens

            logger.info(
                f"Context cache created: {self._cache_name} "
                f"(~{estimated_tokens:,} tokens, TTL={self.ttl})"
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to create context cache: {e}")
            self._cache_name = None
            return False

    def cached_complete(self, prompt: str) -> Optional[str]:
        """Make a completion call using the cached context.

        Args:
            prompt: The new prompt to send (cache provides the context)

        Returns:
            Response text, or None if cache not active or call fails
        """
        if not self._cache_name:
            return None

        try:
            from google.genai import types

            client = self._get_client()

            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    cached_content=self._cache_name,
                    temperature=0.1,
                    max_output_tokens=4096,
                )
            )

            # Extract response text
            if response.candidates and response.candidates[0].content.parts:
                result_text = response.candidates[0].content.parts[0].text
                logger.debug(f"Cached completion successful ({len(result_text)} chars)")
                return result_text

            return None

        except Exception as e:
            logger.warning(f"Cached completion failed: {e}")
            # Cache might have expired
            if "not found" in str(e).lower() or "expired" in str(e).lower():
                self._cache_name = None
            return None

    def delete_cache(self) -> None:
        """Delete the current cache."""
        if not self._cache_name:
            return

        try:
            client = self._get_client()
            client.caches.delete(name=self._cache_name)
            logger.info(f"Context cache deleted: {self._cache_name}")
        except Exception as e:
            logger.debug(f"Cache deletion failed (may have expired): {e}")
        finally:
            self._cache_name = None
            self._cached_token_count = 0

    def __del__(self):
        """Clean up cache on garbage collection."""
        try:
            self.delete_cache()
        except Exception:
            pass
