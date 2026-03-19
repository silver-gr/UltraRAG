"""Tracked LLM wrapper that records token usage for all LLM calls.

Wraps any LlamaIndex LLM and intercepts calls to record token usage
to the LLMTokenTracker. Supports model fallback chains, Gemini retry-delay
parsing, and graceful degradation on complete failure.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional, Sequence, Callable

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from llm_token_tracker import get_llm_tracker, LLMTokenTracker

logger = logging.getLogger(__name__)

# Maximum retry delay we'll honor from Gemini (seconds)
_MAX_RETRY_DELAY = 120
# If Gemini says retry in <= this many seconds, wait instead of falling back
_SHORT_RETRY_THRESHOLD = 10


def is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (transient/rate-limit).

    Retryable errors include:
    - HTTP 429 (rate limit)
    - HTTP 500, 502, 503, 504 (server errors)
    - google.api_core.exceptions with those codes
    - TimeoutError, ConnectionError
    - ValueError with "empty response" in message

    Args:
        exc: The exception to check.

    Returns:
        True if the error is retryable.
    """
    # Timeout and connection errors are always retryable
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True

    # ValueError with "empty response" (Gemini sometimes returns empty)
    if isinstance(exc, ValueError) and "empty response" in str(exc).lower():
        return True

    exc_str = str(exc).lower()

    # Check for HTTP status codes in exception message
    retryable_codes = {"429", "500", "502", "503", "504"}
    for code in retryable_codes:
        if code in exc_str:
            return True

    # Check for google.api_core.exceptions attributes
    status_code = getattr(exc, 'code', None) or getattr(exc, 'grpc_status_code', None)
    if status_code is not None:
        try:
            code_int = int(status_code)
            if code_int in (429, 500, 502, 503, 504):
                return True
        except (ValueError, TypeError):
            pass

    # Check for "rate limit", "resource exhausted", "quota" patterns
    if any(pattern in exc_str for pattern in ("rate limit", "resource exhausted", "quota exceeded", "too many requests")):
        return True

    return False


def extract_gemini_retry_delay(exc: Exception) -> Optional[int]:
    """Extract retry delay from Gemini error details.

    Gemini 429 errors may include a retryDelay field in error details.
    Parses that and returns the delay in seconds, capped at _MAX_RETRY_DELAY.

    Args:
        exc: The exception from a Gemini API call.

    Returns:
        Delay in seconds, or None if no delay found.
    """
    # Check error.errors list (google.api_core style)
    errors = getattr(exc, 'errors', None)
    if errors:
        for detail in errors:
            detail_str = str(detail)
            if 'retryDelay' in detail_str or 'retry_delay' in detail_str:
                match = re.search(r'(\d+)s', detail_str)
                if match:
                    return min(int(match.group(1)), _MAX_RETRY_DELAY)

    # Also check the exception message itself for retryDelay patterns
    exc_str = str(exc)
    if 'retryDelay' in exc_str or 'retry_delay' in exc_str:
        match = re.search(r'retryDelay["\s:=]+(\d+)s', exc_str, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), _MAX_RETRY_DELAY)
        # Also try "Xs" pattern near retryDelay
        match = re.search(r'retry.?[Dd]elay.*?(\d+)\s*s', exc_str)
        if match:
            return min(int(match.group(1)), _MAX_RETRY_DELAY)

    return None


class TrackedLLM(CustomLLM):
    """LLM wrapper that tracks token usage for all calls.

    Wraps any LlamaIndex-compatible LLM and records input/output tokens
    to the LLMTokenTracker for cost tracking. Supports optional fallback
    models on retryable errors.

    Attributes:
        inner_llm: The underlying LLM to wrap
        model_name: Model name for pricing (extracted from inner LLM)
        tracker: The token tracker instance
        fallback_models: List of fallback model names to try on retryable errors
        _llm_factory: Optional factory to create LLM instances for fallback models
    """

    inner_llm: Any = None
    model_name: str = "gemini-3-flash-preview"
    tracker: Optional[LLMTokenTracker] = None
    fallback_models: list[str] = []
    _llm_factory: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Any,
        model_name: Optional[str] = None,
        tracker: Optional[LLMTokenTracker] = None,
        fallback_models: Optional[list[str]] = None,
        llm_factory: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Initialize tracked LLM wrapper.

        Args:
            llm: The underlying LLM to wrap
            model_name: Override model name for pricing (default: extract from llm)
            tracker: Token tracker instance (default: global tracker)
            fallback_models: List of fallback model names (default: empty)
            llm_factory: Callable(model_name) -> LLM for creating fallback instances
        """
        # Extract model name from inner LLM if not provided
        if model_name is None:
            model_name = getattr(llm, 'model', None) or \
                         getattr(llm, 'model_name', None) or \
                         "gemini-3-flash-preview"

        super().__init__(
            inner_llm=llm,
            model_name=model_name,
            tracker=tracker or get_llm_tracker(),
            fallback_models=fallback_models or [],
            _llm_factory=llm_factory,
            **kwargs,
        )

        if self.fallback_models:
            logger.info(
                f"Initialized TrackedLLM wrapping {model_name} "
                f"with fallbacks: {self.fallback_models}"
            )
        else:
            logger.info(f"Initialized TrackedLLM wrapping {model_name}")

    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata from inner LLM."""
        return self.inner_llm.metadata

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (fallback when actual count unavailable)."""
        # Gemini uses roughly 4 chars per token for English
        return max(1, len(text) // 4)

    def _extract_token_counts(self, response: Any) -> tuple[int, int, int, int, int]:
        """Extract token counts from response if available.

        Returns:
            (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, thought_tokens) tuple
        """
        # Try to get actual token counts from response metadata
        # LlamaIndex responses may have raw attribute with usage info
        try:
            # Check for raw response with usage_metadata (Google GenAI)
            raw = getattr(response, 'raw', None)
            if raw:
                usage = getattr(raw, 'usage_metadata', None)
                if usage:
                    input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    thought_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
                    cache_read_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
                    # cache_write_tokens not directly available in usage_metadata; default 0
                    cache_write_tokens = 0
                    if input_tokens > 0 or output_tokens > 0:
                        logger.debug(
                            f"Extracted actual tokens: {input_tokens} in, {output_tokens} out"
                            + (f", {cache_read_tokens} cache_read" if cache_read_tokens else "")
                            + (f", {thought_tokens} thought" if thought_tokens else "")
                        )
                        return input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, thought_tokens

            # Check for additional_kwargs
            additional = getattr(response, 'additional_kwargs', {})
            if additional:
                usage = additional.get('usage', {}) or additional.get('usage_metadata', {})
                if usage:
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('prompt_token_count', 0) or 0
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('candidates_token_count', 0) or 0
                    thought_tokens = usage.get('thoughts_token_count', 0) or 0
                    cache_read_tokens = usage.get('cached_content_token_count', 0) or 0
                    cache_write_tokens = 0
                    if input_tokens > 0 or output_tokens > 0:
                        return input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, thought_tokens

        except Exception as e:
            logger.debug(f"Could not extract token counts: {e}")

        return 0, 0, 0, 0, 0

    def _record_usage(
        self,
        prompt: str,
        response_text: str,
        response: Any = None,
        used_model: Optional[str] = None,
    ) -> None:
        """Record token usage for a call."""
        # Try to extract actual token counts
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, thought_tokens = \
            self._extract_token_counts(response)

        # Fall back to estimates if no actual counts
        if input_tokens == 0:
            input_tokens = self._estimate_tokens(prompt)
        if output_tokens == 0:
            output_tokens = self._estimate_tokens(response_text)

        self.tracker.record_usage(
            input_tokens,
            output_tokens,
            used_model or self.model_name,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            thought_tokens=thought_tokens,
        )

    def _create_fallback_llm(self, fallback_model: str) -> Any:
        """Create an LLM instance for a fallback model.

        Args:
            fallback_model: The model name to create an instance for.

        Returns:
            An LLM instance configured for the fallback model.
        """
        # Use factory if provided
        if self._llm_factory is not None:
            return self._llm_factory(fallback_model)

        # Default: clone inner_llm with different model name
        # Works for GoogleGenAI and similar LlamaIndex LLMs
        try:
            import os
            from llama_index.llms.google_genai import GoogleGenAI

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set for fallback model")

            return GoogleGenAI(
                model=fallback_model,
                api_key=api_key,
                temperature=getattr(self.inner_llm, 'temperature', 0.1),
                max_tokens=getattr(self.inner_llm, 'max_tokens', 65536),
            )
        except ImportError:
            raise RuntimeError(
                f"Cannot create fallback LLM for {fallback_model}: "
                "llama_index.llms.google_genai not available"
            )

    def _handle_retryable_error(
        self, exc: Exception, current_model: str, is_last: bool
    ) -> Optional[int]:
        """Handle a retryable error: check delay, decide whether to wait or fall back.

        Returns:
            Delay in seconds to sleep before retrying same model, or None to fall back.
        """
        delay = extract_gemini_retry_delay(exc)
        if delay is not None and delay <= _SHORT_RETRY_THRESHOLD:
            logger.info(
                f"{current_model}: Gemini retryDelay={delay}s (<= {_SHORT_RETRY_THRESHOLD}s), "
                f"waiting before retry"
            )
            return delay

        if delay is not None:
            logger.info(
                f"{current_model}: Gemini retryDelay={delay}s (> {_SHORT_RETRY_THRESHOLD}s), "
                f"falling back to next model"
            )
        return None

    def _make_graceful_chat_response(self, last_error: Exception) -> ChatResponse:
        """Create a graceful degradation ChatResponse when all models fail."""
        error_msg = (
            "I'm temporarily unable to generate a response due to service issues. "
            "Please try again in a moment."
        )
        logger.error(
            f"All LLM models exhausted. Last error: {last_error}",
            exc_info=True,
        )
        response = ChatResponse(
            message=ChatMessage(role="assistant", content=error_msg),
            additional_kwargs={"error": True, "raw_error": str(last_error)},
        )
        return response

    def _make_graceful_completion_response(self, last_error: Exception) -> CompletionResponse:
        """Create a graceful degradation CompletionResponse when all models fail."""
        error_msg = (
            "I'm temporarily unable to generate a response due to service issues. "
            "Please try again in a moment."
        )
        logger.error(
            f"All LLM models exhausted. Last error: {last_error}",
            exc_info=True,
        )
        return CompletionResponse(
            text=error_msg,
            additional_kwargs={"error": True, "raw_error": str(last_error)},
        )

    def _get_models_to_try(self) -> list[tuple[str, Any]]:
        """Build ordered list of (model_name, llm_instance) to try.

        Returns:
            List of (model_name, llm) tuples. First is primary, rest are fallbacks.
        """
        models = [(self.model_name, self.inner_llm)]
        for fb_model in self.fallback_models:
            try:
                fb_llm = self._create_fallback_llm(fb_model)
                models.append((fb_model, fb_llm))
            except Exception as e:
                logger.warning(f"Could not create fallback LLM for {fb_model}: {e}")
        return models

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt with fallback chain and record usage."""
        if not self.fallback_models:
            # Fast path: no fallback configured
            response = self.inner_llm.complete(prompt, **kwargs)
            self._record_usage(prompt, response.text, response)
            return response

        models = self._get_models_to_try()
        last_error: Optional[Exception] = None

        for idx, (model_name, llm) in enumerate(models):
            is_last = idx == len(models) - 1
            try:
                response = llm.complete(prompt, **kwargs)
                if idx > 0:
                    logger.info(f"Fallback model {model_name} succeeded")
                self._record_usage(prompt, response.text, response, used_model=model_name)
                return response
            except Exception as e:
                last_error = e
                if not is_retryable_error(e) or is_last:
                    if is_last and is_retryable_error(e):
                        return self._make_graceful_completion_response(e)
                    raise
                # Check for short retry delay
                delay = self._handle_retryable_error(e, model_name, is_last)
                if delay is not None:
                    time.sleep(delay)
                    try:
                        response = llm.complete(prompt, **kwargs)
                        self._record_usage(prompt, response.text, response, used_model=model_name)
                        return response
                    except Exception as retry_err:
                        last_error = retry_err
                        if is_last:
                            return self._make_graceful_completion_response(retry_err)
                logger.warning(f"{model_name} failed: {e}. Trying next fallback.")

        return self._make_graceful_completion_response(last_error)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion and record usage after completion."""
        # Streaming uses primary model only (fallback not practical for streaming)
        full_text = ""
        for chunk in self.inner_llm.stream_complete(prompt, **kwargs):
            full_text += chunk.delta or ""
            yield chunk

        # Record after stream completes
        self._record_usage(prompt, full_text)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with fallback chain and record usage."""
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

        if not self.fallback_models:
            # Fast path: no fallback configured
            response = self.inner_llm.chat(messages, **kwargs)
            response_text = response.message.content or ""
            self._record_usage(prompt, response_text, response)
            return response

        models = self._get_models_to_try()
        last_error: Optional[Exception] = None

        for idx, (model_name, llm) in enumerate(models):
            is_last = idx == len(models) - 1
            try:
                response = llm.chat(messages, **kwargs)
                if idx > 0:
                    logger.info(f"Fallback model {model_name} succeeded")
                response_text = response.message.content or ""
                self._record_usage(prompt, response_text, response, used_model=model_name)
                return response
            except Exception as e:
                last_error = e
                if not is_retryable_error(e) or is_last:
                    if is_last and is_retryable_error(e):
                        return self._make_graceful_chat_response(e)
                    raise
                # Check for short retry delay
                delay = self._handle_retryable_error(e, model_name, is_last)
                if delay is not None:
                    time.sleep(delay)
                    try:
                        response = llm.chat(messages, **kwargs)
                        response_text = response.message.content or ""
                        self._record_usage(prompt, response_text, response, used_model=model_name)
                        return response
                    except Exception as retry_err:
                        last_error = retry_err
                        if is_last:
                            return self._make_graceful_chat_response(retry_err)
                logger.warning(f"{model_name} failed: {e}. Trying next fallback.")

        return self._make_graceful_chat_response(last_error)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ):
        """Stream chat and record usage after completion."""
        # Streaming uses primary model only (fallback not practical for streaming)
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        full_text = ""

        for chunk in self.inner_llm.stream_chat(messages, **kwargs):
            if hasattr(chunk, 'delta') and chunk.delta:
                full_text += chunk.delta
            yield chunk

        self._record_usage(prompt, full_text)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async complete with fallback chain and record usage."""
        import asyncio

        if not self.fallback_models:
            # Fast path: no fallback configured
            response = await self.inner_llm.acomplete(prompt, **kwargs)
            self._record_usage(prompt, response.text, response)
            return response

        models = self._get_models_to_try()
        last_error: Optional[Exception] = None

        for idx, (model_name, llm) in enumerate(models):
            is_last = idx == len(models) - 1
            try:
                response = await llm.acomplete(prompt, **kwargs)
                if idx > 0:
                    logger.info(f"Fallback model {model_name} succeeded (async)")
                self._record_usage(prompt, response.text, response, used_model=model_name)
                return response
            except Exception as e:
                last_error = e
                if not is_retryable_error(e) or is_last:
                    if is_last and is_retryable_error(e):
                        return self._make_graceful_completion_response(e)
                    raise
                # Check for short retry delay
                delay = self._handle_retryable_error(e, model_name, is_last)
                if delay is not None:
                    await asyncio.sleep(delay)
                    try:
                        response = await llm.acomplete(prompt, **kwargs)
                        self._record_usage(prompt, response.text, response, used_model=model_name)
                        return response
                    except Exception as retry_err:
                        last_error = retry_err
                        if is_last:
                            return self._make_graceful_completion_response(retry_err)
                logger.warning(f"{model_name} failed: {e}. Trying next fallback (async).")

        return self._make_graceful_completion_response(last_error)

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat with fallback chain and record usage."""
        import asyncio

        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

        if not self.fallback_models:
            # Fast path: no fallback configured
            response = await self.inner_llm.achat(messages, **kwargs)
            response_text = response.message.content or ""
            self._record_usage(prompt, response_text, response)
            return response

        models = self._get_models_to_try()
        last_error: Optional[Exception] = None

        for idx, (model_name, llm) in enumerate(models):
            is_last = idx == len(models) - 1
            try:
                response = await llm.achat(messages, **kwargs)
                if idx > 0:
                    logger.info(f"Fallback model {model_name} succeeded (async)")
                response_text = response.message.content or ""
                self._record_usage(prompt, response_text, response, used_model=model_name)
                return response
            except Exception as e:
                last_error = e
                if not is_retryable_error(e) or is_last:
                    if is_last and is_retryable_error(e):
                        return self._make_graceful_chat_response(e)
                    raise
                # Check for short retry delay
                delay = self._handle_retryable_error(e, model_name, is_last)
                if delay is not None:
                    await asyncio.sleep(delay)
                    try:
                        response = await llm.achat(messages, **kwargs)
                        response_text = response.message.content or ""
                        self._record_usage(prompt, response_text, response, used_model=model_name)
                        return response
                    except Exception as retry_err:
                        last_error = retry_err
                        if is_last:
                            return self._make_graceful_chat_response(retry_err)
                logger.warning(f"{model_name} failed: {e}. Trying next fallback (async).")

        return self._make_graceful_chat_response(last_error)


def wrap_llm_with_tracking(
    llm: Any,
    model_name: Optional[str] = None,
    fallback_models: Optional[list[str]] = None,
    llm_factory: Optional[Callable] = None,
) -> TrackedLLM:
    """Wrap an LLM with token tracking and optional fallback chain.

    Args:
        llm: The LLM to wrap
        model_name: Optional model name override for pricing
        fallback_models: Optional list of fallback model names
        llm_factory: Optional callable(model_name) -> LLM for fallback instances

    Returns:
        TrackedLLM wrapper
    """
    return TrackedLLM(
        llm=llm,
        model_name=model_name,
        fallback_models=fallback_models,
        llm_factory=llm_factory,
    )
