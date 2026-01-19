"""Tracked LLM wrapper that records token usage for all LLM calls.

Wraps any LlamaIndex LLM and intercepts calls to record token usage
to the LLMTokenTracker.
"""
from __future__ import annotations

import logging
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


class TrackedLLM(CustomLLM):
    """LLM wrapper that tracks token usage for all calls.

    Wraps any LlamaIndex-compatible LLM and records input/output tokens
    to the LLMTokenTracker for cost tracking.

    Attributes:
        inner_llm: The underlying LLM to wrap
        model_name: Model name for pricing (extracted from inner LLM)
        tracker: The token tracker instance
    """

    inner_llm: Any = None
    model_name: str = "gemini-3-flash-preview"
    tracker: Optional[LLMTokenTracker] = None

    def __init__(
        self,
        llm: Any,
        model_name: Optional[str] = None,
        tracker: Optional[LLMTokenTracker] = None,
        **kwargs: Any,
    ):
        """Initialize tracked LLM wrapper.

        Args:
            llm: The underlying LLM to wrap
            model_name: Override model name for pricing (default: extract from llm)
            tracker: Token tracker instance (default: global tracker)
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
            **kwargs,
        )

        logger.info(f"Initialized TrackedLLM wrapping {model_name}")

    @property
    def metadata(self) -> LLMMetadata:
        """Return metadata from inner LLM."""
        return self.inner_llm.metadata

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (fallback when actual count unavailable)."""
        # Gemini uses roughly 4 chars per token for English
        return max(1, len(text) // 4)

    def _extract_token_counts(self, response: Any) -> tuple[int, int]:
        """Extract token counts from response if available.

        Returns:
            (input_tokens, output_tokens) tuple
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
                    if input_tokens > 0 or output_tokens > 0:
                        logger.debug(f"Extracted actual tokens: {input_tokens} in, {output_tokens} out")
                        return input_tokens, output_tokens

            # Check for additional_kwargs
            additional = getattr(response, 'additional_kwargs', {})
            if additional:
                usage = additional.get('usage', {}) or additional.get('usage_metadata', {})
                if usage:
                    input_tokens = usage.get('prompt_tokens', 0) or usage.get('prompt_token_count', 0) or 0
                    output_tokens = usage.get('completion_tokens', 0) or usage.get('candidates_token_count', 0) or 0
                    if input_tokens > 0 or output_tokens > 0:
                        return input_tokens, output_tokens

        except Exception as e:
            logger.debug(f"Could not extract token counts: {e}")

        return 0, 0

    def _record_usage(
        self,
        prompt: str,
        response_text: str,
        response: Any = None
    ) -> None:
        """Record token usage for a call."""
        # Try to extract actual token counts
        input_tokens, output_tokens = self._extract_token_counts(response)

        # Fall back to estimates if no actual counts
        if input_tokens == 0:
            input_tokens = self._estimate_tokens(prompt)
        if output_tokens == 0:
            output_tokens = self._estimate_tokens(response_text)

        self.tracker.record_usage(input_tokens, output_tokens, self.model_name)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt and record usage."""
        response = self.inner_llm.complete(prompt, **kwargs)
        self._record_usage(prompt, response.text, response)
        return response

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion and record usage after completion."""
        # Collect full response for token counting
        full_text = ""
        for chunk in self.inner_llm.stream_complete(prompt, **kwargs):
            full_text += chunk.delta or ""
            yield chunk

        # Record after stream completes
        self._record_usage(prompt, full_text)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat and record usage."""
        response = self.inner_llm.chat(messages, **kwargs)

        # Build prompt from messages for estimation
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        response_text = response.message.content or ""

        self._record_usage(prompt, response_text, response)
        return response

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ):
        """Stream chat and record usage after completion."""
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        full_text = ""

        for chunk in self.inner_llm.stream_chat(messages, **kwargs):
            if hasattr(chunk, 'delta') and chunk.delta:
                full_text += chunk.delta
            yield chunk

        self._record_usage(prompt, full_text)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async complete and record usage."""
        response = await self.inner_llm.acomplete(prompt, **kwargs)
        self._record_usage(prompt, response.text, response)
        return response

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat and record usage."""
        response = await self.inner_llm.achat(messages, **kwargs)

        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        response_text = response.message.content or ""

        self._record_usage(prompt, response_text, response)
        return response


def wrap_llm_with_tracking(
    llm: Any,
    model_name: Optional[str] = None
) -> TrackedLLM:
    """Wrap an LLM with token tracking.

    Args:
        llm: The LLM to wrap
        model_name: Optional model name override for pricing

    Returns:
        TrackedLLM wrapper
    """
    return TrackedLLM(llm=llm, model_name=model_name)
