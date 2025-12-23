"""Custom LLM wrapper for Gemini CLI.

This module provides a LlamaIndex-compatible LLM that uses the Gemini CLI
instead of the API, leveraging the CLI's separate free tier quota.
"""
import json
import logging
import subprocess
import shutil
from typing import Any, Optional, Sequence

from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage,
    ChatResponse,
)
from llama_index.core.llms.callbacks import llm_completion_callback

logger = logging.getLogger(__name__)


class GeminiCLI(CustomLLM):
    """LlamaIndex-compatible LLM that uses the Gemini CLI.

    This wrapper calls the `gemini` CLI tool instead of the API directly,
    which has its own free tier quota (1000 requests/day, 60/min).

    Attributes:
        model: The Gemini model to use (e.g., "gemini-3-flash-preview")
        context_window: Maximum context window size
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0-2.0)
        timeout: CLI command timeout in seconds
    """

    model: str = "gemini-3-flash-preview"
    context_window: int = 1000000  # Gemini's 1M context
    max_tokens: int = 8192
    temperature: float = 0.1
    timeout: int = 120  # 2 minute timeout

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        timeout: int = 120,
        **kwargs: Any,
    ):
        """Initialize the Gemini CLI wrapper.

        Args:
            model: Gemini model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: CLI command timeout in seconds
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs,
        )

        # Verify gemini CLI is available
        if not shutil.which("gemini"):
            raise RuntimeError(
                "Gemini CLI not found. Install it with: npm install -g @google/gemini-cli\n"
                "Then authenticate with: gemini"
            )

        logger.info(f"Initialized GeminiCLI with model={model}, temp={temperature}")

    @property
    def metadata(self) -> LLMMetadata:
        """Return LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    def _run_gemini(self, prompt: str) -> str:
        """Execute gemini CLI and return response text.

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            The response text from Gemini

        Raises:
            RuntimeError: If CLI execution fails
        """
        # Build command
        cmd = [
            "gemini",
            prompt,
            "--yolo",  # Auto-approve tool calls
            "-o", "json",  # JSON output for parsing
            "-m", self.model,
        ]

        logger.debug(f"Running gemini CLI: {' '.join(cmd[:3])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(f"Gemini CLI error: {error_msg}")
                raise RuntimeError(f"Gemini CLI failed: {error_msg}")

            # Parse JSON output
            output = result.stdout.strip()

            # Find the JSON object (skip any startup logs)
            json_start = output.find("{")
            if json_start == -1:
                # No JSON found, return raw output
                logger.warning("No JSON in output, returning raw text")
                return output

            json_str = output[json_start:]

            try:
                data = json.loads(json_str)
                response_text = data.get("response", "")

                # Log token usage for monitoring
                stats = data.get("stats", {})
                models = stats.get("models", {})
                for model_name, model_stats in models.items():
                    tokens = model_stats.get("tokens", {})
                    logger.debug(
                        f"Token usage ({model_name}): "
                        f"prompt={tokens.get('prompt', 0)}, "
                        f"output={tokens.get('candidates', 0)}"
                    )

                return response_text

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}, returning raw output")
                return output

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Gemini CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError("Gemini CLI not found in PATH")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete a prompt using Gemini CLI.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments (ignored)

        Returns:
            CompletionResponse with the generated text
        """
        response_text = self._run_gemini(prompt)
        return CompletionResponse(text=response_text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion (not truly streaming, returns full response).

        The CLI doesn't support true streaming to stdout easily,
        so this returns the full response as a single chunk.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional arguments (ignored)

        Yields:
            CompletionResponse with the full text
        """
        response_text = self._run_gemini(prompt)
        yield CompletionResponse(text=response_text, delta=response_text)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the model using message history.

        Converts chat messages to a single prompt for the CLI.

        Args:
            messages: Sequence of chat messages
            **kwargs: Additional arguments

        Returns:
            ChatResponse with the assistant's reply
        """
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            prompt_parts.append(f"[{role.upper()}]: {msg.content}")

        prompt = "\n\n".join(prompt_parts)
        prompt += "\n\n[ASSISTANT]:"

        response_text = self._run_gemini(prompt)

        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text)
        )

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async complete (runs sync in executor).

        Note: This runs the sync version since CLI is blocking.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        """Async chat (runs sync in executor)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))


def get_gemini_cli(
    model: str = "gemini-3-flash-preview",
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> GeminiCLI:
    """Factory function to create a GeminiCLI instance.

    Args:
        model: Gemini model to use
        temperature: Sampling temperature
        max_tokens: Maximum output tokens

    Returns:
        Configured GeminiCLI instance
    """
    return GeminiCLI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
