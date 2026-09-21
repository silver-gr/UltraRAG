"""Central factory for constructing raw (untracked) LLM instances.

All runtime LLM construction routes through here so a single place honors the
configured backend, an optional OpenAI-compatible proxy endpoint, and the
thinking/reasoning effort setting.

Backends
--------
- ``api``    : Google Gemini native (llama_index ``GoogleGenAI``).
- ``cli``    : Gemini CLI (separate free-tier quota, no proxy/effort support).
- ``openai`` : OpenAI-compatible proxy (llama_index ``OpenAILike`` pointed at
               ``LLM_BASE_URL``). Use this to reach Gemini through your own
               OpenAI-style gateway (e.g. LiteLLM). ``OpenAILike`` accepts
               arbitrary (non-OpenAI) model names.

Thinking effort
---------------
``thinking_effort`` accepts ``""`` (leave backend default), ``none``,
``minimal``, ``low``, ``medium`` or ``high``.

- ``openai`` backend: forwarded verbatim as ``reasoning_effort`` via
  ``additional_kwargs`` (the proxy decides how to apply it). Routed through
  ``additional_kwargs`` rather than the top-level param because llama_index only
  forwards the native ``reasoning_effort`` for OpenAI o-series model names.
- ``api`` backend: mapped to ``google.genai`` ``ThinkingConfig``. Only applied
  when non-empty, so the default leaves native behavior untouched.

The functions here return the *raw* LLM. Callers are responsible for wrapping
with token tracking (``tracked_llm.wrap_llm_with_tracking``) when desired.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Effort values accepted from config/env (validated in config.py too).
VALID_EFFORTS = {"", "none", "minimal", "low", "medium", "high"}

# Gemini context windows are large; give the OpenAI client a generous window so
# it does not prematurely truncate context for a non-OpenAI model name.
_OPENAI_CONTEXT_WINDOW = 1_000_000
_OPENAI_REQUEST_TIMEOUT = 600


def _build_thinking_config(effort: str):
    """Map an effort string to a google.genai GenerateContentConfig thinking setup.

    Returns None when effort is empty or the SDK is unavailable, so the native
    backend keeps its default behavior.
    """
    if not effort:
        return None
    try:
        from google.genai import types
    except ImportError:
        logger.warning("google.genai types unavailable; ignoring thinking_effort=%r", effort)
        return None

    effort = effort.lower()
    if effort in ("none", "minimal"):
        thinking = types.ThinkingConfig(thinking_budget=0)
    elif effort == "low":
        thinking = types.ThinkingConfig(thinking_level="low")
    else:  # medium / high
        thinking = types.ThinkingConfig(thinking_level="high")
    return thinking


def create_llm(
    *,
    backend: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
    thinking_effort: str = "",
    temperature: float = 0.1,
    max_tokens: int = 65536,
    is_function_calling_model: bool = True,
) -> Any:
    """Construct a raw LLM instance for the given backend.

    See module docstring for backend/effort semantics. Raises ValueError when a
    required credential/endpoint is missing.
    """
    backend = (backend or "api").lower()

    if backend == "cli":
        from gemini_cli import GeminiCLI

        return GeminiCLI(model=model, temperature=temperature, max_tokens=max_tokens)

    if backend == "openai":
        # OpenAILike (not OpenAI) so non-OpenAI model names (e.g. gemini-*) are
        # accepted instead of being validated against OpenAI's model list.
        from llama_index.llms.openai_like import OpenAILike

        if not base_url:
            raise ValueError(
                "LLM_BACKEND=openai requires LLM_BASE_URL (your OpenAI-compatible endpoint, "
                "e.g. http://localhost:4000/v1)."
            )
        if not api_key:
            raise ValueError(
                "LLM_BACKEND=openai requires an API key. Set LLM_API_KEY (or GOOGLE_API_KEY) "
                "to your proxy's key."
            )

        additional_kwargs: dict[str, Any] = {}
        if thinking_effort:
            # Forwarded verbatim; the proxy maps it to the underlying model.
            additional_kwargs["reasoning_effort"] = thinking_effort

        return OpenAILike(
            model=model,
            api_key=api_key,
            api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=_OPENAI_CONTEXT_WINDOW,
            timeout=_OPENAI_REQUEST_TIMEOUT,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_model,
            additional_kwargs=additional_kwargs,
        )

    # Default: native Google Gemini API.
    from llama_index.llms.google_genai import GoogleGenAI

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it in your .env file.\n"
            "Get your API key from: https://makersuite.google.com/app/apikey"
        )

    generation_config = None
    thinking = _build_thinking_config(thinking_effort)
    if thinking is not None:
        # Passing generation_config replaces GoogleGenAI's internal temperature/
        # max_tokens handling, so build the full config here.
        try:
            from google.genai import types

            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                thinking_config=thinking,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Failed to build thinking generation_config (%s); using defaults", e)
            generation_config = None

    return GoogleGenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        is_function_calling_model=is_function_calling_model,
        generation_config=generation_config,
    )


def _resolve_api_key_from_env() -> str:
    return os.getenv("LLM_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def create_llm_from_config(config, **overrides: Any) -> Any:
    """Create a raw LLM from a RAGConfig, honoring backend/proxy/effort.

    ``overrides`` may set ``model``, ``temperature``, ``max_tokens``,
    ``is_function_calling_model``, ``thinking_effort`` to specialize a sub-LLM
    (e.g. HyDE, gap analysis).
    """
    llm_cfg = config.llm
    api_key = ""
    cfg_key = getattr(llm_cfg, "api_key", None)
    if cfg_key is not None and cfg_key.get_secret_value():
        api_key = cfg_key.get_secret_value()
    if not api_key and getattr(config, "google_api_key", None) is not None:
        api_key = config.google_api_key.get_secret_value()
    if not api_key:
        api_key = _resolve_api_key_from_env()

    params = dict(
        backend=llm_cfg.backend,
        model=llm_cfg.model,
        api_key=api_key,
        base_url=getattr(llm_cfg, "base_url", "") or "",
        thinking_effort=getattr(llm_cfg, "thinking_effort", "") or "",
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
    )
    params.update(overrides)
    return create_llm(**params)


def create_llm_from_env(model: str, **overrides: Any) -> Any:
    """Create a raw LLM using environment variables only (no RAGConfig handle).

    Used by the token-tracking fallback path, which does not carry a config
    object. Reads the same LLM_* variables as config.load_config.
    """
    params = dict(
        backend=os.getenv("LLM_BACKEND", "api"),
        model=model,
        api_key=_resolve_api_key_from_env(),
        base_url=os.getenv("LLM_BASE_URL", ""),
        thinking_effort=os.getenv("LLM_THINKING_EFFORT", ""),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "65536")),
    )
    params.update(overrides)
    return create_llm(**params)
