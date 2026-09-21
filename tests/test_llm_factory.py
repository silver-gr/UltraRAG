"""Tests for the centralized LLM factory."""

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_openai_backend_uses_extended_request_timeout():
    from llm_factory import create_llm

    with patch("llama_index.llms.openai_like.OpenAILike") as openai_like:
        result = create_llm(
            backend="openai",
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:4000/v1",
        )

    assert result is openai_like.return_value
    assert openai_like.call_args.kwargs["timeout"] == 600
