"""Tests for the interactive UltraRAG orchestration entry point."""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("prefix", ["@all ", "!all "])
def test_query_research_exhaustive_prefix_forces_exhaustive_research(prefix):
    from main import UltraRAG
    from llama_index.core import Settings

    rag = object.__new__(UltraRAG)
    rag.query_engine = object()
    rag.index = object()
    rag.query_transformer = None
    rag.llm = object()
    rag.config = SimpleNamespace(
        retrieval=SimpleNamespace(
            top_k=1,
            query_transform_method="none",
            research_max_iterations=3,
            research_confidence_threshold=0.8,
            research_max_subqueries=3,
            enable_research_mode=True,
            research_max_synthesis_sources=0,
        ),
        llm=SimpleNamespace(backend="api"),
    )

    research_result = MagicMock()
    research_result.final_nodes = []
    research_result.total_iterations = 5
    research_result.total_nodes_retrieved = 0
    research_result.final_confidence = 1.0
    research_result.get_iteration_summary.return_value = []
    research_result.get_gap_analyses.return_value = []
    research_result.get_gap_analyses_markdown.return_value = ""
    response = SimpleNamespace(text="answer")

    with (
        patch("research_mode.ResearchRetriever") as retriever_class,
        patch("llama_index.core.retrievers.VectorIndexRetriever", return_value=object()),
        patch("main._build_chunk_prompt", return_value="prompt"),
        patch("research_mode.llm_complete_with_retry", return_value=response),
        patch("citations.validate_citations", side_effect=lambda answer, _: answer),
        patch.object(type(Settings), "llm", new_callable=PropertyMock) as settings_llm,
    ):
        settings_llm.return_value = object()
        retriever_class.return_value.research.return_value = research_result
        result = rag.query_research(f"{prefix}comprehensive query", return_sources=False)

    retriever_class.return_value.research.assert_called_once_with(
        "comprehensive query", force_exhaustive=True
    )
    assert result["answer"] == "answer"
