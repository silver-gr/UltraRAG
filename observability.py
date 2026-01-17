"""
LangSmith Observability Module for UltraRAG.

This module provides non-invasive observability for the UltraRAG system using LangSmith.
It's opt-in: if LANGSMITH_API_KEY is not set, tracing is disabled automatically.

Environment Variables:
    LANGSMITH_TRACING: Set to "true" to enable tracing
    LANGSMITH_API_KEY: Your LangSmith API key
    LANGSMITH_PROJECT: Project name in LangSmith (default: "ultrarag")

Usage:
    from observability import trace_retrieval, trace_generation, trace_query

    # Decorate your functions
    @trace_retrieval
    def my_retrieval_function(...):
        ...

    # Or use the context manager
    with trace_span("custom-operation"):
        ...
"""

import os
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)

# Type variable for generic function decoration
F = TypeVar('F', bound=Callable[..., Any])

# Check if LangSmith is available and configured
LANGSMITH_AVAILABLE = False
TRACING_ENABLED = False

try:
    from langsmith import traceable, Client
    from langsmith.run_helpers import get_current_run_tree, traceable as ls_traceable
    import langsmith as ls

    # Check if tracing is enabled via environment
    LANGSMITH_AVAILABLE = True
    TRACING_ENABLED = os.getenv("LANGSMITH_TRACING", "").lower() == "true"

    if TRACING_ENABLED:
        # Set default project if not specified
        if not os.getenv("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "ultrarag"

        # Disable compression to avoid multipart decompression errors
        # See: https://github.com/langchain-ai/langsmith-sdk/issues/
        os.environ.setdefault("LANGSMITH_COMPRESSION", "none")

        # Log endpoint info
        endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        compression = os.getenv("LANGSMITH_COMPRESSION", "gzip")
        logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGSMITH_PROJECT')} (endpoint: {endpoint}, compression: {compression})")
    else:
        logger.info("LangSmith available but tracing not enabled (set LANGSMITH_TRACING=true)")

except ImportError:
    logger.debug("LangSmith not installed. Install with: pip install langsmith")
    traceable = None
    ls = None


def noop_decorator(func: F) -> F:
    """No-op decorator when LangSmith is not available."""
    return func


def trace_retrieval(func: F) -> F:
    """
    Decorator to trace retrieval operations.

    Captures:
    - Query string
    - Number of retrieved documents
    - Retrieval latency
    - Source document metadata
    """
    if not TRACING_ENABLED or not LANGSMITH_AVAILABLE:
        return func

    @ls_traceable(run_type="retriever", name=func.__name__)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


def trace_generation(func: F) -> F:
    """
    Decorator to trace LLM generation operations.

    Captures:
    - Input prompt/context
    - Generated response
    - Token usage
    - Model parameters
    """
    if not TRACING_ENABLED or not LANGSMITH_AVAILABLE:
        return func

    @ls_traceable(run_type="llm", name=func.__name__)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


def trace_chain(func: F) -> F:
    """
    Decorator to trace chain/pipeline operations.

    Use this for multi-step operations like:
    - Full RAG pipeline (retrieval + generation)
    - Research mode iterations
    - Self-correction loops
    """
    if not TRACING_ENABLED or not LANGSMITH_AVAILABLE:
        return func

    @ls_traceable(run_type="chain", name=func.__name__)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


def trace_tool(func: F) -> F:
    """
    Decorator to trace tool operations.

    Use this for:
    - Embedding generation
    - Reranking
    - Query transformation
    """
    if not TRACING_ENABLED or not LANGSMITH_AVAILABLE:
        return func

    @ls_traceable(run_type="tool", name=func.__name__)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return wrapper


def trace_query(
    query_str: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    Decorator factory for tracing queries with custom metadata.

    Args:
        query_str: The query being executed
        metadata: Additional metadata to log

    Usage:
        @trace_query("user query here", {"mode": "research"})
        def my_query_function():
            ...
    """
    def decorator(func: F) -> F:
        if not TRACING_ENABLED or not LANGSMITH_AVAILABLE:
            return func

        @ls_traceable(
            run_type="chain",
            name=func.__name__,
            metadata={"query": query_str, **(metadata or {})}
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    return decorator


@contextmanager
def trace_span(
    name: str,
    run_type: str = "chain",
    metadata: Optional[Dict[str, Any]] = None,
    inputs: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracing arbitrary spans.

    Args:
        name: Name of the span
        run_type: Type of run (chain, retriever, llm, tool)
        metadata: Additional metadata
        inputs: Input data to log

    Usage:
        with trace_span("research-iteration", metadata={"iteration": 1}):
            # do work
            pass
    """
    if not TRACING_ENABLED or not LANGSMITH_AVAILABLE:
        yield None
        return

    with ls.tracing_context(enabled=True):
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Span '{name}' completed in {duration:.2f}s")


class UltraRAGTracer:
    """
    High-level tracer for UltraRAG operations.

    Provides structured tracing for the main RAG pipeline components.
    """

    def __init__(self, project_name: str = "ultrarag"):
        """
        Initialize the tracer.

        Args:
            project_name: LangSmith project name
        """
        self.project_name = project_name
        self.enabled = TRACING_ENABLED and LANGSMITH_AVAILABLE
        self._client = None

        if self.enabled:
            os.environ["LANGSMITH_PROJECT"] = project_name
            try:
                self._client = Client()
                logger.info(f"UltraRAG Tracer initialized for project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {e}")
                self.enabled = False

    @property
    def client(self) -> Optional['Client']:
        """Get the LangSmith client."""
        return self._client

    def trace_ingestion(self, func: F) -> F:
        """Trace document ingestion operations."""
        return trace_tool(func) if self.enabled else func

    def trace_chunking(self, func: F) -> F:
        """Trace chunking operations."""
        return trace_tool(func) if self.enabled else func

    def trace_embedding(self, func: F) -> F:
        """Trace embedding generation."""
        return trace_tool(func) if self.enabled else func

    def trace_retrieval(self, func: F) -> F:
        """Trace retrieval operations."""
        return trace_retrieval(func) if self.enabled else func

    def trace_reranking(self, func: F) -> F:
        """Trace reranking operations."""
        return trace_tool(func) if self.enabled else func

    def trace_generation(self, func: F) -> F:
        """Trace LLM generation."""
        return trace_generation(func) if self.enabled else func

    def trace_research_mode(self, func: F) -> F:
        """Trace research mode iterations."""
        return trace_chain(func) if self.enabled else func

    def trace_self_correction(self, func: F) -> F:
        """Trace self-correction loops."""
        return trace_chain(func) if self.enabled else func

    def log_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        comment: Optional[str] = None
    ) -> bool:
        """
        Log feedback for a traced run.

        Args:
            run_id: The run ID to attach feedback to
            key: Feedback key (e.g., "correctness", "helpfulness")
            score: Score value (0-1)
            comment: Optional comment

        Returns:
            True if feedback was logged successfully
        """
        if not self.enabled or not self._client:
            return False

        try:
            self._client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment
            )
            logger.debug(f"Logged feedback for run {run_id}: {key}={score}")
            return True
        except Exception as e:
            logger.warning(f"Failed to log feedback: {e}")
            return False

    def get_recent_runs(
        self,
        limit: int = 10,
        error_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get recent traced runs.

        Args:
            limit: Maximum number of runs to return
            error_only: Only return runs with errors

        Returns:
            List of run dictionaries
        """
        if not self.enabled or not self._client:
            return []

        try:
            runs = self._client.list_runs(
                project_name=self.project_name,
                execution_order=1,
                error=error_only if error_only else None,
                limit=limit
            )
            return [
                {
                    "id": str(run.id),
                    "name": run.name,
                    "status": run.status,
                    "start_time": run.start_time,
                    "end_time": run.end_time,
                    "error": run.error,
                    "total_tokens": getattr(run, 'total_tokens', None),
                    "total_cost": getattr(run, 'total_cost', None),
                }
                for run in runs
            ]
        except Exception as e:
            logger.warning(f"Failed to get recent runs: {e}")
            return []


# Global tracer instance (lazy initialization)
_tracer: Optional[UltraRAGTracer] = None


def get_tracer(project_name: str = "ultrarag") -> UltraRAGTracer:
    """
    Get or create the global tracer instance.

    Args:
        project_name: LangSmith project name

    Returns:
        UltraRAGTracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = UltraRAGTracer(project_name)
    return _tracer


# Convenience functions for quick tracing
def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return TRACING_ENABLED and LANGSMITH_AVAILABLE


def get_tracing_status() -> Dict[str, Any]:
    """Get current tracing status and configuration."""
    return {
        "langsmith_available": LANGSMITH_AVAILABLE,
        "tracing_enabled": TRACING_ENABLED,
        "project": os.getenv("LANGSMITH_PROJECT", "ultrarag"),
        "api_key_set": bool(os.getenv("LANGSMITH_API_KEY")),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    }
