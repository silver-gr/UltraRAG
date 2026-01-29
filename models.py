"""
UltraRAG Type Definitions

All dataclasses, protocols, and type aliases in one place.
Agent note: Import from here, not from individual modules.

IMPORTANT: Named models.py (not types.py) to avoid conflict with Python built-in types module.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Literal, Any
from datetime import datetime


# === EXCEPTIONS (Agent-friendly with fix hints) ===

class UltraRAGError(Exception):
    """Base exception with actionable context for agents."""
    def __init__(self, message: str, fix_hint: str, related_file: str | None = None):
        self.message = message
        self.fix_hint = fix_hint
        self.related_file = related_file
        super().__init__(f"{message}\n💡 Fix: {fix_hint}")


class IndexNotFoundError(UltraRAGError):
    """Index doesn't exist - needs indexing first."""
    def __init__(self, source: str):
        super().__init__(
            message=f"Index '{source}' not found at expected path",
            fix_hint=f"Run index_{source}() first, or check VAULT_PATH in .env",
            related_file="indexing.py"
        )


class EmbeddingQuotaError(UltraRAGError):
    """Voyage API quota exceeded."""
    def __init__(self, used: int, limit: int):
        super().__init__(
            message=f"Voyage quota exceeded: {used:,}/{limit:,} tokens",
            fix_hint="Wait for monthly reset or use cached embeddings (re-run won't re-embed)",
            related_file="data/voyage_usage.json"
        )


class LLMRateLimitError(UltraRAGError):
    """Gemini rate limit hit."""
    def __init__(self, retry_after: int | None = None):
        hint = f"Retry after {retry_after}s" if retry_after else "Set LLM_BACKEND=cli in .env for separate quota"
        super().__init__(
            message="Gemini API rate limit exceeded (429)",
            fix_hint=hint,
            related_file=".env"
        )


class ConfigurationError(UltraRAGError):
    """Missing or invalid configuration."""
    def __init__(self, key: str, expected: str):
        super().__init__(
            message=f"Configuration error: {key} is missing or invalid",
            fix_hint=f"Add {key}={expected} to .env file",
            related_file=".env"
        )


# === ENUMS ===

class IndexProfile(Enum):
    """Convergence profile for research mode."""
    PERSONAL = "personal"      # Quick convergence for personal notes
    RESEARCH = "research"      # Thorough for research documents
    BALANCED = "balanced"      # Default balance


class SourceType(Enum):
    """Types of content sources."""
    VAULT = "vault"
    CONVERSATIONS = "conversations"
    BOOKS = "books"
    WEB = "web"


# === CORE DATACLASSES ===

@dataclass
class NodeMetadata:
    """Metadata attached to each text node."""
    file_path: str
    file_name: str
    source_type: str = "vault"
    title: str = ""
    created_date: datetime | None = None
    modified_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceNode:
    """A source reference in query results."""
    file_path: str
    file_name: str
    excerpt: str
    score: float
    source_type: str = "vault"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Result from a query operation."""
    answer: str
    sources: list[SourceNode]
    tokens_used: int
    exec_time: float
    query: str = ""
    mode: str = "hybrid"


@dataclass
class GapAnalysis:
    """Gap analysis from research iteration."""
    iteration: int
    gaps_found: list[str]
    subqueries_generated: list[str]
    confidence: float
    new_nodes_count: int


@dataclass
class ResearchResult(QueryResult):
    """Extended result from research mode."""
    iterations: int = 0
    gaps_analyzed: list[GapAnalysis] = field(default_factory=list)
    confidence: float = 0.0
    is_exhaustive: bool = False
    total_unique_nodes: int = 0

    def get_gap_analyses_markdown(self) -> str:
        """Format gap analyses as markdown for display."""
        if not self.gaps_analyzed:
            return ""

        lines = ["## Research Analysis\n"]
        for gap in self.gaps_analyzed:
            lines.append(f"### Iteration {gap.iteration}")
            lines.append(f"- Confidence: {gap.confidence:.1%}")
            lines.append(f"- New nodes: {gap.new_nodes_count}")
            if gap.gaps_found:
                lines.append(f"- Gaps: {', '.join(gap.gaps_found)}")
            lines.append("")
        return "\n".join(lines)


@dataclass
class ConvergenceConfig:
    """Configuration for research convergence detection."""
    min_gain_threshold: float = 0.05
    confidence_threshold: float = 0.8
    max_iterations: int = 3
    enable_exhaustive: bool = True

    @classmethod
    def for_profile(cls, profile: IndexProfile) -> "ConvergenceConfig":
        """Get config for a specific profile."""
        configs = {
            IndexProfile.PERSONAL: cls(min_gain_threshold=0.1, max_iterations=2),
            IndexProfile.RESEARCH: cls(min_gain_threshold=0.03, max_iterations=5),
            IndexProfile.BALANCED: cls(),
        }
        return configs.get(profile, cls())


@dataclass
class IndexSource:
    """A source index for federated queries."""
    name: str
    index: Any  # VectorStoreIndex
    source_type: SourceType
    weight: float = 1.0
    nodes: list[Any] | None = None
    wikilink_graph: dict[str, list[str]] | None = None


# === PROTOCOLS (for type hints) ===

class Retriever(Protocol):
    """Protocol for retriever implementations."""
    def retrieve(self, query: str) -> list[Any]: ...


class QueryEngine(Protocol):
    """Protocol for query engine implementations."""
    def query(self, query_str: str, **kwargs) -> QueryResult: ...


class IndexManager(Protocol):
    """Protocol for index management."""
    def index(self) -> Any: ...
    def load(self) -> Any | None: ...
    def exists(self) -> bool: ...


# === TYPE ALIASES ===

WikilinkGraph = dict[str, list[str]]
NodeList = list[Any]  # List of TextNode
SourceFilter = Literal["vault", "conversations", "books", "all"] | None
QueryMode = Literal["simple", "hybrid", "research"]
