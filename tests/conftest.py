"""
Shared pytest fixtures for UltraRAG tests.

Agent note:
- All fixtures use test-specific paths (tests/fixtures/)
- Mock fixtures avoid API calls for fast unit tests
- Integration tests use @pytest.mark.integration
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_vault() -> Generator[Path, None, None]:
    """Create a temporary vault directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)
        yield vault_path


@pytest.fixture
def sample_note_content() -> str:
    """Sample markdown content with frontmatter."""
    return """---
title: Test Note
tags: [project, daily-notes]
created: 2024-01-01
---

# Test Note

This is a test note with [[Note1]] and [[Note2|alias]].

## Section 1

Some content here with #tag1 and #tag2.

```python
def hello():
    print("Hello World")
```

See also [[Another Note]].
"""


@pytest.fixture
def sample_note_no_frontmatter() -> str:
    """Sample markdown content without frontmatter."""
    return """# Simple Note

This note has [[Link1]] and tags: #simple #test

Another paragraph with more content.
"""


@pytest.fixture
def create_test_note(temp_vault: Path):
    """Factory fixture to create test notes."""
    def _create_note(filename: str, content: str) -> Path:
        note_path = temp_vault / filename
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(content, encoding='utf-8')
        return note_path
    return _create_note


@pytest.fixture
def mock_embed_model() -> Mock:
    """Create a mock embedding model."""
    # Don't use spec= as it prevents adding attributes not on BaseEmbedding
    mock = Mock()
    # Mock the embed methods to return dummy embeddings
    mock.get_text_embedding.return_value = [0.1] * 1024
    mock.get_text_embeddings.return_value = [[0.1] * 1024]
    mock._get_text_embedding.return_value = [0.1] * 1024
    mock._get_text_embeddings.return_value = [[0.1] * 1024]
    return mock


@pytest.fixture
def sample_wikilink_content() -> str:
    """Sample content with various wikilink formats."""
    return """
[[Simple Link]]
[[Link with|Alias]]
[[Link with spaces]]
[[nested/path/Note]]
[[Link]] and [[Another Link]]
"""


@pytest.fixture
def sample_tag_content() -> str:
    """Sample content with various tag formats."""
    return """
#simple-tag
#nested/tag/structure
#project/work
#daily-notes
Tags: #inline #multiple
"""


@pytest.fixture
def sample_code_block_content() -> str:
    """Sample content with code blocks."""
    return """# Code Example

Here's some Python code:

```python
def calculate(x, y):
    return x + y
```

And some JavaScript:

```javascript
function hello() {
    console.log("Hello");
}
```

Regular text here.
"""


# === NEW FIXTURES FOR AI-OPTIMIZED MODULES ===

@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_vault_path(fixtures_path) -> Path:
    """Path to sample vault with 3 test notes."""
    return fixtures_path / "sample_vault"


@pytest.fixture
def temp_lancedb_path(tmp_path) -> Path:
    """Temporary LanceDB path for isolated tests."""
    return tmp_path / "test_lancedb"


@pytest.fixture
def test_config(sample_vault_path, temp_lancedb_path):
    """
    Test configuration with safe defaults.

    Agent note: Uses test paths to avoid touching real data.
    """
    from config import RAGConfig, EmbeddingConfig, VectorDBConfig

    return RAGConfig(
        vault_path=sample_vault_path,
        embedding=EmbeddingConfig(
            model="voyage-4-lite",
            chunk_size=256,  # Smaller for tests
            chunk_overlap=25,
        ),
        vector_db=VectorDBConfig(
            db_type="lancedb",
            lancedb_path=temp_lancedb_path,
        ),
    )


@pytest.fixture
def mock_embedding_model():
    """
    Mock embedding model that returns deterministic vectors.

    Agent note: Use this for unit tests to avoid Voyage API calls.
    """
    mock = MagicMock()
    # Return 1024-dim vector (Voyage default)
    mock.get_text_embedding.return_value = [0.1] * 1024
    mock.get_text_embedding_batch.return_value = [[0.1] * 1024]
    mock._get_text_embeddings.return_value = [[0.1] * 1024]
    mock.embed_dim = 1024
    return mock


@pytest.fixture
def mock_llm():
    """
    Mock LLM that returns canned responses.

    Agent note: Use for unit tests. Integration tests use real LLM.
    """
    mock = MagicMock()
    mock.complete.return_value = MagicMock(text="This is a test response about RAG systems.")
    mock.chat.return_value = MagicMock(message=MagicMock(content="Test chat response"))
    return mock


@pytest.fixture
def mock_llm_response():
    """
    Canned LLM response for deterministic tests.

    Agent note: Edit to change expected test outputs.
    """
    return {
        "text": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation.",
        "tokens_used": 150,
        "model": "gemini-3-flash-preview"
    }


@pytest.fixture
def sample_documents(sample_vault_path):
    """
    Load sample documents from test vault.

    Agent note: Returns 3 test notes with wikilinks and tags.
    """
    from llama_index.core import Document

    docs = []
    for md_file in sample_vault_path.glob("*.md"):
        content = md_file.read_text()
        docs.append(Document(
            text=content,
            metadata={
                "file_path": str(md_file),
                "file_name": md_file.name,
            }
        ))
    return docs


@pytest.fixture
def sample_nodes(sample_documents, mock_embedding_model):
    """
    Pre-chunked nodes from sample documents.

    Agent note: Use for retrieval tests that don't need to test chunking.
    """
    from llama_index.core.schema import TextNode

    nodes = []
    for i, doc in enumerate(sample_documents):
        node = TextNode(
            text=doc.text[:500],  # First 500 chars
            metadata=doc.metadata,
            embedding=[0.1 + i * 0.01] * 1024,  # Slightly different embeddings
        )
        nodes.append(node)
    return nodes


# === HELPER FUNCTIONS ===

def create_mock_query_result(answer: str = "Test answer", num_sources: int = 3):
    """Helper to create mock QueryResult for tests."""
    from models import QueryResult, SourceNode

    sources = [
        SourceNode(
            file_path=f"/test/note_{i}.md",
            file_name=f"note_{i}.md",
            excerpt=f"Excerpt from note {i}",
            score=0.9 - i * 0.1,
        )
        for i in range(num_sources)
    ]

    return QueryResult(
        answer=answer,
        sources=sources,
        tokens_used=100,
        exec_time=0.5,
        query="test query",
        mode="hybrid",
    )


# === MARKERS ===

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast tests, no external API calls")
    config.addinivalue_line("markers", "integration: Requires LLM/embedding APIs")
    config.addinivalue_line("markers", "slow: Takes >10 seconds")
