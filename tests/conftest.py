"""Pytest configuration and shared fixtures."""
import pytest
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock
from llama_index.core.embeddings import BaseEmbedding


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
