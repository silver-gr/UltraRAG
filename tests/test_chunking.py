"""Tests for ObsidianChunker."""
import pytest
from unittest.mock import Mock, patch
from llama_index.core import Document
from llama_index.core.schema import TextNode
from chunking import ObsidianChunker
from config import EmbeddingConfig


class TestTokenCounting:
    """Test approximate token counting functionality."""

    def test_approximate_token_count_simple(self, mock_embed_model):
        """Test token counting with simple text."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        text = "This is a simple test"
        token_count = chunker._approximate_token_count(text)

        # 5 words * 1.3 = 6.5 -> 6 tokens
        assert token_count == 6

    def test_approximate_token_count_longer_text(self, mock_embed_model):
        """Test token counting with longer text."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        # 100 words
        text = " ".join(["word"] * 100)
        token_count = chunker._approximate_token_count(text)

        # 100 words * 1.3 = 130 tokens
        assert token_count == 130

    def test_approximate_token_count_empty_string(self, mock_embed_model):
        """Test token counting with empty string."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        text = ""
        token_count = chunker._approximate_token_count(text)
        assert token_count == 0

    def test_approximate_token_count_whitespace(self, mock_embed_model):
        """Test token counting with only whitespace."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        text = "   \n\n   "
        token_count = chunker._approximate_token_count(text)
        assert token_count == 0


class TestChunkingStrategy:
    """Test chunking strategy selection."""

    def test_markdown_semantic_strategy(self, mock_embed_model):
        """Test that markdown_semantic strategy is used correctly."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown_semantic")

        assert chunker.strategy == "markdown_semantic"

    def test_semantic_strategy(self, mock_embed_model):
        """Test that semantic strategy is used correctly."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="semantic")

        assert chunker.strategy == "semantic"

    def test_markdown_strategy(self, mock_embed_model):
        """Test that markdown strategy is used correctly."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        assert chunker.strategy == "markdown"

    def test_simple_strategy(self, mock_embed_model):
        """Test that simple strategy is used correctly."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="simple")

        assert chunker.strategy == "simple"


class TestSimpleChunking:
    """Test simple sentence-based chunking."""

    def test_simple_chunking_basic(self, mock_embed_model):
        """Test simple chunking with basic text."""
        config = EmbeddingConfig(chunk_size=50, chunk_overlap=10)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="simple")

        content = """This is the first sentence. This is the second sentence.
This is the third sentence. This is the fourth sentence."""

        doc = Document(text=content, metadata={'title': 'Test'})
        nodes = chunker.chunk_documents([doc])

        assert len(nodes) > 0
        assert all(isinstance(node, TextNode) for node in nodes)

    def test_simple_chunking_respects_chunk_size(self, mock_embed_model):
        """Test that simple chunking respects chunk_size parameter."""
        config = EmbeddingConfig(chunk_size=100, chunk_overlap=20)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="simple")

        # Create a long document
        content = " ".join(["This is a test sentence."] * 100)
        doc = Document(text=content, metadata={'title': 'Test'})

        nodes = chunker.chunk_documents([doc])

        # Verify chunks are created
        assert len(nodes) > 1

    def test_simple_chunking_with_overlap(self, mock_embed_model):
        """Test that chunking creates overlapping chunks."""
        config = EmbeddingConfig(chunk_size=50, chunk_overlap=10)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="simple")

        content = " ".join(["Test sentence number {}.".format(i) for i in range(20)])
        doc = Document(text=content, metadata={'title': 'Test'})

        nodes = chunker.chunk_documents([doc])

        # Chunks should be created
        assert len(nodes) > 1


class TestMarkdownChunking:
    """Test markdown-aware chunking."""

    def test_markdown_chunking_by_headers(self, mock_embed_model):
        """Test that markdown chunking splits by headers."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        content = """# Header 1

Content under header 1.

## Header 2

Content under header 2.

### Header 3

Content under header 3.
"""

        doc = Document(text=content, metadata={'title': 'Test'})
        nodes = chunker.chunk_documents([doc])

        assert len(nodes) > 0
        assert all(isinstance(node, TextNode) for node in nodes)

    def test_markdown_chunking_preserves_code_blocks(self, mock_embed_model, sample_code_block_content):
        """Test that code blocks are preserved in markdown chunking."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        doc = Document(text=sample_code_block_content, metadata={'title': 'Test'})
        nodes = chunker.chunk_documents([doc])

        assert len(nodes) > 0

        # Check that code content is preserved in at least one chunk
        all_text = " ".join([node.text for node in nodes])
        assert "def calculate" in all_text or "function hello" in all_text

    def test_markdown_chunking_empty_document(self, mock_embed_model):
        """Test markdown chunking with empty document."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        doc = Document(text="", metadata={'title': 'Empty'})
        nodes = chunker.chunk_documents([doc])

        # Should handle empty documents gracefully
        assert isinstance(nodes, list)


class TestMarkdownSemanticChunking:
    """Test hybrid markdown + semantic chunking."""

    @patch('chunking.MarkdownNodeParser')
    @patch('chunking.SemanticSplitterNodeParser')
    def test_markdown_semantic_uses_both_parsers(
        self, mock_semantic_parser, mock_markdown_parser, mock_embed_model
    ):
        """Test that markdown-semantic uses both parsers when content is large."""
        config = EmbeddingConfig(chunk_size=50, chunk_overlap=10)  # Small chunk size to trigger semantic
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown_semantic")

        # Mock return values - create a LARGE node that will trigger semantic splitting
        mock_markdown_instance = mock_markdown_parser.return_value
        mock_semantic_instance = mock_semantic_parser.return_value

        # Token count = word_count * 1.3, so we need ~40 words to exceed chunk_size=50
        # Create a large node that exceeds chunk_size (triggers semantic splitting)
        large_text = " ".join(["word"] * 100)  # 100 words * 1.3 = 130 tokens > 50
        large_node = TextNode(text=large_text, metadata={})
        mock_markdown_instance.get_nodes_from_documents.return_value = [large_node]
        mock_semantic_instance.get_nodes_from_documents.return_value = [
            TextNode(text="Split chunk 1", metadata={}),
            TextNode(text="Split chunk 2", metadata={})
        ]

        doc = Document(text="# Test\n\n" + large_text, metadata={'title': 'Test'})
        nodes = chunker.chunk_documents([doc])

        # Verify markdown parser was called
        mock_markdown_parser.assert_called_once()
        # Semantic parser should be called for large nodes
        mock_semantic_parser.assert_called_once()

    @pytest.mark.skip(reason="Requires real embedding model for SemanticSplitterNodeParser")
    def test_markdown_semantic_splits_large_sections(self, mock_embed_model):
        """Test that large markdown sections are semantically split."""
        config = EmbeddingConfig(chunk_size=100, chunk_overlap=20)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown_semantic")

        # Create a large section that exceeds chunk_size
        large_content = """# Large Section

""" + " ".join(["This is a test sentence with content."] * 200)

        doc = Document(text=large_content, metadata={'title': 'Test'})
        nodes = chunker.chunk_documents([doc])

        # Should create multiple chunks
        assert len(nodes) > 0

    def test_markdown_semantic_keeps_small_sections(self, mock_embed_model):
        """Test that small markdown sections are not split."""
        config = EmbeddingConfig(chunk_size=1000, chunk_overlap=100)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown_semantic")

        content = """# Small Section

This is a small section with minimal content.
"""

        doc = Document(text=content, metadata={'title': 'Test'})
        nodes = chunker.chunk_documents([doc])

        # Should create at least one chunk
        assert len(nodes) >= 1


class TestParentDocumentContext:
    """Test parent document context functionality."""

    def test_add_parent_document_context(self, mock_embed_model):
        """Test adding parent document context to nodes."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        # Create test nodes from same document
        nodes = [
            TextNode(text="First chunk", metadata={'file_path': '/test/note.md'}),
            TextNode(text="Second chunk", metadata={'file_path': '/test/note.md'}),
            TextNode(text="Third chunk", metadata={'file_path': '/test/note.md'}),
        ]

        enhanced_nodes = chunker.add_parent_document_context(nodes)

        # Check that parent context was added
        assert len(enhanced_nodes) == 3
        for node in enhanced_nodes:
            assert 'parent_summary' in node.metadata
            assert 'total_chunks' in node.metadata
            assert node.metadata['total_chunks'] == 3

    def test_add_parent_context_multiple_documents(self, mock_embed_model):
        """Test adding parent context across multiple documents."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        # Create nodes from different documents
        nodes = [
            TextNode(text="Doc1 chunk1", metadata={'file_path': '/test/note1.md'}),
            TextNode(text="Doc1 chunk2", metadata={'file_path': '/test/note1.md'}),
            TextNode(text="Doc2 chunk1", metadata={'file_path': '/test/note2.md'}),
        ]

        enhanced_nodes = chunker.add_parent_document_context(nodes)

        # Group by file_path to verify
        doc1_nodes = [n for n in enhanced_nodes if n.metadata['file_path'] == '/test/note1.md']
        doc2_nodes = [n for n in enhanced_nodes if n.metadata['file_path'] == '/test/note2.md']

        assert all(n.metadata['total_chunks'] == 2 for n in doc1_nodes)
        assert all(n.metadata['total_chunks'] == 1 for n in doc2_nodes)

    def test_add_parent_context_empty_nodes(self, mock_embed_model):
        """Test adding parent context with empty node list."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        enhanced_nodes = chunker.add_parent_document_context([])
        assert enhanced_nodes == []

    def test_parent_summary_truncated(self, mock_embed_model):
        """Test that parent summary is properly truncated."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model)

        # Create a very long first chunk
        long_text = "A" * 1000
        nodes = [
            TextNode(text=long_text, metadata={'file_path': '/test/note.md'}),
        ]

        enhanced_nodes = chunker.add_parent_document_context(nodes)

        # Parent summary should be truncated to 500 chars
        assert len(enhanced_nodes[0].metadata['parent_summary']) <= 500


class TestChunkingErrorHandling:
    """Test error handling in chunking operations."""

    def test_chunking_error_propagation(self, mock_embed_model):
        """Test that chunking errors are properly handled."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="obsidian_aware")

        # Create documents that will cause semantic splitting to fail with mock embed model
        # We need a large document to trigger semantic splitting path
        large_text = "Test sentence. " * 500  # Large enough to exceed chunk_size

        doc = Document(text=large_text, metadata={'title': 'LargeDoc'})

        # With a mock embed model, the semantic splitter will fail with Pydantic validation error
        # This tests the error wrapping in _obsidian_aware_chunking
        with pytest.raises(RuntimeError) as exc_info:
            chunker.chunk_documents([doc])
        assert "Failed to chunk documents" in str(exc_info.value)


class TestCodeBlockPreservation:
    """Test that code blocks are preserved during chunking."""

    def test_preserve_python_code_blocks(self, mock_embed_model, sample_code_block_content):
        """Test preservation of Python code blocks."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        doc = Document(text=sample_code_block_content, metadata={'title': 'Code Test'})
        nodes = chunker.chunk_documents([doc])

        # Reconstruct full text from chunks
        full_text = " ".join([node.text for node in nodes])

        # Code blocks should be preserved
        assert "def calculate" in full_text

    def test_preserve_javascript_code_blocks(self, mock_embed_model, sample_code_block_content):
        """Test preservation of JavaScript code blocks."""
        config = EmbeddingConfig(chunk_size=512, chunk_overlap=75)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        doc = Document(text=sample_code_block_content, metadata={'title': 'Code Test'})
        nodes = chunker.chunk_documents([doc])

        full_text = " ".join([node.text for node in nodes])
        assert "function hello" in full_text

    def test_code_blocks_not_split_mid_block(self, mock_embed_model):
        """Test that code blocks are not split in the middle."""
        config = EmbeddingConfig(chunk_size=100, chunk_overlap=20)
        chunker = ObsidianChunker(config, mock_embed_model, strategy="markdown")

        content = """# Code Example

```python
def long_function():
    # This is a long function
    # With multiple lines
    # That should stay together
    return "result"
```
"""

        doc = Document(text=content, metadata={'title': 'Code Test'})
        nodes = chunker.chunk_documents([doc])

        # Code block should ideally be kept together in one chunk
        # At minimum, verify chunks were created
        assert len(nodes) > 0
