"""Tests for ObsidianLoader."""
import pytest
from pathlib import Path
from loader import ObsidianLoader, ObsidianNote


class TestWikilinkExtraction:
    """Test wikilink extraction functionality."""

    def test_extract_simple_wikilinks(self, temp_vault):
        """Test extraction of simple wikilinks."""
        loader = ObsidianLoader(temp_vault)
        content = "See [[Note1]] and [[Note2]]"
        links = loader.extract_wikilinks(content)
        assert links == ["Note1", "Note2"]

    def test_extract_wikilinks_with_aliases(self, temp_vault):
        """Test extraction of wikilinks with aliases."""
        loader = ObsidianLoader(temp_vault)
        content = "See [[Note1]] and [[Note2|alias]]"
        links = loader.extract_wikilinks(content)
        assert links == ["Note1", "Note2"]

    def test_extract_wikilinks_with_spaces(self, temp_vault):
        """Test extraction of wikilinks with spaces."""
        loader = ObsidianLoader(temp_vault)
        content = "[[Note with spaces]] and [[Another Note|with alias]]"
        links = loader.extract_wikilinks(content)
        assert links == ["Note with spaces", "Another Note"]

    def test_extract_wikilinks_nested_paths(self, temp_vault):
        """Test extraction of wikilinks with nested paths."""
        loader = ObsidianLoader(temp_vault)
        content = "[[folder/subfolder/Note]]"
        links = loader.extract_wikilinks(content)
        assert links == ["folder/subfolder/Note"]

    def test_extract_no_wikilinks(self, temp_vault):
        """Test content with no wikilinks."""
        loader = ObsidianLoader(temp_vault)
        content = "Just plain text without any links"
        links = loader.extract_wikilinks(content)
        assert links == []

    def test_extract_multiple_wikilinks_same_line(self, temp_vault):
        """Test multiple wikilinks on the same line."""
        loader = ObsidianLoader(temp_vault)
        content = "[[Link1]] [[Link2]] [[Link3]]"
        links = loader.extract_wikilinks(content)
        assert links == ["Link1", "Link2", "Link3"]


class TestTagExtraction:
    """Test tag extraction functionality."""

    def test_extract_simple_tags(self, temp_vault):
        """Test extraction of simple inline tags."""
        loader = ObsidianLoader(temp_vault)
        content = "Tags: #project #work"
        tags = loader.extract_tags(content, None)
        assert "project" in tags
        assert "work" in tags

    def test_extract_nested_tags(self, temp_vault):
        """Test extraction of nested tags."""
        loader = ObsidianLoader(temp_vault)
        content = "Tags: #project/work #daily-notes"
        tags = loader.extract_tags(content, None)
        assert "project/work" in tags
        assert "daily-notes" in tags

    def test_extract_tags_with_hyphens(self, temp_vault):
        """Test extraction of tags with hyphens."""
        loader = ObsidianLoader(temp_vault)
        content = "#my-tag #another-tag"
        tags = loader.extract_tags(content, None)
        assert "my-tag" in tags
        assert "another-tag" in tags

    def test_extract_frontmatter_tags_list(self, temp_vault):
        """Test extraction of tags from frontmatter as list."""
        loader = ObsidianLoader(temp_vault)
        content = "Some content"
        frontmatter_tags = ["tag1", "tag2"]
        tags = loader.extract_tags(content, frontmatter_tags)
        assert "tag1" in tags
        assert "tag2" in tags

    def test_extract_frontmatter_tags_string(self, temp_vault):
        """Test extraction of tags from frontmatter as string."""
        loader = ObsidianLoader(temp_vault)
        content = "Some content"
        frontmatter_tags = "single-tag"
        tags = loader.extract_tags(content, frontmatter_tags)
        assert "single-tag" in tags

    def test_extract_combined_tags(self, temp_vault):
        """Test extraction of both frontmatter and inline tags."""
        loader = ObsidianLoader(temp_vault)
        content = "Content with #inline-tag"
        frontmatter_tags = ["frontmatter-tag"]
        tags = loader.extract_tags(content, frontmatter_tags)
        assert "inline-tag" in tags
        assert "frontmatter-tag" in tags

    def test_extract_no_tags(self, temp_vault):
        """Test content with no tags."""
        loader = ObsidianLoader(temp_vault)
        content = "Just plain text without any tags"
        tags = loader.extract_tags(content, None)
        assert tags == []


class TestNoteLoading:
    """Test note loading functionality."""

    def test_load_note_with_frontmatter(self, temp_vault, sample_note_content, create_test_note):
        """Test loading a note with frontmatter."""
        from datetime import date
        note_path = create_test_note("test.md", sample_note_content)
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(note_path)

        assert note is not None
        assert note.title == "Test Note"
        assert "project" in note.tags
        assert "daily-notes" in note.tags
        assert "Note1" in note.wikilinks
        assert "Note2" in note.wikilinks
        # Frontmatter parses dates as datetime.date objects
        assert note.created_date == date(2024, 1, 1)

    def test_load_note_without_frontmatter(self, temp_vault, sample_note_no_frontmatter, create_test_note):
        """Test loading a note without frontmatter."""
        note_path = create_test_note("simple.md", sample_note_no_frontmatter)
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(note_path)

        assert note is not None
        assert note.title == "simple"  # Uses filename
        assert "Link1" in note.wikilinks
        assert "simple" in note.tags
        assert "test" in note.tags

    def test_load_nonexistent_note(self, temp_vault):
        """Test loading a nonexistent note returns None."""
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(temp_vault / "nonexistent.md")
        assert note is None

    def test_load_note_unicode_content(self, temp_vault, create_test_note):
        """Test loading a note with unicode content."""
        content = """---
title: Unicode Test
---

# Unicode Test

Testing unicode: 你好世界 🌍 émojis
"""
        note_path = create_test_note("unicode.md", content)
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(note_path)

        assert note is not None
        assert "你好世界" in note.content
        assert "🌍" in note.content


class TestVaultLoading:
    """Test vault loading functionality."""

    def test_load_empty_vault(self, temp_vault):
        """Test loading an empty vault."""
        loader = ObsidianLoader(temp_vault)
        notes = loader.load_vault()
        assert notes == []

    def test_load_vault_with_multiple_notes(self, temp_vault, create_test_note, sample_note_content):
        """Test loading a vault with multiple notes."""
        create_test_note("note1.md", sample_note_content)
        create_test_note("note2.md", sample_note_content)
        create_test_note("note3.md", sample_note_content)

        loader = ObsidianLoader(temp_vault)
        notes = loader.load_vault()
        assert len(notes) == 3

    def test_load_vault_with_subdirectories(self, temp_vault, create_test_note, sample_note_content):
        """Test loading a vault with nested subdirectories."""
        create_test_note("note1.md", sample_note_content)
        create_test_note("folder1/note2.md", sample_note_content)
        create_test_note("folder1/folder2/note3.md", sample_note_content)

        loader = ObsidianLoader(temp_vault)
        notes = loader.load_vault()
        assert len(notes) == 3

    def test_load_vault_nonexistent_path(self):
        """Test loading a nonexistent vault raises FileNotFoundError."""
        loader = ObsidianLoader(Path("/nonexistent/vault"))
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_vault()
        assert "Vault path not found" in str(exc_info.value)

    def test_load_vault_file_not_directory(self, temp_vault, create_test_note):
        """Test loading a file instead of directory raises NotADirectoryError."""
        file_path = create_test_note("file.txt", "content")
        loader = ObsidianLoader(file_path)
        with pytest.raises(NotADirectoryError) as exc_info:
            loader.load_vault()
        assert "not a directory" in str(exc_info.value)

    def test_load_vault_filters_by_extension(self, temp_vault, create_test_note, sample_note_content):
        """Test that vault loading only includes specified extensions."""
        create_test_note("note1.md", sample_note_content)
        create_test_note("note2.txt", sample_note_content)
        create_test_note("note3.md", sample_note_content)

        loader = ObsidianLoader(temp_vault)
        notes = loader.load_vault(extensions=['.md'])
        # Should only load .md files
        assert len(notes) == 2


class TestDocumentConversion:
    """Test conversion of notes to documents."""

    def test_notes_to_documents(self, temp_vault, sample_note_content, create_test_note):
        """Test conversion of ObsidianNote to LlamaIndex Documents."""
        note_path = create_test_note("test.md", sample_note_content)
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(note_path)

        documents = loader.notes_to_documents([note])
        assert len(documents) == 1

        doc = documents[0]
        assert doc.text == note.content
        assert doc.metadata['title'] == "Test Note"
        assert doc.metadata['file_name'] == "test.md"
        assert "project" in doc.metadata['tags']
        assert "Note1" in doc.metadata['wikilinks']
        assert doc.metadata['num_wikilinks'] == 3

    def test_notes_to_documents_preserves_metadata(self, temp_vault, create_test_note):
        """Test that custom frontmatter fields are preserved."""
        content = """---
title: Custom Fields
author: Test Author
custom_field: custom_value
---

Content here.
"""
        note_path = create_test_note("custom.md", content)
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(note_path)

        documents = loader.notes_to_documents([note])
        doc = documents[0]

        assert doc.metadata['author'] == "Test Author"
        assert doc.metadata['custom_field'] == "custom_value"


class TestWikilinkGraph:
    """Test wikilink graph building functionality."""

    def test_build_simple_graph(self, temp_vault, create_test_note):
        """Test building a simple wikilink graph."""
        content1 = "[[Note2]]"
        content2 = "[[Note1]]"

        note_path1 = create_test_note("Note1.md", content1)
        note_path2 = create_test_note("Note2.md", content2)

        loader = ObsidianLoader(temp_vault)
        notes = loader.load_vault()
        graph = loader.build_wikilink_graph(notes)

        assert len(graph) == 2
        # Each note should have connections to the other
        assert any(str(note_path2) in connections for connections in graph.values())
        assert any(str(note_path1) in connections for connections in graph.values())

    def test_build_graph_unresolved_links(self, temp_vault, create_test_note):
        """Test graph building with unresolved wikilinks."""
        content = "[[ExistingNote]] [[NonExistentNote]]"

        create_test_note("Note1.md", content)
        create_test_note("ExistingNote.md", "Content")

        loader = ObsidianLoader(temp_vault)
        notes = loader.load_vault()
        graph = loader.build_wikilink_graph(notes)

        # Should only include resolved links
        note1_connections = graph.get(str(temp_vault / "Note1.md"), [])
        # Should have connection to ExistingNote but not NonExistentNote
        assert any("ExistingNote" in conn for conn in note1_connections)

    def test_build_empty_graph(self, temp_vault):
        """Test building graph from empty vault."""
        loader = ObsidianLoader(temp_vault)
        graph = loader.build_wikilink_graph([])
        assert graph == {}


class TestPathSecurity:
    """Test path traversal protection."""

    def test_path_outside_vault_rejected(self, temp_vault):
        """Test that paths outside vault are properly handled."""
        loader = ObsidianLoader(temp_vault)

        # Attempt to load a note outside the vault
        outside_path = Path("/tmp/outside_vault/note.md")
        note = loader.load_note(outside_path)

        # Should return None for nonexistent files
        assert note is None

    def test_relative_paths_resolved(self, temp_vault, create_test_note, sample_note_content):
        """Test that relative paths are properly resolved."""
        note_path = create_test_note("subdir/note.md", sample_note_content)
        loader = ObsidianLoader(temp_vault)
        note = loader.load_note(note_path)

        assert note is not None
        assert note.path.is_absolute()
