"""Tests for Calibre metadata extraction and book metadata caching."""

import json
import tempfile
from pathlib import Path

import pytest

from models import BookMetadata, BookFilter, _normalize_category
from calibre_metadata import (
    CalibreMetadataExtractor,
    BookMetadataCache,
    DEFAULT_EXCLUDE_TAGS,
)


# === BookMetadata tests ===

class TestBookMetadata:
    def test_book_uid_with_calibre_id(self):
        meta = BookMetadata(
            title="Deep Work", file_path="/books/deep.epub",
            file_type="epub", file_size=1000, calibre_id=42,
        )
        assert meta.book_uid == "calibre:42"

    def test_book_uid_without_calibre_id(self):
        meta = BookMetadata(
            title="Deep Work", file_path="/books/deep.epub",
            file_type="epub", file_size=1000,
        )
        uid = meta.book_uid
        assert uid.startswith("hash:")
        assert len(uid) > 5

    def test_book_uid_stability(self):
        """Same path + size should produce same uid."""
        meta1 = BookMetadata(
            title="A", file_path="/books/a.epub", file_type="epub", file_size=100,
        )
        meta2 = BookMetadata(
            title="B", file_path="/books/a.epub", file_type="epub", file_size=100,
        )
        assert meta1.book_uid == meta2.book_uid

    def test_book_uid_changes_with_size(self):
        meta1 = BookMetadata(
            title="A", file_path="/books/a.epub", file_type="epub", file_size=100,
        )
        meta2 = BookMetadata(
            title="A", file_path="/books/a.epub", file_type="epub", file_size=200,
        )
        assert meta1.book_uid != meta2.book_uid


# === BookFilter tests ===

class TestBookFilter:
    def test_empty_filter(self):
        f = BookFilter()
        assert f.to_lance_where() is None

    def test_category_filter(self):
        f = BookFilter(categories=["Productivity"])
        where = f.to_lance_where()
        assert "array_has_any" in where
        assert "make_array" in where
        assert "'productivity'" in where  # normalized to lowercase

    def test_author_filter(self):
        f = BookFilter(authors=["Cal Newport"])
        where = f.to_lance_where()
        assert "metadata.book_author IN" in where
        assert "'Cal Newport'" in where

    def test_uid_filter(self):
        f = BookFilter(book_uids=["calibre:42", "hash:abc123"])
        where = f.to_lance_where()
        assert "metadata.book_uid IN" in where
        assert "'calibre:42'" in where
        assert "'hash:abc123'" in where

    def test_language_filter(self):
        f = BookFilter(language="eng")
        where = f.to_lance_where()
        assert "metadata.book_language = 'eng'" in where

    def test_combined_filter_and_semantics(self):
        f = BookFilter(categories=["Productivity"], authors=["Cal Newport"], language="eng")
        where = f.to_lance_where()
        assert " AND " in where
        parts = where.split(" AND ")
        assert len(parts) == 3

    def test_sql_escape_single_quote(self):
        f = BookFilter(authors=["Tim O'Reilly"])
        where = f.to_lance_where()
        assert "O''Reilly" in where
        assert "O'Reilly" not in where.replace("O''Reilly", "")

    def test_multiple_categories_or_semantics(self):
        f = BookFilter(categories=["Productivity", "Habits"])
        where = f.to_lance_where()
        assert "'productivity'" in where
        assert "'habits'" in where
        # Both in same array_has_any = OR semantics
        assert where.count("array_has_any") == 1

    def test_multiple_authors_or_semantics(self):
        f = BookFilter(authors=["Cal Newport", "James Clear"])
        where = f.to_lance_where()
        assert "IN" in where
        assert "'Cal Newport'" in where
        assert "'James Clear'" in where


# === _normalize_category tests ===

class TestNormalizeCategory:
    def test_lowercase(self):
        assert _normalize_category("Self-Help") == "self-help"

    def test_strip_whitespace(self):
        assert _normalize_category("  productivity  ") == "productivity"

    def test_collapse_whitespace(self):
        assert _normalize_category("self  improvement") == "self improvement"

    def test_consistency(self):
        assert _normalize_category("Self-Improvement") == _normalize_category("self-improvement")
        assert _normalize_category("  HABITS ") == _normalize_category("habits")


# === CalibreMetadataExtractor tests ===

class TestCalibreExtractor:
    def test_missing_db_returns_none(self, tmp_path):
        extractor = CalibreMetadataExtractor(tmp_path / "nonexistent.db")
        result = extractor.match_book(Path("/fake/book.epub"))
        assert result is None

    def test_parse_filename_author_title(self):
        author, title = CalibreMetadataExtractor._parse_filename("Cal Newport - Deep Work.epub")
        assert author == "Cal Newport"
        assert title == "Deep Work"

    def test_parse_filename_no_separator(self):
        author, title = CalibreMetadataExtractor._parse_filename("DeepWork.epub")
        assert author == ""
        assert title == "DeepWork"

    def test_parse_filename_multiple_separators(self):
        author, title = CalibreMetadataExtractor._parse_filename("Author - Title - Subtitle.pdf")
        assert author == "Author"
        assert title == "Title - Subtitle"

    def test_similarity_exact(self):
        assert CalibreMetadataExtractor._similarity("hello", "hello") == 1.0

    def test_similarity_case_insensitive(self):
        assert CalibreMetadataExtractor._similarity("Hello", "hello") == 1.0

    def test_similarity_different(self):
        score = CalibreMetadataExtractor._similarity("hello", "world")
        assert score < 0.5

    def test_excluded_tags_filtered(self):
        """Verify default exclude tags are applied."""
        assert "audio" in DEFAULT_EXCLUDE_TAGS
        assert "notbooks" in DEFAULT_EXCLUDE_TAGS


# === BookMetadataCache tests ===

class TestBookMetadataCache:
    def test_put_and_lookup(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        meta = BookMetadata(
            title="Deep Work", file_path="/books/deep.epub",
            file_type="epub", file_size=1000, calibre_id=42,
            author="Cal Newport", categories=["productivity"],
        )
        cache.put(meta)
        result = cache.lookup(Path("/books/deep.epub"))
        assert result is not None
        assert result.title == "Deep Work"
        assert result.calibre_id == 42
        assert result.book_uid == "calibre:42"

    def test_lookup_missing(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        result = cache.lookup(Path("/nonexistent/book.epub"))
        assert result is None

    def test_save_and_reload(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache1 = BookMetadataCache(cache_path)
        meta = BookMetadata(
            title="Atomic Habits", file_path="/books/habits.epub",
            file_type="epub", file_size=2000, calibre_id=99,
        )
        cache1.put(meta)
        cache1.save()

        # Reload
        cache2 = BookMetadataCache(cache_path)
        result = cache2.lookup(Path("/books/habits.epub"))
        assert result is not None
        assert result.title == "Atomic Habits"

    def test_cache_key_calibre(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        meta = BookMetadata(
            title="Test", file_path="/books/test.epub",
            file_type="epub", file_size=500, calibre_id=10,
        )
        key = cache._cache_key(meta)
        assert key == "calibre:10"

    def test_cache_key_hash(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        meta = BookMetadata(
            title="Test", file_path="/books/test.epub",
            file_type="epub", file_size=500,
        )
        key = cache._cache_key(meta)
        assert key.startswith("hash:")

    def test_atomic_write(self, tmp_path):
        """Cache file should exist after save, even if path didn't exist."""
        cache_path = tmp_path / "subdir" / "cache.json"
        cache = BookMetadataCache(cache_path)
        meta = BookMetadata(
            title="Test", file_path="/books/test.epub",
            file_type="epub", file_size=100,
        )
        cache.put(meta)
        cache.save()
        assert cache_path.exists()

        # Verify JSON is valid
        data = json.loads(cache_path.read_text())
        assert data["_version"] == 1
        assert len(data["entries"]) == 1

    def test_clear(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        meta = BookMetadata(
            title="Test", file_path="/books/test.epub",
            file_type="epub", file_size=100, calibre_id=1,
        )
        cache.put(meta)
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_stats(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        meta1 = BookMetadata(
            title="A", file_path="/a.epub", file_type="epub",
            file_size=100, calibre_id=1, match_confidence=0.85,
            metadata_source="calibre",
        )
        meta2 = BookMetadata(
            title="B", file_path="/b.epub", file_type="epub",
            file_size=200, metadata_source="filename",
        )
        cache.put(meta1)
        cache.put(meta2)

        stats = cache.stats()
        assert stats["total"] == 2
        assert "calibre" in stats["by_source"]

    def test_contains(self, tmp_path):
        cache = BookMetadataCache(tmp_path / "cache.json")
        meta = BookMetadata(
            title="Test", file_path="/books/test.epub",
            file_type="epub", file_size=100,
        )
        assert Path("/books/test.epub") not in cache
        cache.put(meta)
        assert Path("/books/test.epub") in cache

    def test_version_mismatch_rebuilds(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        # Write old version
        cache_path.write_text(json.dumps({
            "_version": 999,
            "entries": {"fake": {}},
            "_path_index": {},
        }))
        cache = BookMetadataCache(cache_path)
        assert len(cache) == 0  # Should have been cleared
