"""Calibre metadata extraction and book metadata caching.

Extracts rich metadata (author, categories, description, language, publisher)
from a Calibre metadata.db SQLite database and caches results for fast lookup
during indexing.
"""

import json
import logging
import os
import re
import sqlite3
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

from models import BookMetadata, _normalize_category

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_TAGS = frozenset({
    "audio", "notbooks", "have read", "oldstuff", "reading", "quick", "su",
})


class CalibreMetadataExtractor:
    """Extract book metadata from a Calibre metadata.db SQLite database."""

    def __init__(
        self,
        db_path: Path,
        match_threshold: float = 0.75,
        exclude_tags: frozenset[str] | None = None,
    ):
        self.db_path = db_path
        self.match_threshold = match_threshold
        self.exclude_tags = exclude_tags if exclude_tags is not None else DEFAULT_EXCLUDE_TAGS
        self._entries: list[dict[str, Any]] | None = None

    def _load_entries(self) -> list[dict[str, Any]]:
        """Load all Calibre entries with a single JOIN query."""
        if self._entries is not None:
            return self._entries

        if not self.db_path.exists():
            logger.warning("Calibre DB not found: %s", self.db_path)
            self._entries = []
            return self._entries

        uri = f"file:{self.db_path}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError as e:
            logger.error("Failed to open Calibre DB: %s", e)
            self._entries = []
            return self._entries

        try:
            query = """
                SELECT
                    b.id,
                    b.title,
                    b.path AS calibre_path,
                    b.isbn,
                    GROUP_CONCAT(DISTINCT a.name) AS authors,
                    GROUP_CONCAT(DISTINCT t.name) AS tags,
                    c.text AS description,
                    GROUP_CONCAT(DISTINCT p.name) AS publishers,
                    GROUP_CONCAT(DISTINCT l.lang_code) AS languages
                FROM books b
                LEFT JOIN books_authors_link bal ON b.id = bal.book
                LEFT JOIN authors a ON bal.author = a.id
                LEFT JOIN books_tags_link btl ON b.id = btl.book
                LEFT JOIN tags t ON btl.tag = t.id
                LEFT JOIN comments c ON b.id = c.book
                LEFT JOIN books_publishers_link bpl ON b.id = bpl.book
                LEFT JOIN publishers p ON bpl.publisher = p.id
                LEFT JOIN books_languages_link bll ON b.id = bll.book
                LEFT JOIN languages l ON bll.lang_code = l.id
                GROUP BY b.id
            """
            rows = conn.execute(query).fetchall()

            entries = []
            for row in rows:
                tags_raw = (row["tags"] or "").split(",")
                tags = [
                    _normalize_category(t)
                    for t in tags_raw
                    if t.strip() and _normalize_category(t) not in self.exclude_tags
                ]
                # Deduplicate while preserving order
                seen = set()
                unique_tags = []
                for tag in tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)

                entries.append({
                    "id": row["id"],
                    "title": row["title"] or "",
                    "authors": row["authors"] or "",
                    "tags": unique_tags,
                    "description": row["description"] or "",
                    "publishers": row["publishers"] or "",
                    "languages": row["languages"] or "",
                    "isbn": row["isbn"] or "",
                    "calibre_path": row["calibre_path"] or "",
                })

            self._entries = entries
            logger.info("Loaded %d entries from Calibre DB", len(entries))
            return self._entries

        except sqlite3.Error as e:
            logger.error("Calibre DB query failed: %s", e)
            self._entries = []
            return self._entries
        finally:
            conn.close()

    @staticmethod
    def _parse_filename(filename: str) -> tuple[str, str]:
        """Extract author and title from filename pattern 'Author - Title.ext'.

        Returns (author, title). If no separator found, returns ("", filename).
        """
        stem = Path(filename).stem
        # Try "Author - Title" pattern
        if " - " in stem:
            parts = stem.split(" - ", 1)
            return parts[0].strip(), parts[1].strip()
        return "", stem.strip()

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute similarity ratio between two strings (case-insensitive)."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def match_book(self, book_path: Path) -> BookMetadata | None:
        """Fuzzy-match a book file to a Calibre entry.

        Uses dual-gate matching: both author AND title must individually
        pass the threshold.

        Returns BookMetadata with match_confidence, or None if no match.
        """
        entries = self._load_entries()
        if not entries:
            return None

        file_author, file_title = self._parse_filename(book_path.name)
        file_size = book_path.stat().st_size if book_path.exists() else 0
        file_type = book_path.suffix.lower().lstrip(".")

        best_match = None
        best_score = 0.0

        for entry in entries:
            title_sim = self._similarity(file_title, entry["title"])

            if file_author:
                # First author from comma-separated list
                calibre_author = entry["authors"].split(",")[0].strip() if entry["authors"] else ""
                author_sim = self._similarity(file_author, calibre_author)

                # Dual-gate: both must pass threshold
                if title_sim < self.match_threshold or author_sim < self.match_threshold:
                    continue

                combined_score = (title_sim + author_sim) / 2
            else:
                # No author in filename — title-only matching with higher bar
                if title_sim < self.match_threshold:
                    continue
                combined_score = title_sim * 0.9  # Slight penalty for no author confirmation

            if combined_score > best_score:
                best_score = combined_score
                best_match = entry

        if best_match is None:
            return None

        return BookMetadata(
            title=best_match["title"],
            file_path=str(book_path),
            file_type=file_type,
            file_size=file_size,
            author=best_match["authors"].split(",")[0].strip() if best_match["authors"] else "",
            categories=best_match["tags"],
            description=best_match["description"],
            language=best_match["languages"].split(",")[0].strip() if best_match["languages"] else "",
            publisher=best_match["publishers"].split(",")[0].strip() if best_match["publishers"] else "",
            isbn=best_match["isbn"],
            calibre_id=best_match["id"],
            match_confidence=round(best_score, 4),
            metadata_source="calibre",
        )


class BookMetadataCache:
    """Persistent JSON cache for book metadata with atomic writes.

    Cache key: calibre_id when matched, otherwise sha256(path+size).
    Lookup API: cache.lookup(book_path) -> BookMetadata | None
    """

    _VERSION = 1

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._entries: dict[str, dict[str, Any]] = {}
        self._by_file_path: dict[str, str] = {}  # file_path -> cache_key
        self._load()

    def _load(self):
        """Load cache from disk."""
        if not self.cache_path.exists():
            return

        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            version = data.get("_version", 0)
            if version != self._VERSION:
                logger.warning(
                    "Cache version mismatch (got %d, expected %d), rebuilding",
                    version, self._VERSION,
                )
                return

            self._entries = data.get("entries", {})
            self._by_file_path = data.get("_path_index", {})
            logger.info("Loaded metadata cache: %d entries", len(self._entries))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load metadata cache: %s", e)
            self._entries = {}
            self._by_file_path = {}

    def _save(self):
        """Atomic write: temp file + os.replace()."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "_version": self._VERSION,
            "entries": self._entries,
            "_path_index": self._by_file_path,
        }
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.cache_path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(self.cache_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _make_hash_key(file_path: str, file_size: int) -> str:
        """Generate hash-based cache key from path + size."""
        import hashlib
        h = hashlib.sha256(f"{file_path}:{file_size}".encode()).hexdigest()[:16]
        return f"hash:{h}"

    def _cache_key(self, meta: BookMetadata) -> str:
        """Determine the cache key for a BookMetadata entry."""
        if meta.calibre_id is not None:
            return f"calibre:{meta.calibre_id}"
        return self._make_hash_key(meta.file_path, meta.file_size)

    def put(self, meta: BookMetadata):
        """Store a BookMetadata entry in the cache."""
        key = self._cache_key(meta)
        self._entries[key] = {
            "title": meta.title,
            "file_path": meta.file_path,
            "file_type": meta.file_type,
            "file_size": meta.file_size,
            "author": meta.author,
            "categories": meta.categories,
            "description": meta.description,
            "language": meta.language,
            "publisher": meta.publisher,
            "isbn": meta.isbn,
            "calibre_id": meta.calibre_id,
            "match_confidence": meta.match_confidence,
            "metadata_source": meta.metadata_source,
        }
        self._by_file_path[meta.file_path] = key

    def lookup(self, book_path: Path) -> BookMetadata | None:
        """Look up metadata for a book by its file path.

        Returns BookMetadata if found in cache, None otherwise.
        """
        path_str = str(book_path)
        key = self._by_file_path.get(path_str)
        if key is None:
            return None

        entry = self._entries.get(key)
        if entry is None:
            return None

        return BookMetadata(**entry)

    def save(self):
        """Persist cache to disk (atomic)."""
        self._save()

    def clear(self):
        """Clear all cache entries."""
        self._entries.clear()
        self._by_file_path.clear()

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about cached entries."""
        if not self._entries:
            return {"total": 0}

        sources = {}
        confidences = []
        for entry in self._entries.values():
            src = entry.get("metadata_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            conf = entry.get("match_confidence", 0)
            if conf > 0:
                confidences.append(conf)

        return {
            "total": len(self._entries),
            "by_source": sources,
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0,
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, book_path: Path) -> bool:
        return str(book_path) in self._by_file_path

    def items(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Iterate over (cache_key, entry_dict) pairs."""
        return iter(self._entries.items())
