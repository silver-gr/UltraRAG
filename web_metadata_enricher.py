"""Web-based metadata enrichment for books missing Calibre data.

Uses Tavily web search to find genre/category/description for books
that couldn't be matched in Calibre or lack categories.

Optional dependency: tavily-python (import guarded).
"""

import json
import logging
import time
from pathlib import Path

from models import BookMetadata, _normalize_category

logger = logging.getLogger(__name__)

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.debug("tavily-python not installed — web enrichment disabled")


class WebMetadataEnricher:
    """Enrich book metadata using web search (Tavily)."""

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float = 1.0,
        max_per_run: int = 50,
    ):
        self.rate_limit_delay = rate_limit_delay
        self.max_per_run = max_per_run
        self._client = None

        if not TAVILY_AVAILABLE:
            logger.warning("tavily-python not installed — web enrichment unavailable")
            return

        if not api_key:
            import os
            api_key = os.getenv("TAVILY_API_KEY", "")

        if api_key:
            self._client = TavilyClient(api_key=api_key)
        else:
            logger.warning("No TAVILY_API_KEY — web enrichment disabled")

    @property
    def available(self) -> bool:
        return self._client is not None

    def enrich(self, meta: BookMetadata) -> BookMetadata:
        """Enrich a single BookMetadata entry with web search results.

        Searches for book genre/category/description and updates the entry.
        Returns the updated BookMetadata (same object mutated).
        """
        if not self.available:
            return meta

        query = f'book "{meta.title}"'
        if meta.author:
            query += f" by {meta.author}"
        query += " genre category description"

        try:
            results = self._client.search(query, max_results=3)
            content_parts = []
            for result in results.get("results", []):
                content_parts.append(result.get("content", ""))

            if content_parts:
                combined = "\n".join(content_parts)
                extracted = self._extract_metadata(combined, meta.title, meta.author)

                if extracted.get("categories") and not meta.categories:
                    meta.categories = [_normalize_category(c) for c in extracted["categories"]]
                if extracted.get("description") and not meta.description:
                    meta.description = extracted["description"][:500]

                # Update source
                if meta.metadata_source == "calibre":
                    meta.metadata_source = "calibre+web"
                elif meta.metadata_source == "filename":
                    meta.metadata_source = "web"

        except Exception as e:
            logger.warning("Web enrichment failed for '%s': %s", meta.title, e)

        return meta

    def enrich_batch(
        self,
        entries: list[BookMetadata],
    ) -> tuple[list[BookMetadata], int]:
        """Enrich multiple entries, respecting rate limits.

        Returns (enriched_entries, count_enriched).
        """
        if not self.available:
            logger.warning("Web enrichment not available — skipping batch")
            return entries, 0

        count = 0
        for meta in entries:
            if count >= self.max_per_run:
                logger.info("Hit max web enrichments per run (%d)", self.max_per_run)
                break

            # Only enrich if missing categories or description
            if meta.categories and meta.description:
                continue

            self.enrich(meta)
            count += 1

            if count < self.max_per_run:
                time.sleep(self.rate_limit_delay)

        logger.info("Web-enriched %d books", count)
        return entries, count

    @staticmethod
    def _extract_metadata(text: str, title: str, author: str) -> dict:
        """Extract categories and description from web search text.

        Simple heuristic extraction — looks for genre/category keywords.
        """
        result: dict = {"categories": [], "description": ""}

        # Common book categories to look for in text
        known_categories = [
            "self-help", "self-improvement", "productivity", "psychology",
            "business", "finance", "science", "technology", "philosophy",
            "history", "biography", "memoir", "fiction", "non-fiction",
            "health", "wellness", "mindfulness", "meditation", "leadership",
            "management", "economics", "neuroscience", "cognitive science",
            "personal development", "motivation", "habits", "creativity",
            "communication", "relationships", "parenting", "education",
            "spirituality", "religion", "politics", "sociology",
            "computer science", "programming", "data science", "ai",
            "machine learning", "design", "marketing", "entrepreneurship",
        ]

        text_lower = text.lower()
        found_categories = []
        for cat in known_categories:
            if cat in text_lower:
                found_categories.append(_normalize_category(cat))

        # Deduplicate
        seen = set()
        for cat in found_categories:
            if cat not in seen:
                seen.add(cat)
                result["categories"].append(cat)

        # Extract first meaningful sentence as description
        sentences = text.split(".")
        for sentence in sentences[:10]:
            cleaned = sentence.strip()
            if len(cleaned) > 50 and title.lower().split()[0] in cleaned.lower():
                result["description"] = cleaned[:500] + "."
                break

        return result
