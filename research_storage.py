"""Research storage and export module for UltraRAG Content Research.

Handles saving, loading, and exporting federated search results.
Results are stored internally for query history and re-loading,
and can be exported as Markdown or JSON for external consumption.

Directory structure:
    ~/Projects/~HQ/research-exports/
    ├── .internal/                  # Full results (auto-saved, not user-facing)
    │   └── 2026-01-23_adhd-tips-and-tricks_a1b2c3d4.json
    ├── .query_history.json         # Query index (past queries with metadata)
    ├── 2026-01-23_adhd-tips-and-tricks.md   # User-facing markdown export
    └── 2026-01-23_adhd-tips-and-tricks.json # User-facing JSON export
"""

import json
import logging
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXPORTS_DIR = Path.home() / "Projects" / "~HQ" / "research-exports"
INTERNAL_DIR = EXPORTS_DIR / ".internal"
HISTORY_FILE = EXPORTS_DIR / ".query_history.json"


def _slugify(text: str, max_length: int = 30) -> str:
    """Convert query text to a filesystem-safe slug.

    Lowercases, replaces spaces with hyphens, strips non-alphanumeric/hyphen
    characters (including Unicode), and truncates to max_length.

    Args:
        text: The query text to slugify.
        max_length: Maximum slug length (default 30).

    Returns:
        A filesystem-safe slug string.
    """
    slug = text.lower().replace(" ", "-")
    # Strip anything that is not ASCII alphanumeric or hyphen
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug[:max_length]


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a ResearchResult object to a dict, excluding the embedding field.

    Handles both dataclass instances (with __dict__) and objects with explicit
    attributes (rank, source, title, chunk, raw_score, weighted_score, metadata).

    Args:
        result: A ResearchResult object with .source, .title, .chunk,
                .raw_score, .weighted_score, .metadata, .rank attributes.

    Returns:
        Dictionary representation without the 'embedding' field.
    """
    # Try dataclasses.asdict first for proper dataclass handling
    try:
        from dataclasses import asdict
        d = asdict(result)
    except (TypeError, ImportError):
        # Fallback: extract known attributes manually
        d = {
            "rank": getattr(result, "rank", 0),
            "source": getattr(result, "source", "unknown"),
            "title": getattr(result, "title", ""),
            "chunk": getattr(result, "chunk", ""),
            "raw_score": getattr(result, "raw_score", 0.0),
            "weighted_score": getattr(result, "weighted_score", 0.0),
            "metadata": getattr(result, "metadata", {}),
        }

    # Remove the embedding field if present (large, not needed for storage)
    d.pop("embedding", None)
    return d


def _get_internal_filepath(query: str, result_id: str) -> Path:
    """Build the internal storage filepath for a result.

    Args:
        query: The search query text.
        result_id: The unique result identifier.

    Returns:
        Path to the internal JSON file.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(query)
    filename = f"{date_str}_{slug}_{result_id}.json"
    return INTERNAL_DIR / filename


def _update_internal_storage(result_id: str, data: Dict[str, Any]) -> None:
    """Write updated data back to internal storage for a given result_id.

    Searches for the file matching the result_id and overwrites it.

    Args:
        result_id: The unique result identifier.
        data: The full result data dict to write.
    """
    for filepath in INTERNAL_DIR.glob(f"*_{result_id}.json"):
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.debug(f"Updated internal storage: {filepath.name}")
        return

    logger.warning(f"Could not find internal file for result_id={result_id} to update")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_results(
    query_settings: Dict[str, Any],
    results: List[Any],
    dedup_log: List[Dict[str, Any]]
) -> str:
    """Save results internally after a federated search.

    Creates the internal directory if needed, serializes results (excluding
    embeddings), computes per-source stats, and updates the query history index.

    Args:
        query_settings: Dict containing query (str), weights (dict),
                        top_k (int), threshold (float), dedup_similarity (float).
        results: List of ResearchResult objects with .source, .title, .chunk,
                 .raw_score, .weighted_score, .metadata, .rank attributes.
        dedup_log: List of dedup decision dicts.

    Returns:
        The generated result_id (uuid hex[:8]).
    """
    INTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    result_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().isoformat(timespec="seconds")
    query = query_settings.get("query", "")

    # Serialize results, excluding embedding field
    serialized_results = []
    for idx, result in enumerate(results, start=1):
        result_dict = _result_to_dict(result)
        # Ensure rank reflects final ordering
        result_dict["rank"] = idx
        serialized_results.append(result_dict)

    # Compute per-source statistics
    source_counts = Counter(r.get("source", "unknown") if isinstance(r, dict) else getattr(r, "source", "unknown") for r in results)
    per_source = dict(source_counts)

    data = {
        "id": result_id,
        "query": query,
        "timestamp": timestamp,
        "settings": query_settings,
        "stats": {
            "total_results": len(results),
            "per_source": per_source,
        },
        "results": serialized_results,
        "dedup_log": dedup_log,
        "exported_as": [],
    }

    # Write internal file
    filepath = _get_internal_filepath(query, result_id)
    filepath.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"Saved results internally: {filepath.name} ({len(results)} results)")

    # Update query history index
    update_history(result_id, data)

    return result_id


def load_results(result_id: str) -> Optional[Dict[str, Any]]:
    """Load full results from internal storage by ID.

    Searches for files matching *_{result_id}.json in the .internal directory.

    Args:
        result_id: The unique result identifier (uuid hex[:8]).

    Returns:
        The full result data dict, or None if not found.
    """
    if not INTERNAL_DIR.exists():
        logger.warning(f"Internal directory does not exist: {INTERNAL_DIR}")
        return None

    matches = list(INTERNAL_DIR.glob(f"*_{result_id}.json"))
    if not matches:
        logger.warning(f"No internal file found for result_id={result_id}")
        return None

    filepath = matches[0]
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        logger.debug(f"Loaded results from: {filepath.name}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading results from {filepath}: {e}")
        return None


def get_history() -> List[Dict[str, Any]]:
    """Get the query history (metadata only, sorted by timestamp desc).

    Reads from .query_history.json. Returns an empty list if the file
    does not exist or cannot be parsed.

    Returns:
        List of history entry dicts, most recent first.
    """
    if not HISTORY_FILE.exists():
        return []

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logger.warning("History file is not a list, returning empty")
        return []
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error reading history file: {e}")
        return []


def update_history(result_id: str, data: Dict[str, Any]) -> None:
    """Add or update an entry in the query history index.

    New entries are inserted at position 0 (most recent first).
    If an entry with the same result_id already exists, it is replaced.

    Args:
        result_id: The unique result identifier.
        data: The full result data dict (id, query, timestamp, stats, settings, exported_as).
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    history = get_history()

    entry = {
        "id": result_id,
        "query": data.get("query", ""),
        "timestamp": data.get("timestamp", ""),
        "stats": data.get("stats", {}),
        "settings": data.get("settings", {}),
        "exported_as": data.get("exported_as", []),
    }

    # Remove existing entry with same ID if present (for updates)
    history = [h for h in history if h.get("id") != result_id]

    # Insert at position 0 (most recent first)
    history.insert(0, entry)

    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.debug(f"Updated history: {result_id} ({entry['query'][:40]})")


def clear_history() -> None:
    """Delete the query history file and all files in .internal/.

    Removes .query_history.json and all JSON files in the .internal directory.
    Silently succeeds if files/directories do not exist.
    """
    # Remove history file
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
        logger.info("Deleted query history file")

    # Remove all internal files
    if INTERNAL_DIR.exists():
        removed = 0
        for filepath in INTERNAL_DIR.glob("*.json"):
            filepath.unlink()
            removed += 1
        logger.info(f"Deleted {removed} internal result files")


def export_markdown(result_id: str) -> Optional[Path]:
    """Export results as formatted Markdown to the exports directory.

    Groups results by source, includes scores, chunk previews (first 500 chars),
    and source-specific metadata. Updates the internal storage's exported_as field.

    Args:
        result_id: The unique result identifier.

    Returns:
        Path to the exported markdown file, or None if result_id not found.
    """
    data = load_results(result_id)
    if not data:
        logger.error(f"Cannot export markdown: result_id={result_id} not found")
        return None

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    slug = _slugify(data["query"])
    date_str = data["timestamp"][:10]
    filename = f"{date_str}_{slug}.md"
    filepath = EXPORTS_DIR / filename

    # Build weights display string
    weights = data.get("settings", {}).get("weights", {})
    weights_str = ", ".join(f"{k} ({v})" for k, v in weights.items())

    total_results = data.get("stats", {}).get("total_results", len(data.get("results", [])))

    lines = [
        f"# Research: {data['query']}",
        "",
        f"Generated: {data['timestamp']}",
        f"Sources: {weights_str}",
        f"Results: {total_results}",
        "",
        "---",
        "",
    ]

    # Source display labels and ordering
    source_labels = {
        "vault": "Vault",
        "conversations": "AI Conversations",
        "saved_items": "The Source",
    }
    source_order = ["vault", "conversations", "saved_items"]

    results = data.get("results", [])

    for source_key in source_order:
        source_results = [r for r in results if r.get("source") == source_key]
        if not source_results:
            continue

        label = source_labels.get(source_key, source_key)
        lines.append(f"## Source: {label} ({len(source_results)} results)")
        lines.append("")

        for i, r in enumerate(source_results, 1):
            score = r.get("weighted_score", 0.0)
            title = r.get("title", "Untitled")
            chunk = r.get("chunk", "")

            lines.append(f"### {i}. [{score:.2f}] {title}")

            # Chunk preview (first 500 chars)
            chunk_preview = chunk[:500]
            if len(chunk) > 500:
                chunk_preview += "..."
            lines.append(f"> {chunk_preview}")
            lines.append("")

            # Source-specific metadata
            metadata = r.get("metadata", {})
            for key, value in metadata.items():
                if value:  # Skip empty values
                    lines.append(f"**{key}:** {value}")

            lines.append("")
            lines.append("---")
            lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Exported markdown: {filepath}")

    # Update exported_as in internal storage
    if "markdown" not in data.get("exported_as", []):
        data.setdefault("exported_as", []).append("markdown")
        _update_internal_storage(result_id, data)

        # Also update history entry
        _update_history_exported_as(result_id, data["exported_as"])

    return filepath


def export_json(result_id: str) -> Optional[Path]:
    """Export results as JSON to the exports directory.

    Same structure as internal storage but WITHOUT the dedup_log field.
    Updates the internal storage's exported_as field.

    Args:
        result_id: The unique result identifier.

    Returns:
        Path to the exported JSON file, or None if result_id not found.
    """
    data = load_results(result_id)
    if not data:
        logger.error(f"Cannot export JSON: result_id={result_id} not found")
        return None

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    slug = _slugify(data["query"])
    date_str = data["timestamp"][:10]
    filename = f"{date_str}_{slug}.json"
    filepath = EXPORTS_DIR / filename

    # Export everything except dedup_log
    export_data = {k: v for k, v in data.items() if k != "dedup_log"}

    filepath.write_text(
        json.dumps(export_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    logger.info(f"Exported JSON: {filepath}")

    # Update exported_as in internal storage
    if "json" not in data.get("exported_as", []):
        data.setdefault("exported_as", []).append("json")
        _update_internal_storage(result_id, data)

        # Also update history entry
        _update_history_exported_as(result_id, data["exported_as"])

    return filepath


def _update_history_exported_as(result_id: str, exported_as: List[str]) -> None:
    """Update the exported_as field in the history for a given result_id.

    Args:
        result_id: The unique result identifier.
        exported_as: Updated list of export formats.
    """
    history = get_history()
    updated = False
    for entry in history:
        if entry.get("id") == result_id:
            entry["exported_as"] = exported_as
            updated = True
            break

    if updated:
        HISTORY_FILE.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.debug(f"Updated history exported_as for {result_id}: {exported_as}")
