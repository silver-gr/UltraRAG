"""Research storage and export module for UltraRAG Content Research.

Results are stored per-user for privacy-safe multi-user deployments.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from user_storage import user_content_research_dir

logger = logging.getLogger(__name__)


def get_exports_dir(username: str, base_dir: str | Path | None = None) -> Path:
    """Resolve per-user export directory with optional env/file override."""
    if base_dir is not None:
        return Path(base_dir)

    env_base = os.getenv("ULTRARAG_CONTENT_RESEARCH_DIR", "").strip()
    if env_base:
        return Path(env_base) / username

    return user_content_research_dir(username)


def _internal_dir(username: str, base_dir: str | Path | None = None) -> Path:
    return get_exports_dir(username, base_dir=base_dir) / ".internal"


def _history_file(username: str, base_dir: str | Path | None = None) -> Path:
    return get_exports_dir(username, base_dir=base_dir) / ".query_history.json"


def _slugify(text: str, max_length: int = 30) -> str:
    """Convert query text to a filesystem-safe slug."""
    slug = text.lower().replace(" ", "-")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug[:max_length]


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to native Python for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a ResearchResult object to a dict, excluding the embedding field."""
    try:
        from dataclasses import asdict

        data = asdict(result)
    except (TypeError, ImportError):
        data = {
            "rank": getattr(result, "rank", 0),
            "source": getattr(result, "source", "unknown"),
            "title": getattr(result, "title", ""),
            "chunk": getattr(result, "chunk", ""),
            "raw_score": getattr(result, "raw_score", 0.0),
            "weighted_score": getattr(result, "weighted_score", 0.0),
            "metadata": getattr(result, "metadata", {}),
        }

    data.pop("embedding", None)
    return _make_serializable(data)


def _get_internal_filepath(query: str, result_id: str, username: str, base_dir: str | Path | None = None) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = _slugify(query)
    filename = f"{date_str}_{slug}_{result_id}.json"
    return _internal_dir(username, base_dir=base_dir) / filename


def _update_internal_storage(
    result_id: str,
    data: Dict[str, Any],
    username: str,
    base_dir: str | Path | None = None,
) -> None:
    internal_dir = _internal_dir(username, base_dir=base_dir)
    for filepath in internal_dir.glob(f"*_{result_id}.json"):
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Updated internal storage: %s", filepath.name)
        return

    logger.warning("Could not find internal file for result_id=%s to update", result_id)


def save_results(
    query_settings: Dict[str, Any],
    results: List[Any],
    dedup_log: List[Dict[str, Any]],
    username: str,
    base_dir: str | Path | None = None,
) -> str:
    """Save search results internally for a single user."""
    internal_dir = _internal_dir(username, base_dir=base_dir)
    internal_dir.mkdir(parents=True, exist_ok=True)

    result_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().isoformat(timespec="seconds")
    query = query_settings.get("query", "")

    serialized_results = []
    for idx, result in enumerate(results, start=1):
        result_dict = _result_to_dict(result)
        result_dict["rank"] = idx
        serialized_results.append(result_dict)

    source_counts = Counter(
        r.get("source", "unknown") if isinstance(r, dict) else getattr(r, "source", "unknown")
        for r in results
    )

    data = {
        "id": result_id,
        "query": query,
        "timestamp": timestamp,
        "settings": query_settings,
        "stats": {
            "total_results": len(results),
            "per_source": dict(source_counts),
        },
        "results": serialized_results,
        "dedup_log": dedup_log,
        "exported_as": [],
    }

    filepath = _get_internal_filepath(query, result_id, username, base_dir=base_dir)
    filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved results internally: %s (%d results)", filepath.name, len(results))

    update_history(result_id, data, username, base_dir=base_dir)
    return result_id


def load_results(
    result_id: str,
    username: str,
    base_dir: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    """Load full results from internal storage by ID for a specific user."""
    internal_dir = _internal_dir(username, base_dir=base_dir)
    if not internal_dir.exists():
        logger.warning("Internal directory does not exist: %s", internal_dir)
        return None

    matches = list(internal_dir.glob(f"*_{result_id}.json"))
    if not matches:
        logger.warning("No internal file found for result_id=%s", result_id)
        return None

    filepath = matches[0]
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error loading results from %s: %s", filepath, exc)
        return None


def get_history(username: str, base_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    """Get per-user query history, sorted by newest first."""
    history_file = _history_file(username, base_dir=base_dir)
    if not history_file.exists():
        return []

    try:
        data = json.loads(history_file.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error reading history file %s: %s", history_file, exc)
        return []


def update_history(
    result_id: str,
    data: Dict[str, Any],
    username: str,
    base_dir: str | Path | None = None,
) -> None:
    """Add or update an entry in the per-user query history index."""
    exports_dir = get_exports_dir(username, base_dir=base_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    history = get_history(username, base_dir=base_dir)
    entry = {
        "id": result_id,
        "query": data.get("query", ""),
        "timestamp": data.get("timestamp", ""),
        "stats": data.get("stats", {}),
        "settings": data.get("settings", {}),
        "exported_as": data.get("exported_as", []),
    }

    history = [item for item in history if item.get("id") != result_id]
    history.insert(0, entry)

    history_file = _history_file(username, base_dir=base_dir)
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def clear_history(username: str, base_dir: str | Path | None = None) -> None:
    """Delete per-user history file and internal result files."""
    history_file = _history_file(username, base_dir=base_dir)
    if history_file.exists():
        history_file.unlink()
        logger.info("Deleted query history file: %s", history_file)

    internal_dir = _internal_dir(username, base_dir=base_dir)
    if internal_dir.exists():
        removed = 0
        for filepath in internal_dir.glob("*.json"):
            filepath.unlink()
            removed += 1
        logger.info("Deleted %d internal result files for user=%s", removed, username)


def export_markdown(
    result_id: str,
    username: str,
    base_dir: str | Path | None = None,
) -> Optional[Path]:
    """Export results as formatted Markdown to per-user exports directory."""
    data = load_results(result_id, username=username, base_dir=base_dir)
    if not data:
        logger.error("Cannot export markdown: result_id=%s not found", result_id)
        return None

    exports_dir = get_exports_dir(username, base_dir=base_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(data["query"])
    date_str = data["timestamp"][:10]
    filepath = exports_dir / f"{date_str}_{slug}.md"

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

        for idx, result in enumerate(source_results, 1):
            score = result.get("weighted_score", 0.0)
            title = result.get("title", "Untitled")
            chunk = result.get("chunk", "")

            lines.append(f"### {idx}. [{score:.2f}] {title}")
            lines.append(f"> {chunk}")
            lines.append("")

            metadata = result.get("metadata", {})
            for key, value in metadata.items():
                if value:
                    lines.append(f"**{key}:** {value}")

            lines.append("")
            lines.append("---")
            lines.append("")

    filepath.write_text("\n".join(lines), encoding="utf-8")

    if "markdown" not in data.get("exported_as", []):
        data.setdefault("exported_as", []).append("markdown")
        _update_internal_storage(result_id, data, username=username, base_dir=base_dir)
        _update_history_exported_as(result_id, data["exported_as"], username=username, base_dir=base_dir)

    return filepath


def export_json(
    result_id: str,
    username: str,
    base_dir: str | Path | None = None,
) -> Optional[Path]:
    """Export results as JSON to per-user exports directory."""
    data = load_results(result_id, username=username, base_dir=base_dir)
    if not data:
        logger.error("Cannot export JSON: result_id=%s not found", result_id)
        return None

    exports_dir = get_exports_dir(username, base_dir=base_dir)
    exports_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(data["query"])
    date_str = data["timestamp"][:10]
    filepath = exports_dir / f"{date_str}_{slug}.json"

    export_data = {k: v for k, v in data.items() if k != "dedup_log"}
    filepath.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if "json" not in data.get("exported_as", []):
        data.setdefault("exported_as", []).append("json")
        _update_internal_storage(result_id, data, username=username, base_dir=base_dir)
        _update_history_exported_as(result_id, data["exported_as"], username=username, base_dir=base_dir)

    return filepath


def _update_history_exported_as(
    result_id: str,
    exported_as: List[str],
    username: str,
    base_dir: str | Path | None = None,
) -> None:
    history = get_history(username=username, base_dir=base_dir)
    updated = False

    for entry in history:
        if entry.get("id") == result_id:
            entry["exported_as"] = exported_as
            updated = True
            break

    if updated:
        history_file = _history_file(username, base_dir=base_dir)
        history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
