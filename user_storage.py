"""Per-user local storage paths."""
from __future__ import annotations

import re
from pathlib import Path


def sanitize_username(name: str) -> str:
    """Sanitize a username for safe filesystem directory usage."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", (name or "").strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or "user"


def user_root(username: str) -> Path:
    """Return the root directory for a user's data."""
    return Path("data") / "users" / sanitize_username(username)


def user_query_history_path(username: str) -> Path:
    """Return per-user query history JSON path."""
    return user_root(username) / "query_history.json"


def user_content_research_dir(username: str) -> Path:
    """Return per-user content research export directory."""
    return user_root(username) / "research-exports"
