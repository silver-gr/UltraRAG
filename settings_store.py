"""Generic settings storage using LanceDB."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

# Settings table schema
SETTINGS_SCHEMA = pa.schema([
    pa.field("key", pa.string()),
    pa.field("value", pa.string()),  # JSON-encoded
    pa.field("updated_at", pa.string()),  # ISO timestamp
])


def _list_tables(db: Any) -> list[str]:
    """Return LanceDB table names across API versions."""
    tables_resp = db.list_tables()
    return tables_resp.tables if hasattr(tables_resp, "tables") else tables_resp


class SettingsStore:
    """Generic key-value settings storage in LanceDB."""

    TABLE_NAME = "settings"

    def __init__(self, db_path: str | Path):
        """Initialize settings store.

        Args:
            db_path: Path to LanceDB database directory
        """
        self.db_path = Path(db_path)
        self.db = lancedb.connect(str(self.db_path))
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create settings table if it doesn't exist."""
        if self.TABLE_NAME not in _list_tables(self.db):
            # Create empty table with schema
            self.db.create_table(
                self.TABLE_NAME,
                schema=SETTINGS_SCHEMA,
                mode="overwrite"
            )
            logger.info(f"Created settings table in {self.db_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value (JSON decoded).

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Decoded value or default
        """
        try:
            table = self.db.open_table(self.TABLE_NAME)
            results = table.search().where(f"key = '{key}'", prefilter=True).limit(1).to_list()

            if not results:
                return default

            return json.loads(results[0]["value"])
        except Exception as e:
            logger.warning(f"Failed to get setting '{key}': {e}")
            return default

    def set(self, key: str, value: Any) -> None:
        """Set setting value (JSON encoded).

        Args:
            key: Setting key
            value: Value to store (will be JSON encoded)
        """
        try:
            table = self.db.open_table(self.TABLE_NAME)

            # Delete existing key if present
            try:
                table.delete(f"key = '{key}'")
            except Exception:
                pass  # Key might not exist

            # Insert new value
            table.add([{
                "key": key,
                "value": json.dumps(value),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }])

            logger.debug(f"Set setting '{key}'")
        except Exception as e:
            logger.error(f"Failed to set setting '{key}': {e}")
            raise

    def delete(self, key: str) -> bool:
        """Delete a setting.

        Args:
            key: Setting key to delete

        Returns:
            True if deleted, False if key didn't exist
        """
        try:
            table = self.db.open_table(self.TABLE_NAME)
            # Check if exists first
            results = table.search().where(f"key = '{key}'", prefilter=True).limit(1).to_list()
            if not results:
                return False

            table.delete(f"key = '{key}'")
            logger.debug(f"Deleted setting '{key}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete setting '{key}': {e}")
            return False

    def list_keys(self) -> list[str]:
        """List all setting keys.

        Returns:
            List of keys
        """
        try:
            table = self.db.open_table(self.TABLE_NAME)
            df = table.to_pandas()
            return df["key"].tolist()
        except Exception as e:
            logger.warning(f"Failed to list settings: {e}")
            return []


# =============================================================================
# Exclusion-specific convenience functions
# =============================================================================

EXCLUSIONS_KEY = "file_exclusions"


def get_exclusions(db_path: str | Path) -> list[dict]:
    """Get all exclusion patterns.

    Args:
        db_path: Path to LanceDB database

    Returns:
        List of exclusion dicts: [{"pattern": "...", "type": "glob|exact|regex", "added_at": "..."}, ...]
    """
    store = SettingsStore(db_path)
    data = store.get(EXCLUSIONS_KEY, {"patterns": []})
    return data.get("patterns", [])


def add_exclusion(db_path: str | Path, pattern: str, pattern_type: str = "glob") -> None:
    """Add an exclusion pattern.

    Args:
        db_path: Path to LanceDB database
        pattern: Pattern string
        pattern_type: One of "exact", "glob", "regex"
    """
    if pattern_type not in ("exact", "glob", "regex"):
        raise ValueError(f"Invalid pattern type: {pattern_type}. Must be 'exact', 'glob', or 'regex'")

    store = SettingsStore(db_path)
    data = store.get(EXCLUSIONS_KEY, {"patterns": []})
    patterns = data.get("patterns", [])

    # Check for duplicate
    if any(p["pattern"] == pattern and p["type"] == pattern_type for p in patterns):
        logger.info(f"Exclusion pattern already exists: {pattern} ({pattern_type})")
        return

    patterns.append({
        "pattern": pattern,
        "type": pattern_type,
        "added_at": datetime.now(timezone.utc).isoformat(),
    })

    store.set(EXCLUSIONS_KEY, {"patterns": patterns})
    logger.info(f"Added exclusion: {pattern} ({pattern_type})")


def remove_exclusion(db_path: str | Path, pattern: str, pattern_type: str | None = None) -> bool:
    """Remove an exclusion pattern.

    Args:
        db_path: Path to LanceDB database
        pattern: Pattern string to remove
        pattern_type: Optional type filter (if None, removes all matching patterns)

    Returns:
        True if removed, False if not found
    """
    store = SettingsStore(db_path)
    data = store.get(EXCLUSIONS_KEY, {"patterns": []})
    patterns = data.get("patterns", [])

    original_count = len(patterns)

    if pattern_type:
        patterns = [p for p in patterns if not (p["pattern"] == pattern and p["type"] == pattern_type)]
    else:
        patterns = [p for p in patterns if p["pattern"] != pattern]

    if len(patterns) == original_count:
        return False

    store.set(EXCLUSIONS_KEY, {"patterns": patterns})
    logger.info(f"Removed exclusion: {pattern}")
    return True


def clear_exclusions(db_path: str | Path) -> int:
    """Clear all exclusion patterns.

    Args:
        db_path: Path to LanceDB database

    Returns:
        Number of patterns cleared
    """
    store = SettingsStore(db_path)
    data = store.get(EXCLUSIONS_KEY, {"patterns": []})
    count = len(data.get("patterns", []))

    store.set(EXCLUSIONS_KEY, {"patterns": []})
    logger.info(f"Cleared {count} exclusion patterns")
    return count
