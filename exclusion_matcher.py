"""Pattern matching for file exclusions."""
from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ExclusionMatcher:
    """Efficient pattern matcher for file exclusions.

    Supports three pattern types:
    - exact: Exact string match (O(1) lookup)
    - glob: Shell-style wildcards (fnmatch)
    - regex: Regular expressions
    """

    def __init__(self, patterns: list[dict] | None = None):
        """Initialize matcher with patterns.

        Args:
            patterns: List of pattern dicts:
                [{"pattern": "...", "type": "exact|glob|regex"}, ...]
        """
        self.exact: set[str] = set()  # O(1) lookup
        self.globs: list[str] = []  # fnmatch patterns
        self.regexes: list[tuple[str, re.Pattern]] = []  # (original, compiled)

        if patterns:
            self._compile_patterns(patterns)

    def _compile_patterns(self, patterns: list[dict]) -> None:
        """Compile patterns into efficient lookup structures."""
        for p in patterns:
            pattern = p.get("pattern", "")
            pattern_type = p.get("type", "glob")

            if not pattern:
                continue

            if pattern_type == "exact":
                # Normalize path separators
                self.exact.add(pattern.replace("\\", "/"))

            elif pattern_type == "glob":
                self.globs.append(pattern)

            elif pattern_type == "regex":
                try:
                    compiled = re.compile(pattern)
                    self.regexes.append((pattern, compiled))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")

        logger.debug(
            f"Compiled {len(self.exact)} exact, "
            f"{len(self.globs)} glob, "
            f"{len(self.regexes)} regex patterns"
        )

    def is_excluded(self, file_path: str) -> tuple[bool, str | None]:
        """Check if a path is excluded.

        Args:
            file_path: Path to check (relative to vault root, normalized with /)

        Returns:
            Tuple of (is_excluded, matching_pattern or None)
        """
        # Normalize path
        normalized = file_path.replace("\\", "/")

        # 1. Check exact match (O(1))
        if normalized in self.exact:
            return True, normalized

        # 2. Check glob patterns
        for glob_pattern in self.globs:
            if fnmatch.fnmatch(normalized, glob_pattern):
                return True, glob_pattern

        # 3. Check regex patterns
        for original, compiled in self.regexes:
            if compiled.search(normalized):
                return True, original

        return False, None

    def get_excluded_files(
        self,
        all_files: list[Path],
        vault_root: Path
    ) -> list[tuple[Path, str]]:
        """Get all files that match exclusions.

        Args:
            all_files: List of absolute file paths
            vault_root: Root path to make paths relative

        Returns:
            List of (absolute_path, matching_pattern) tuples
        """
        excluded = []
        for file_path in all_files:
            try:
                relative = str(file_path.relative_to(vault_root)).replace("\\", "/")
                is_exc, pattern = self.is_excluded(relative)
                if is_exc:
                    excluded.append((file_path, pattern))
            except ValueError:
                # File is not relative to vault_root
                continue

        return excluded

    def filter_files(
        self,
        all_files: list[Path],
        vault_root: Path
    ) -> tuple[list[Path], int]:
        """Filter out excluded files.

        Args:
            all_files: List of absolute file paths
            vault_root: Root path to make paths relative

        Returns:
            Tuple of (included_files, excluded_count)
        """
        included = []
        excluded_count = 0

        for file_path in all_files:
            try:
                relative = str(file_path.relative_to(vault_root)).replace("\\", "/")
                is_exc, _ = self.is_excluded(relative)
                if is_exc:
                    excluded_count += 1
                else:
                    included.append(file_path)
            except ValueError:
                # File is not relative to vault_root, include it
                included.append(file_path)

        return included, excluded_count

    @property
    def pattern_count(self) -> int:
        """Total number of patterns."""
        return len(self.exact) + len(self.globs) + len(self.regexes)

    def __bool__(self) -> bool:
        """Return True if any patterns are configured."""
        return self.pattern_count > 0


def preview_exclusions(
    patterns: list[dict],
    vault_path: Path,
    file_pattern: str = "*.md"
) -> dict:
    """Preview which files would be excluded by given patterns.

    Args:
        patterns: List of exclusion pattern dicts
        vault_path: Path to vault root
        file_pattern: Glob pattern for files to scan (default: *.md)

    Returns:
        Dict with:
        - total_files: Total files scanned
        - excluded_files: List of (relative_path, matching_pattern)
        - excluded_count: Number of excluded files
        - included_count: Number of included files
    """
    matcher = ExclusionMatcher(patterns)

    all_files = list(vault_path.rglob(file_pattern))
    excluded = matcher.get_excluded_files(all_files, vault_path)

    excluded_paths = [
        (str(f.relative_to(vault_path)), pattern)
        for f, pattern in excluded
    ]

    return {
        "total_files": len(all_files),
        "excluded_files": excluded_paths,
        "excluded_count": len(excluded_paths),
        "included_count": len(all_files) - len(excluded_paths),
    }
