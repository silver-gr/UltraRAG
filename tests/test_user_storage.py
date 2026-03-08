from __future__ import annotations

import pytest

from user_storage import (
    sanitize_username,
    user_root,
    user_query_history_path,
    user_content_research_dir,
)


@pytest.mark.unit
def test_sanitize_username_blocks_traversal() -> None:
    sanitized = sanitize_username("../../Admin User")
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert "\\" not in sanitized


@pytest.mark.unit
def test_user_paths_are_scoped_per_user() -> None:
    username = "Silver.User"
    safe = sanitize_username(username)

    assert user_root(username).as_posix() == f"data/users/{safe}"
    assert user_query_history_path(username).as_posix() == f"data/users/{safe}/query_history.json"
    assert user_content_research_dir(username).as_posix() == f"data/users/{safe}/research-exports"
