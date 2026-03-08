from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import streamlit as st

import rate_limit


@pytest.mark.unit
def test_query_cooldown_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rate_limit, "_MIN_SECONDS_BETWEEN_QUERIES", 2.0)
    monkeypatch.setattr(rate_limit, "_SEMAPHORE", threading.BoundedSemaphore(1))

    ctx = SimpleNamespace(username="tester")
    st.session_state["ultrarag_last_query_ts_tester"] = time.time()

    with pytest.raises(rate_limit.QueryCooldownError):
        with rate_limit.run_with_limits(ctx, "search_query"):
            pass


@pytest.mark.unit
def test_semaphore_acquire_release_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(rate_limit, "_MIN_SECONDS_BETWEEN_QUERIES", 0.0)
    monkeypatch.setattr(rate_limit, "_SEMAPHORE", sem)

    ctx = SimpleNamespace(username="tester")

    with rate_limit.run_with_limits(ctx, "search_query"):
        assert not sem.acquire(blocking=False)

    # Must be released after context exit
    assert sem.acquire(blocking=False)
    sem.release()
