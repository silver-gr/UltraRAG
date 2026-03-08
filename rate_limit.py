"""App-level concurrency and per-user query cooldown controls."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

import streamlit as st


class RateLimitError(Exception):
    """Base rate-limit exception."""


class QueryCooldownError(RateLimitError):
    """Raised when per-user query cooldown is active."""

    def __init__(self, remaining_seconds: float):
        super().__init__(f"Please wait {remaining_seconds:.1f}s before the next query.")
        self.remaining_seconds = remaining_seconds


class SystemBusyError(RateLimitError):
    """Raised when global worker capacity cannot be acquired."""


_MAX_CONCURRENT = max(1, int(os.getenv("ULTRARAG_MAX_CONCURRENT_JOBS", "1")))
_MIN_SECONDS_BETWEEN_QUERIES = max(0.0, float(os.getenv("ULTRARAG_MIN_SECONDS_BETWEEN_QUERIES", "2")))
_ACQUIRE_TIMEOUT_SECONDS = max(1.0, float(os.getenv("ULTRARAG_JOB_ACQUIRE_TIMEOUT_SECONDS", "120")))
_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT)


def _query_cooldown_key(username: str) -> str:
    return f"ultrarag_last_query_ts_{username}"


def _is_query_job(job_name: str) -> bool:
    lowered = (job_name or "").lower()
    return any(token in lowered for token in ("query", "search", "research", "raptor"))


def _enforce_cooldown(username: str) -> None:
    if _MIN_SECONDS_BETWEEN_QUERIES <= 0:
        return

    now = time.time()
    key = _query_cooldown_key(username)
    last_ts = float(st.session_state.get(key, 0.0))
    elapsed = now - last_ts
    if elapsed < _MIN_SECONDS_BETWEEN_QUERIES:
        raise QueryCooldownError(_MIN_SECONDS_BETWEEN_QUERIES - elapsed)


def _mark_query(username: str) -> None:
    st.session_state[_query_cooldown_key(username)] = time.time()


@contextmanager
def run_with_limits(ctx, job_name: str):
    """Run a job with global concurrency and per-user query throttling."""
    username = getattr(ctx, "username", "anonymous")
    is_query_job = _is_query_job(job_name)

    if is_query_job:
        _enforce_cooldown(username)

    acquired = _SEMAPHORE.acquire(blocking=False)
    if not acquired:
        with st.spinner("System busy. Waiting for available capacity..."):
            acquired = _SEMAPHORE.acquire(timeout=_ACQUIRE_TIMEOUT_SECONDS)

    if not acquired:
        raise SystemBusyError("System is busy. Please try again shortly.")

    try:
        yield
    finally:
        _SEMAPHORE.release()
        if is_query_job:
            _mark_query(username)
