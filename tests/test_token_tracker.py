"""Tests for Voyage token tracking (token_tracker.py)."""

import json

import pytest

from token_tracker import VoyageTokenTracker


@pytest.mark.unit
class TestModelSwitch:
    """Switching models must not carry the old model's usage over."""

    def test_rerank_model_switch_archives_previous_usage(self, tmp_path):
        path = tmp_path / "voyage_usage.json"
        tracker = VoyageTokenTracker(storage_path=path)

        tracker.record_rerank_usage(1000, model="rerank-2.5")
        tracker.record_rerank_usage(500, model="rerank-2.5")
        tracker.record_rerank_usage(200, model="rerank-3")

        assert tracker.rerank_usage.model == "rerank-3"
        assert tracker.rerank_usage.tokens_used == 200
        assert tracker.rerank_usage.requests_count == 1
        assert [h["model"] for h in tracker.rerank_history] == ["rerank-2.5"]
        assert tracker.rerank_history[0]["tokens_used"] == 1500

        saved = json.loads(path.read_text())
        assert saved["rerank_history"][0]["tokens_used"] == 1500
        assert VoyageTokenTracker(storage_path=path).rerank_history == tracker.rerank_history

    def test_embedding_model_switch_archives_previous_usage(self, tmp_path):
        tracker = VoyageTokenTracker(storage_path=tmp_path / "voyage_usage.json")

        tracker.record_embedding_usage(1000, model="voyage-4-lite")
        tracker.record_embedding_usage(300, model="voyage-4-large", usage_type="query")

        assert tracker.embedding_usage.model == "voyage-4-large"
        assert tracker.embedding_usage.tokens_used == 300
        assert [h["model"] for h in tracker.embedding_history] == ["voyage-4-lite"]
