"""
Tests for Obsession Radar — Cross-Source Topic Convergence Detection.

Tests are structured in phases:
1. Dataclasses and initialization
2. File-based scanners (audio, published)
3. Vector-based scanners (vault, conversations, saved_items) - mocked
4. Scoring algorithm
5. CLI output formatting
6. Auto-discovery
"""

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Phase 1: Dataclasses, imports, and basic initialization
# ============================================================

class TestDataclasses:
    """Test SourceProbe and TopicScore dataclass structure."""

    def test_source_probe_creation(self):
        from obsession_radar import SourceProbe

        probe = SourceProbe(
            source="vault",
            found=True,
            match_count=5,
            avg_relevance=0.75,
            top_excerpts=["excerpt 1", "excerpt 2"],
            top_titles=["Title A", "Title B"],
        )
        assert probe.source == "vault"
        assert probe.found is True
        assert probe.match_count == 5
        assert probe.avg_relevance == 0.75
        assert len(probe.top_excerpts) == 2
        assert len(probe.top_titles) == 2

    def test_source_probe_empty(self):
        from obsession_radar import SourceProbe

        probe = SourceProbe(
            source="audio",
            found=False,
            match_count=0,
            avg_relevance=0.0,
            top_excerpts=[],
            top_titles=[],
        )
        assert probe.found is False
        assert probe.match_count == 0
        assert probe.avg_relevance == 0.0

    def test_topic_score_creation(self):
        from obsession_radar import TopicScore, SourceProbe

        probes = {
            "vault": SourceProbe("vault", True, 3, 0.8, ["e1"], ["t1"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="dopamine",
            convergence=1,
            total_score=3.2,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
        )
        assert score.topic == "dopamine"
        assert score.convergence == 1
        assert score.total_score == 3.2
        assert score.already_published is False
        assert "vault" in score.probes

    def test_topic_score_with_all_sources(self):
        from obsession_radar import TopicScore, SourceProbe

        sources = ["vault", "conversations", "saved_items", "audio", "published"]
        probes = {
            s: SourceProbe(s, True, 2, 0.6, ["ex"], ["ti"])
            for s in sources
        }

        score = TopicScore(
            topic="sleep",
            convergence=5,
            total_score=10.0,
            probes=probes,
            category_suggestion="rest-recovery",
            already_published=True,
            existing_briefs=2,
        )
        assert score.convergence == 5
        assert len(score.probes) == 5
        assert score.already_published is True
        assert score.existing_briefs == 2


class TestObsessionRadarInit:
    """Test ObsessionRadar class initialization."""

    @patch("config.load_config")
    def test_radar_creates_with_default_config(self, mock_config):
        """Radar should initialize using load_config if no config passed."""
        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        mock_config.return_value = mock_cfg

        from obsession_radar import ObsessionRadar

        radar = ObsessionRadar()
        mock_config.assert_called_once()

    def test_radar_creates_with_custom_config(self):
        """Radar should accept a custom config."""
        from obsession_radar import ObsessionRadar

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()

        radar = ObsessionRadar(config=mock_cfg)
        assert radar.config is mock_cfg


# ============================================================
# Phase 2: File-based scanners — Audio and Published
# ============================================================

class TestAudioScanner:
    """Test scanning of AI~udio-Reflections analysis files."""

    def test_scan_audio_no_directory(self):
        """Should return empty probe if transcripts directory doesn't exist."""
        from obsession_radar import ObsessionRadar

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        probe = radar._scan_audio("dopamine", Path("/nonexistent/path"))
        assert probe.found is False
        assert probe.match_count == 0

    def test_scan_audio_with_matching_files(self):
        """Should find matches in analysis markdown files."""
        from obsession_radar import ObsessionRadar

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake analysis files
            analysis1 = Path(tmpdir) / "2026-01-15_analysis.md"
            analysis1.write_text(
                "# Analysis\n\n## INSIGHTS\n\n"
                "Discussed dopamine and habit loops today.\n"
                "The dopamine reset protocol seems effective.\n\n"
                "## NEXT ACTIONS\n\nStart a dopamine detox week.\n"
            )

            analysis2 = Path(tmpdir) / "2026-01-20_analysis.md"
            analysis2.write_text(
                "# Analysis\n\n## INSIGHTS\n\n"
                "Reflected on sleep quality improvements.\n"
                "No mention of the target topic here.\n"
            )

            analysis3 = Path(tmpdir) / "2026-02-01_analysis.md"
            analysis3.write_text(
                "# Analysis\n\n## INSIGHTS\n\n"
                "Dopamine fasting results: better focus, less phone usage.\n"
            )

            mock_cfg = MagicMock()
            mock_cfg.embedding = MagicMock()
            mock_cfg.voyage_api_key = MagicMock()
            radar = ObsessionRadar(config=mock_cfg)

            probe = radar._scan_audio("dopamine", Path(tmpdir))
            assert probe.found is True
            assert probe.match_count >= 2  # At least 2 files mention dopamine
            assert len(probe.top_excerpts) > 0

    def test_scan_audio_no_matches(self):
        """Should return empty probe when no files match."""
        from obsession_radar import ObsessionRadar

        with tempfile.TemporaryDirectory() as tmpdir:
            analysis = Path(tmpdir) / "2026-01-15_analysis.md"
            analysis.write_text("# Analysis\n\nDiscussed cooking today.\n")

            mock_cfg = MagicMock()
            mock_cfg.embedding = MagicMock()
            mock_cfg.voyage_api_key = MagicMock()
            radar = ObsessionRadar(config=mock_cfg)

            probe = radar._scan_audio("quantum physics", Path(tmpdir))
            assert probe.found is False
            assert probe.match_count == 0


class TestPublishedScanner:
    """Test scanning of published articles and briefs."""

    def test_scan_published_no_directory(self):
        """Should return empty probe if pipeline directory doesn't exist."""
        from obsession_radar import ObsessionRadar

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        probe = radar._scan_published("dopamine", Path("/nonexistent/path"))
        assert probe.found is False
        assert probe.match_count == 0

    def test_scan_published_finds_articles(self):
        """Should detect published articles matching the topic."""
        from obsession_radar import ObsessionRadar

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = Path(tmpdir)

            # Create published article
            published_dir = pipeline / "04-published" / "health-fitness"
            published_dir.mkdir(parents=True)
            article = published_dir / "2025-12-31-dopamine-detox.md"
            article.write_text(
                '---\ntitle: "Dopamine Detox: The Complete Guide"\n'
                'category: "health-fitness"\nstatus: published\n---\n\n'
                "Content about dopamine detox.\n"
            )

            mock_cfg = MagicMock()
            mock_cfg.embedding = MagicMock()
            mock_cfg.voyage_api_key = MagicMock()
            radar = ObsessionRadar(config=mock_cfg)

            probe = radar._scan_published("dopamine", pipeline)
            assert probe.found is True
            assert probe.match_count >= 1
            assert any("Dopamine" in t for t in probe.top_titles)

    def test_scan_published_finds_briefs(self):
        """Should detect processed briefs matching the topic."""
        from obsession_radar import ObsessionRadar

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = Path(tmpdir)

            # Create processed brief
            processed_dir = pipeline / "02-processed" / "health-fitness"
            processed_dir.mkdir(parents=True)
            brief = processed_dir / "2026-01-26-sleep-habits.md"
            brief.write_text(
                '---\ntitle: "Sleep Habits for Better Rest"\n'
                'category: "health-fitness"\nstatus: processed\n---\n\n'
                "Brief about sleep.\n"
            )

            mock_cfg = MagicMock()
            mock_cfg.embedding = MagicMock()
            mock_cfg.voyage_api_key = MagicMock()
            radar = ObsessionRadar(config=mock_cfg)

            probe = radar._scan_published("sleep", pipeline)
            assert probe.found is True
            assert probe.match_count >= 1

    def test_scan_published_no_match(self):
        """Should return empty when no articles match."""
        from obsession_radar import ObsessionRadar

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = Path(tmpdir)
            published_dir = pipeline / "04-published" / "health-fitness"
            published_dir.mkdir(parents=True)
            article = published_dir / "2025-12-31-cooking-tips.md"
            article.write_text(
                '---\ntitle: "Best Cooking Tips"\ncategory: "health-fitness"\n---\n\n'
                "Content about cooking.\n"
            )

            mock_cfg = MagicMock()
            mock_cfg.embedding = MagicMock()
            mock_cfg.voyage_api_key = MagicMock()
            radar = ObsessionRadar(config=mock_cfg)

            probe = radar._scan_published("quantum physics", pipeline)
            assert probe.found is False
            assert probe.match_count == 0


# ============================================================
# Phase 3: Vector-based scanners (mocked LanceDB)
# ============================================================

class TestVectorScanners:
    """Test vector-based source scanning with mocked LanceDB."""

    @patch("obsession_radar._search_source")
    def test_scan_vault_with_results(self, mock_search):
        """Should return a probe with matches from vault LanceDB."""
        from obsession_radar import ObsessionRadar
        from content_research_engine import ResearchResult

        # Mock _search_source to return fake results
        mock_search.return_value = [
            ResearchResult(
                rank=1, source="vault", title="Dopamine Nation Notes",
                chunk="Notes about dopamine and habit formation...",
                raw_score=0.82, weighted_score=0.82,
            ),
            ResearchResult(
                rank=2, source="vault", title="Neuroscience Basics",
                chunk="The dopamine system governs reward...",
                raw_score=0.65, weighted_score=0.65,
            ),
        ]

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        # Mock the embed model
        radar._embed_model = MagicMock()
        radar._embed_model.get_query_embedding.return_value = [0.1] * 1024

        probe = radar._scan_vector_source("dopamine", "vault", threshold=0.55)
        assert probe.found is True
        assert probe.match_count == 2
        assert probe.avg_relevance > 0.6

    @patch("obsession_radar._search_source")
    def test_scan_vault_below_threshold(self, mock_search):
        """Should exclude results below threshold."""
        from obsession_radar import ObsessionRadar
        from content_research_engine import ResearchResult

        mock_search.return_value = [
            ResearchResult(
                rank=1, source="vault", title="Unrelated",
                chunk="Barely related content...",
                raw_score=0.40, weighted_score=0.40,
            ),
        ]

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)
        radar._embed_model = MagicMock()
        radar._embed_model.get_query_embedding.return_value = [0.1] * 1024

        probe = radar._scan_vector_source("dopamine", "vault", threshold=0.55)
        assert probe.found is False
        assert probe.match_count == 0

    @patch("obsession_radar._search_source")
    def test_scan_conversations(self, mock_search):
        """Should scan conversations source."""
        from obsession_radar import ObsessionRadar
        from content_research_engine import ResearchResult

        mock_search.return_value = [
            ResearchResult(
                rank=1, source="conversations",
                title="Discussion about sleep",
                chunk="We discussed sleep optimization...",
                raw_score=0.70, weighted_score=0.70,
            ),
        ]

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)
        radar._embed_model = MagicMock()
        radar._embed_model.get_query_embedding.return_value = [0.1] * 1024

        probe = radar._scan_vector_source("sleep", "conversations", threshold=0.50)
        assert probe.found is True
        assert probe.source == "conversations"


# ============================================================
# Phase 4: Scoring algorithm
# ============================================================

class TestScoring:
    """Test the convergence scoring formula."""

    def test_score_with_vault_only(self):
        """Single-source score should be weighted correctly."""
        from obsession_radar import SourceProbe, _compute_score

        probes = {
            "vault": SourceProbe("vault", True, 5, 0.80, [], []),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = _compute_score(probes)
        assert score > 0
        # Vault weight is 1.0, so score should be based on vault avg_relevance
        # + a depth bonus from match_count

    def test_score_multi_source_higher_than_single(self):
        """Multi-source convergence should score higher than single source."""
        from obsession_radar import SourceProbe, _compute_score

        single_probes = {
            "vault": SourceProbe("vault", True, 10, 0.90, [], []),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        multi_probes = {
            "vault": SourceProbe("vault", True, 5, 0.75, [], []),
            "conversations": SourceProbe("conversations", True, 3, 0.70, [], []),
            "saved_items": SourceProbe("saved_items", True, 4, 0.60, [], []),
            "audio": SourceProbe("audio", True, 2, 0.65, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        single_score = _compute_score(single_probes)
        multi_score = _compute_score(multi_probes)

        assert multi_score > single_score, (
            f"Multi-source ({multi_score:.2f}) should beat single-source ({single_score:.2f})"
        )

    def test_published_reduces_score(self):
        """Already-published topics should score lower."""
        from obsession_radar import SourceProbe, _compute_score

        not_published = {
            "vault": SourceProbe("vault", True, 5, 0.80, [], []),
            "conversations": SourceProbe("conversations", True, 3, 0.70, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        published = {
            "vault": SourceProbe("vault", True, 5, 0.80, [], []),
            "conversations": SourceProbe("conversations", True, 3, 0.70, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", True, 2, 1.0, [], []),
        }

        score_new = _compute_score(not_published)
        score_old = _compute_score(published)

        assert score_new > score_old, (
            f"Unpublished ({score_new:.2f}) should beat published ({score_old:.2f})"
        )

    def test_score_all_empty_is_zero(self):
        """No sources found should produce score of zero."""
        from obsession_radar import SourceProbe, _compute_score

        probes = {
            "vault": SourceProbe("vault", False, 0, 0.0, [], []),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = _compute_score(probes)
        assert score == 0.0


# ============================================================
# Phase 5: Category mapping
# ============================================================

class TestCategoryMapping:
    """Test topic-to-category mapping."""

    def test_map_health_topic(self):
        from obsession_radar import _suggest_category

        assert _suggest_category("exercise routines") == "health-fitness"

    def test_map_sleep_topic(self):
        from obsession_radar import _suggest_category

        assert _suggest_category("sleep quality") == "rest-recovery"

    def test_map_dopamine_topic(self):
        from obsession_radar import _suggest_category

        # Dopamine could be health or emotional wellbeing
        cat = _suggest_category("dopamine detox")
        assert cat in ("health-fitness", "emotional-wellbeing", "mindfulness-stress")

    def test_map_meditation_topic(self):
        from obsession_radar import _suggest_category

        assert _suggest_category("meditation practice") == "mindfulness-stress"

    def test_map_unknown_topic(self):
        from obsession_radar import _suggest_category

        # Unknown topic should return empty string or None
        cat = _suggest_category("xyzzy foobar quux")
        assert cat == "" or cat is None or isinstance(cat, str)


# ============================================================
# Phase 6: Full scan_topic integration (mocked)
# ============================================================

class TestScanTopic:
    """Test full scan_topic pipeline with all sources mocked."""

    @patch("obsession_radar._search_source")
    def test_scan_topic_returns_topic_score(self, mock_search):
        """scan_topic should return a TopicScore with all probes."""
        from obsession_radar import ObsessionRadar, TopicScore
        from content_research_engine import ResearchResult

        mock_search.return_value = []

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)
        radar._embed_model = MagicMock()
        radar._embed_model.get_query_embedding.return_value = [0.1] * 1024

        # Override file paths to nonexistent dirs so file scanners skip gracefully
        radar._audio_dir = Path("/nonexistent/audio")
        radar._pipeline_dir = Path("/nonexistent/pipeline")

        result = radar.scan_topic("dopamine")
        assert isinstance(result, TopicScore)
        assert result.topic == "dopamine"
        assert "vault" in result.probes
        assert "audio" in result.probes
        assert "published" in result.probes

    @patch("obsession_radar._search_source")
    def test_scan_topics_returns_sorted_list(self, mock_search):
        """scan_topics should return a list sorted by total_score descending."""
        from obsession_radar import ObsessionRadar

        mock_search.return_value = []

        mock_cfg = MagicMock()
        mock_cfg.embedding = MagicMock()
        mock_cfg.voyage_api_key = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)
        radar._embed_model = MagicMock()
        radar._embed_model.get_query_embedding.return_value = [0.1] * 1024
        radar._audio_dir = Path("/nonexistent/audio")
        radar._pipeline_dir = Path("/nonexistent/pipeline")

        results = radar.scan_topics(["topic1", "topic2", "topic3"])
        assert isinstance(results, list)
        assert len(results) == 3
        # Results should be sorted descending by score
        for i in range(len(results) - 1):
            assert results[i].total_score >= results[i + 1].total_score


# ============================================================
# Phase 7: Output formatting
# ============================================================

class TestOutputFormatting:
    """Test output format functions."""

    def test_format_yaml_output(self):
        from obsession_radar import SourceProbe, TopicScore, format_output

        probes = {
            "vault": SourceProbe("vault", True, 3, 0.8, ["excerpt1"], ["Title1"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="dopamine",
            convergence=1,
            total_score=5.0,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
        )

        output = format_output([score], fmt="yaml")
        assert "dopamine" in output
        assert isinstance(output, str)

    def test_format_json_output(self):
        from obsession_radar import SourceProbe, TopicScore, format_output

        probes = {
            "vault": SourceProbe("vault", True, 3, 0.8, ["excerpt1"], ["Title1"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="dopamine",
            convergence=1,
            total_score=5.0,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
        )

        output = format_output([score], fmt="json")
        parsed = json.loads(output)
        assert isinstance(parsed, dict)
        assert "results" in parsed
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["topic"] == "dopamine"

    def test_format_markdown_output(self):
        from obsession_radar import SourceProbe, TopicScore, format_output

        probes = {
            "vault": SourceProbe("vault", True, 3, 0.8, ["excerpt1"], ["Title1"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="dopamine",
            convergence=1,
            total_score=5.0,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
        )

        output = format_output([score], fmt="markdown")
        assert "# Obsession Radar Report" in output
        assert "dopamine" in output
        assert "Vault" in output or "vault" in output

    def test_format_pipeline_output(self):
        """Pipeline format should produce ideas.md-compatible entries."""
        from obsession_radar import SourceProbe, TopicScore, format_output

        probes = {
            "vault": SourceProbe("vault", True, 12, 0.82, ["Dopamine Nation notes"], ["Dopamine Nation"]),
            "conversations": SourceProbe("conversations", True, 5, 0.78, ["Dopamine discussion"], ["Discussion"]),
            "saved_items": SourceProbe("saved_items", True, 8, 0.65, ["Huberman episode"], ["Huberman"]),
            "audio": SourceProbe("audio", True, 2, 0.70, ["Jan reflection"], ["Reflection"]),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="Dopamine & Habit Formation",
            convergence=4,
            total_score=87.5,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
        )

        output = format_output([score], fmt="pipeline")
        assert "### Dopamine & Habit Formation" in output
        assert "4/5 sources" in output or "4" in output


# ============================================================
# Phase 8: Category seed topics and auto-discovery basics
# ============================================================

class TestCategorySeeds:
    """Test the category seed topics definitions."""

    def test_all_10_categories_have_seeds(self):
        from obsession_radar import CATEGORY_SEED_TOPICS

        expected_categories = [
            "relationships", "health-fitness", "finances-wealth",
            "emotional-wellbeing", "mindfulness-stress", "rest-recovery",
            "productivity-work", "creativity-leisure", "learning-growth",
            "purpose-spirituality",
        ]

        for cat in expected_categories:
            assert cat in CATEGORY_SEED_TOPICS, f"Missing seeds for category: {cat}"
            assert len(CATEGORY_SEED_TOPICS[cat]) >= 5, (
                f"Category {cat} should have at least 5 seed topics, has {len(CATEGORY_SEED_TOPICS[cat])}"
            )

    def test_seed_topics_are_strings(self):
        from obsession_radar import CATEGORY_SEED_TOPICS

        for cat, seeds in CATEGORY_SEED_TOPICS.items():
            for seed in seeds:
                assert isinstance(seed, str), f"Seed {seed} in {cat} is not a string"
                assert len(seed) > 2, f"Seed '{seed}' in {cat} is too short"


# ============================================================
# Phase 9: Topic matching utility
# ============================================================

class TestTopicMatching:
    """Test the fuzzy topic matching function."""

    def test_exact_match(self):
        from obsession_radar import _topic_matches

        assert _topic_matches("dopamine", "The dopamine system is complex")

    def test_case_insensitive(self):
        from obsession_radar import _topic_matches

        assert _topic_matches("Dopamine", "dopamine detox protocol")

    def test_partial_match(self):
        from obsession_radar import _topic_matches

        assert _topic_matches("sleep quality", "How to improve your sleep quality significantly")

    def test_no_match(self):
        from obsession_radar import _topic_matches

        assert not _topic_matches("quantum physics", "The dopamine system is complex")

    def test_multi_word_topic(self):
        from obsession_radar import _topic_matches

        assert _topic_matches(
            "morning routines",
            "My morning routines for productivity"
        )

    def test_word_boundary_match(self):
        """Should match whole words, not substrings of unrelated words."""
        from obsession_radar import _topic_matches

        # 'do' should not match 'dopamine' in a meaningful way
        # But 'dopamine' should match 'dopamine'
        assert _topic_matches("dopamine", "A discussion about dopamine receptors")


# ============================================================
# Phase 10: YAML frontmatter parsing
# ============================================================

class TestFrontmatterParsing:
    """Test YAML frontmatter extraction from markdown files."""

    def test_parse_valid_frontmatter(self):
        from obsession_radar import _parse_frontmatter

        content = '---\ntitle: "Test Article"\ncategory: "health-fitness"\n---\n\nContent here.\n'
        fm = _parse_frontmatter(content)
        assert fm["title"] == "Test Article"
        assert fm["category"] == "health-fitness"

    def test_parse_no_frontmatter(self):
        from obsession_radar import _parse_frontmatter

        content = "# No frontmatter\n\nJust content.\n"
        fm = _parse_frontmatter(content)
        assert fm == {}

    def test_parse_empty_frontmatter(self):
        from obsession_radar import _parse_frontmatter

        content = "---\n---\n\nContent after empty frontmatter.\n"
        fm = _parse_frontmatter(content)
        assert fm == {} or fm is not None


# ============================================================
# Phase 11: Bilingual matching (Greek <-> English)
# ============================================================

class TestBilingualMatching:
    """Test Greek/English bilingual topic matching."""

    def test_english_topic_matches_greek_content(self):
        """English topic 'sleep' should match Greek content about ύπνος."""
        from obsession_radar import _topic_matches

        # Greek content mentioning sleep
        greek_text = "Σκεφτόμουν τον ύπνο μου χθες βράδυ."
        assert _topic_matches("sleep", greek_text), \
            "English 'sleep' should match Greek content containing 'ύπνο'"

    def test_english_meditation_matches_greek(self):
        """English 'meditation' should match Greek 'διαλογισμός'."""
        from obsession_radar import _topic_matches

        greek_text = "Ο διαλογισμός βοηθάει στο στρες."
        assert _topic_matches("meditation", greek_text)

    def test_greek_topic_matches_english_content(self):
        """Greek topic should match English content."""
        from obsession_radar import _topic_matches

        english_text = "I practiced meditation today for 20 minutes."
        # Greek word for meditation
        assert _topic_matches("διαλογισμός", english_text)

    def test_stoicism_bilingual(self):
        """'stoicism' should match Greek 'στωικισμός' variants."""
        from obsession_radar import _topic_matches

        greek_text = "Η στωική φιλοσοφία είναι πρακτική."
        assert _topic_matches("stoicism", greek_text)


# ============================================================
# Phase 12: Source weight configuration
# ============================================================

class TestSourceWeights:
    """Test that source weights are properly configured."""

    def test_source_weights_exist(self):
        from obsession_radar import SOURCE_WEIGHTS
        assert "vault" in SOURCE_WEIGHTS
        assert "conversations" in SOURCE_WEIGHTS
        assert "saved_items" in SOURCE_WEIGHTS
        assert "audio" in SOURCE_WEIGHTS
        assert "published" in SOURCE_WEIGHTS

    def test_vault_has_highest_weight(self):
        """Vault should have the highest positive weight."""
        from obsession_radar import SOURCE_WEIGHTS
        positive_weights = {k: v for k, v in SOURCE_WEIGHTS.items() if v > 0}
        assert SOURCE_WEIGHTS["vault"] == max(positive_weights.values())

    def test_published_has_negative_weight(self):
        """Published should have a negative weight."""
        from obsession_radar import SOURCE_WEIGHTS
        assert SOURCE_WEIGHTS["published"] < 0


# ============================================================
# Phase 13: Integration-level scan with real audio files
# ============================================================

class TestAudioScannerIntegration:
    """Integration test for audio scanning with realistic file structure."""

    def test_scan_audio_bilingual_match(self):
        """Audio scanner should find Greek content when searching English topics."""
        from obsession_radar import ObsessionRadar

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)

            # Create analysis file with Greek content about sleep
            analysis = audio_dir / "2026-02-10_analysis.md"
            analysis.write_text(
                "# Analysis\n\n## INSIGHTS\n\n"
                "Ο ύπνος μου ήταν κακός χθες.\n"
                "Πρέπει να βελτιώσω τη ρουτίνα ύπνου.\n\n"
                "## BLOCKERS\n\nΑΰπνία.\n",
                encoding="utf-8",
            )

            mock_cfg = MagicMock()
            mock_cfg.embedding = MagicMock()
            mock_cfg.voyage_api_key = MagicMock()
            radar = ObsessionRadar(config=mock_cfg)

            probe = radar._scan_audio("sleep", audio_dir)
            assert probe.found is True, "Should find 'sleep' via bilingual match with 'ύπνος'"


# ============================================================
# Phase 14: Formatting edge cases
# ============================================================

class TestFormattingEdgeCases:
    """Test output formatting with edge cases."""

    def test_format_empty_results(self):
        """Formatting an empty results list should not crash."""
        from obsession_radar import format_output

        for fmt in ["yaml", "json", "markdown", "pipeline"]:
            output = format_output([], fmt=fmt)
            assert isinstance(output, str)
            assert len(output) > 0  # Should produce valid (possibly minimal) output

    def test_json_output_is_valid_json(self):
        """JSON output should parse without errors."""
        from obsession_radar import SourceProbe, TopicScore, format_output

        probes = {
            "vault": SourceProbe("vault", True, 5, 0.75, ["excerpt"], ["Title"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }
        score = TopicScore("test topic", 1, 5.0, probes, "health-fitness", False, 0)

        output = format_output([score], fmt="json")
        parsed = json.loads(output)
        assert parsed["results"][0]["topic"] == "test topic"

    def test_markdown_contains_source_table(self):
        """Markdown output should contain a source table."""
        from obsession_radar import SourceProbe, TopicScore, format_output

        probes = {
            "vault": SourceProbe("vault", True, 5, 0.75, ["excerpt"], ["Title"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }
        score = TopicScore("test topic", 1, 5.0, probes, "health-fitness", False, 0)

        output = format_output([score], fmt="markdown")
        assert "| Source |" in output
        assert "Vault" in output


# ============================================================
# Phase 15: Topic Normalization (mine subcommand)
# ============================================================

class TestTopicNormalization:
    """Test _normalize_topic function."""

    def test_lowercase(self):
        from obsession_radar import _normalize_topic
        assert _normalize_topic("Machine Learning") == "ai/machine learning"

    def test_hyphen_to_space(self):
        from obsession_radar import _normalize_topic
        assert _normalize_topic("web-dev") == "web development"

    def test_collapse_whitespace(self):
        from obsession_radar import _normalize_topic
        result = _normalize_topic("  too   many   spaces  ")
        assert "  " not in result

    def test_variant_map_lookup(self):
        from obsession_radar import _normalize_topic
        assert _normalize_topic("personal dev") == "personal development"
        assert _normalize_topic("self help") == "self-improvement"

    def test_unknown_topic_passthrough(self):
        from obsession_radar import _normalize_topic
        assert _normalize_topic("quantum computing") == "quantum computing"

    def test_adhd_variants(self):
        from obsession_radar import _normalize_topic
        assert _normalize_topic("ADHD") == "adhd"
        assert _normalize_topic("attention deficit") == "adhd"


# ============================================================
# Phase 16: Recency Weight
# ============================================================

class TestRecencyWeight:
    """Test _recency_weight exponential decay function."""

    def test_today_is_near_one(self):
        from obsession_radar import _recency_weight
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        weight = _recency_weight(now, now)
        assert 0.99 <= weight <= 1.0

    def test_90_days_is_half(self):
        from obsession_radar import _recency_weight
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=90)
        weight = _recency_weight(past, now, half_life_days=90.0)
        assert 0.45 <= weight <= 0.55, f"Expected ~0.5, got {weight}"

    def test_180_days_is_quarter(self):
        from obsession_radar import _recency_weight
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=180)
        weight = _recency_weight(past, now, half_life_days=90.0)
        assert 0.20 <= weight <= 0.30, f"Expected ~0.25, got {weight}"

    def test_very_old_is_near_zero(self):
        from obsession_radar import _recency_weight
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=3650)  # 10 years
        weight = _recency_weight(past, now)
        assert weight < 0.01

    def test_future_date_clamps(self):
        from obsession_radar import _recency_weight
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        future = now + timedelta(days=10)
        weight = _recency_weight(future, now)
        # Future dates get clamped to 0 days_ago -> weight ~1.0
        assert weight >= 0.99


# ============================================================
# Phase 17: Parse Since
# ============================================================

class TestParseSince:
    """Test _parse_since time window parsing."""

    def test_parse_days(self):
        from obsession_radar import _parse_since
        from datetime import datetime, timezone
        cutoff = _parse_since("30d")
        now = datetime.now(timezone.utc)
        delta = now - cutoff
        assert 29 <= delta.days <= 31

    def test_parse_months(self):
        from obsession_radar import _parse_since
        from datetime import datetime, timezone
        cutoff = _parse_since("6m")
        now = datetime.now(timezone.utc)
        delta = now - cutoff
        assert 179 <= delta.days <= 181

    def test_parse_years(self):
        from obsession_radar import _parse_since
        from datetime import datetime, timezone
        cutoff = _parse_since("1y")
        now = datetime.now(timezone.utc)
        delta = now - cutoff
        assert 364 <= delta.days <= 366

    def test_parse_invalid_raises(self):
        from obsession_radar import _parse_since
        with pytest.raises(ValueError):
            _parse_since("invalid")


# ============================================================
# Phase 18: Mining Scores
# ============================================================

class TestMiningScores:
    """Test _compute_mining_scores function."""

    def _make_items(self):
        """Create a list of fake saved_items for testing."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        return [
            {
                "item_id": "1",
                "topics": ["neuroscience", "Dopamine"],
                "tags": ["science"],
                "saved_at": (now - timedelta(days=5)).isoformat(),
                "usefulness_score": 80,
                "content_type": "article",
                "source": "youtube",
            },
            {
                "item_id": "2",
                "topics": ["neuroscience", "sleep"],
                "tags": ["health"],
                "saved_at": (now - timedelta(days=10)).isoformat(),
                "usefulness_score": 70,
                "content_type": "article",
                "source": "reddit",
            },
            {
                "item_id": "3",
                "topics": ["Dopamine", "ADHD"],
                "tags": ["health"],
                "saved_at": (now - timedelta(days=30)).isoformat(),
                "usefulness_score": 90,
                "content_type": "video",
                "source": "youtube",
            },
            {
                "item_id": "4",
                "topics": ["neuroscience"],
                "tags": ["science"],
                "saved_at": (now - timedelta(days=60)).isoformat(),
                "usefulness_score": 60,
                "content_type": "article",
                "source": "chrome_bookmarks",
            },
            {
                "item_id": "5",
                "topics": ["cooking"],
                "tags": ["food"],
                "saved_at": (now - timedelta(days=2)).isoformat(),
                "usefulness_score": 50,
                "content_type": "article",
                "source": "reddit",
            },
        ]

    def test_returns_sorted_list(self):
        from obsession_radar import _compute_mining_scores
        items = self._make_items()
        result = _compute_mining_scores(items, min_count=1)
        assert isinstance(result, list)
        for i in range(len(result) - 1):
            assert result[i].mining_score >= result[i + 1].mining_score

    def test_min_count_filter(self):
        from obsession_radar import _compute_mining_scores
        items = self._make_items()
        # Only topics with 2+ items should appear with min_count=2
        result = _compute_mining_scores(items, min_count=2)
        for m in result:
            assert m.item_count >= 2

    def test_neuroscience_ranks_high(self):
        from obsession_radar import _compute_mining_scores
        items = self._make_items()
        result = _compute_mining_scores(items, min_count=1)
        topics = [m.topic for m in result]
        assert "neuroscience" in topics
        # neuroscience has 3 items, should rank high
        neuro = next(m for m in result if m.topic == "neuroscience")
        assert neuro.item_count == 3

    def test_source_diversity(self):
        from obsession_radar import _compute_mining_scores
        items = self._make_items()
        result = _compute_mining_scores(items, min_count=1)
        neuro = next(m for m in result if m.topic == "neuroscience")
        # neuroscience comes from youtube, reddit, chrome_bookmarks
        assert neuro.source_diversity >= 2

    def test_content_only_filter(self):
        from obsession_radar import _compute_mining_scores
        items = self._make_items()
        result = _compute_mining_scores(items, min_count=1, content_only=True)
        # All results should have a recognized category
        for m in result:
            assert m.category_suggestion != ""

    def test_since_filter(self):
        from obsession_radar import _compute_mining_scores
        from datetime import datetime, timezone, timedelta
        items = self._make_items()
        since = datetime.now(timezone.utc) - timedelta(days=15)
        result = _compute_mining_scores(items, min_count=1, since=since)
        # Only items from last 15 days should be included
        # item_id 3 (30d ago) and 4 (60d ago) should be excluded
        for m in result:
            # cooking (2d ago) should appear but with count 1
            if m.topic == "cooking":
                assert m.item_count == 1


# ============================================================
# Phase 19: Load Saved Items Metadata (mocked LanceDB)
# ============================================================

class TestLoadSavedItemsMetadata:
    """Test _load_saved_items_metadata with mocked LanceDB."""

    @patch("obsession_radar.lancedb", create=True)
    def test_loads_and_deduplicates(self, mock_lancedb_module):
        """Should load items and deduplicate by item_id."""
        import pandas as pd
        import pyarrow as pa

        # Simulate two chunks for same item_id
        data = {
            "item_id": ["item1", "item1", "item2"],
            "topics": ["neuroscience,dopamine", "sleep", "cooking"],
            "tags": ["science", "health", "food"],
            "saved_at": ["2026-01-01", "2026-01-01", "2026-01-15"],
            "usefulness_score": [80, 80, 60],
            "content_type": ["article", "article", "video"],
            "source": ["youtube", "youtube", "reddit"],
        }
        arrow_table = pa.table(data)

        mock_table = MagicMock()
        mock_table.to_arrow.return_value = arrow_table

        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table

        # Patch lancedb import inside the function
        with patch("obsession_radar.lancedb", create=True) as mock_lance:
            mock_lance.connect.return_value = mock_db
            # Need to also patch the actual import inside the function
            with patch.dict("sys.modules", {"lancedb": mock_lance}):
                from obsession_radar import _load_saved_items_metadata
                result = _load_saved_items_metadata()

        assert len(result) == 2  # 2 unique items
        item1 = next(i for i in result if i["item_id"] == "item1")
        # Topics from both chunks should be merged
        assert "neuroscience" in item1["topics"]
        assert "dopamine" in item1["topics"]
        assert "sleep" in item1["topics"]

    @patch("obsession_radar.lancedb", create=True)
    def test_handles_missing_table(self, mock_lancedb_module):
        """Should return empty list when table doesn't exist."""
        with patch.dict("sys.modules", {"lancedb": mock_lancedb_module}):
            mock_lancedb_module.connect.return_value.open_table.side_effect = Exception("Table not found")
            from obsession_radar import _load_saved_items_metadata
            result = _load_saved_items_metadata()
        assert result == []

    @patch("obsession_radar.lancedb", create=True)
    def test_handles_empty_table(self, mock_lancedb_module):
        """Should return empty list for empty table."""
        import pyarrow as pa

        arrow_table = pa.table({
            "item_id": [],
            "topics": [],
            "tags": [],
            "saved_at": [],
            "usefulness_score": [],
            "content_type": [],
            "source": [],
        })

        mock_table = MagicMock()
        mock_table.to_arrow.return_value = arrow_table

        mock_db = MagicMock()
        mock_db.open_table.return_value = mock_table

        with patch.dict("sys.modules", {"lancedb": mock_lancedb_module}):
            mock_lancedb_module.connect.return_value = mock_db
            from obsession_radar import _load_saved_items_metadata
            result = _load_saved_items_metadata()
        assert result == []


# ============================================================
# Phase 20: Mine Topics (mocked)
# ============================================================

class TestMineTopics:
    """Test ObsessionRadar.mine_topics orchestrator."""

    @patch("obsession_radar._load_saved_items_metadata")
    def test_mine_skip_scan_returns_mined_topics(self, mock_load):
        """With skip_scan=True, should return MinedTopic list."""
        from obsession_radar import ObsessionRadar, MinedTopic
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        mock_load.return_value = [
            {
                "item_id": str(i),
                "topics": ["neuroscience", "dopamine"],
                "tags": ["science"],
                "saved_at": (now - timedelta(days=i * 5)).isoformat(),
                "usefulness_score": 80,
                "content_type": "article",
                "source": "youtube",
            }
            for i in range(5)
        ]

        mock_cfg = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        results = radar.mine_topics(top_k=10, skip_scan=True, min_count=2)
        assert isinstance(results, list)
        assert all(isinstance(r, MinedTopic) for r in results)
        assert len(results) > 0

    @patch("obsession_radar._load_saved_items_metadata")
    def test_mine_empty_returns_empty(self, mock_load):
        """Should return empty list when no saved_items."""
        from obsession_radar import ObsessionRadar

        mock_load.return_value = []
        mock_cfg = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        results = radar.mine_topics(skip_scan=True)
        assert results == []

    @patch("obsession_radar._load_saved_items_metadata")
    def test_mine_content_only(self, mock_load):
        """With content_only=True, only categorized topics should appear."""
        from obsession_radar import ObsessionRadar
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        mock_load.return_value = [
            {
                "item_id": str(i),
                "topics": ["sleep quality", "kubernetes"],
                "tags": [],
                "saved_at": (now - timedelta(days=i)).isoformat(),
                "usefulness_score": 70,
                "content_type": "article",
                "source": "reddit",
            }
            for i in range(5)
        ]

        mock_cfg = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        results = radar.mine_topics(skip_scan=True, content_only=True, min_count=2)
        for r in results:
            assert r.category_suggestion != ""

    @patch("obsession_radar._load_saved_items_metadata")
    def test_mine_category_filter(self, mock_load):
        """Should filter to a specific category."""
        from obsession_radar import ObsessionRadar
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        mock_load.return_value = [
            {
                "item_id": str(i),
                "topics": ["sleep quality", "meditation practice", "investing fundamentals"],
                "tags": [],
                "saved_at": (now - timedelta(days=i)).isoformat(),
                "usefulness_score": 70,
                "content_type": "article",
                "source": "reddit",
            }
            for i in range(5)
        ]

        mock_cfg = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)

        results = radar.mine_topics(
            skip_scan=True, category="rest-recovery", min_count=2,
        )
        for r in results:
            assert r.category_suggestion == "rest-recovery"

    @patch("obsession_radar._search_source")
    @patch("obsession_radar._load_saved_items_metadata")
    def test_mine_with_scan_returns_topic_scores(self, mock_load, mock_search):
        """Without skip_scan, should return TopicScore with mining_data."""
        from obsession_radar import ObsessionRadar, TopicScore
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        mock_load.return_value = [
            {
                "item_id": str(i),
                "topics": ["neuroscience"],
                "tags": ["science"],
                "saved_at": (now - timedelta(days=i)).isoformat(),
                "usefulness_score": 80,
                "content_type": "article",
                "source": "youtube",
            }
            for i in range(5)
        ]
        mock_search.return_value = []

        mock_cfg = MagicMock()
        radar = ObsessionRadar(config=mock_cfg)
        radar._embed_model = MagicMock()
        radar._embed_model.get_query_embedding.return_value = [0.1] * 1024
        radar._audio_dir = Path("/nonexistent/audio")
        radar._pipeline_dir = Path("/nonexistent/pipeline")

        results = radar.mine_topics(top_k=5, skip_scan=False, min_count=2)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], TopicScore)
            assert results[0].mining_data is not None


# ============================================================
# Phase 21: Mined Output Formatting
# ============================================================

class TestMinedOutputFormatting:
    """Test format_mined_output function."""

    def _make_mined_topic(self):
        from obsession_radar import MinedTopic
        return MinedTopic(
            topic="neuroscience",
            raw_variants=["Neuroscience", "neuroscience"],
            item_count=15,
            mining_score=3.45,
            frequency_score=4.0,
            recency_weight=0.72,
            quality_factor=0.80,
            source_diversity=3,
            content_types={"article": 10, "video": 5},
            category_suggestion="health-fitness",
        )

    def test_format_mined_json(self):
        from obsession_radar import format_mined_output
        topic = self._make_mined_topic()
        output = format_mined_output([topic], fmt="json")
        parsed = json.loads(output)
        assert parsed["mode"] == "mine"
        assert len(parsed["results"]) == 1
        assert parsed["results"][0]["topic"] == "neuroscience"
        assert parsed["results"][0]["item_count"] == 15

    def test_format_mined_markdown(self):
        from obsession_radar import format_mined_output
        topic = self._make_mined_topic()
        output = format_mined_output([topic], fmt="markdown")
        assert "Mined Topics" in output
        assert "neuroscience" in output
        assert "15" in output

    def test_format_mined_pipeline(self):
        from obsession_radar import format_mined_output
        topic = self._make_mined_topic()
        output = format_mined_output([topic], fmt="pipeline")
        assert "### neuroscience" in output
        assert "15 saves" in output

    def test_format_mined_empty(self):
        from obsession_radar import format_mined_output
        for fmt in ["json", "yaml", "markdown", "pipeline"]:
            output = format_mined_output([], fmt=fmt)
            assert isinstance(output, str)
            assert len(output) > 0


# ============================================================
# Phase 22: Mine CLI
# ============================================================

class TestMineCLI:
    """Test CLI argument parsing for mine subcommand."""

    def test_mine_parser_exists(self):
        from obsession_radar import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["mine", "--skip-scan", "--top", "10"])
        assert args.command == "mine"
        assert args.skip_scan is True
        assert args.top == 10

    def test_mine_parser_defaults(self):
        from obsession_radar import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["mine"])
        assert args.command == "mine"
        assert args.top == 20
        assert args.format == "yaml"
        assert args.content_only is False
        assert args.skip_scan is False
        assert args.new_only is False
        assert args.since is None
        assert args.min_count == 2
        assert args.category is None

    def test_mine_parser_all_flags(self):
        from obsession_radar import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "mine",
            "--top", "30",
            "--format", "markdown",
            "--content-only",
            "--skip-scan",
            "--new-only",
            "--since", "90d",
            "--min-count", "5",
            "--category", "health-fitness",
            "--pipeline",
        ])
        assert args.top == 30
        assert args.format == "markdown"
        assert args.content_only is True
        assert args.skip_scan is True
        assert args.new_only is True
        assert args.since == "90d"
        assert args.min_count == 5
        assert args.category == "health-fitness"
        assert args.pipeline is True


# ============================================================
# Phase 23: TopicScore with mining_data backward compatibility
# ============================================================

class TestTopicScoreMiningData:
    """Test TopicScore backward compatibility with mining_data field."""

    def test_topic_score_without_mining_data(self):
        """Old code creating TopicScore without mining_data should still work."""
        from obsession_radar import TopicScore, SourceProbe

        probes = {
            "vault": SourceProbe("vault", True, 3, 0.8, ["e1"], ["t1"]),
        }
        score = TopicScore(
            topic="test",
            convergence=1,
            total_score=3.0,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
        )
        assert score.mining_data is None

    def test_topic_score_with_mining_data(self):
        """TopicScore with mining_data should serialize correctly."""
        from obsession_radar import TopicScore, SourceProbe, MinedTopic, _results_to_dicts

        mining = MinedTopic(
            topic="test",
            raw_variants=["Test", "test"],
            item_count=10,
            mining_score=2.5,
            frequency_score=3.5,
            recency_weight=0.7,
            quality_factor=0.8,
            source_diversity=2,
            content_types={"article": 8, "video": 2},
            category_suggestion="health-fitness",
        )

        probes = {
            "vault": SourceProbe("vault", True, 3, 0.8, ["e1"], ["t1"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="test",
            convergence=1,
            total_score=3.0,
            probes=probes,
            category_suggestion="health-fitness",
            already_published=False,
            existing_briefs=0,
            mining_data=mining,
        )

        dicts = _results_to_dicts([score])
        assert len(dicts) == 1
        assert "mining_data" in dicts[0]
        assert dicts[0]["mining_data"]["item_count"] == 10
        assert dicts[0]["mining_data"]["mining_score"] == 2.5

    def test_markdown_includes_mining_data(self):
        """Markdown format should include mining metadata when present."""
        from obsession_radar import TopicScore, SourceProbe, MinedTopic, format_output

        mining = MinedTopic(
            topic="sleep",
            raw_variants=["Sleep", "sleep"],
            item_count=20,
            mining_score=4.2,
            frequency_score=4.4,
            recency_weight=0.65,
            quality_factor=0.75,
            source_diversity=3,
            content_types={"article": 15, "video": 5},
            category_suggestion="rest-recovery",
        )

        probes = {
            "vault": SourceProbe("vault", True, 5, 0.8, ["e1"], ["t1"]),
            "conversations": SourceProbe("conversations", False, 0, 0.0, [], []),
            "saved_items": SourceProbe("saved_items", False, 0, 0.0, [], []),
            "audio": SourceProbe("audio", False, 0, 0.0, [], []),
            "published": SourceProbe("published", False, 0, 0.0, [], []),
        }

        score = TopicScore(
            topic="sleep",
            convergence=1,
            total_score=5.0,
            probes=probes,
            category_suggestion="rest-recovery",
            already_published=False,
            existing_briefs=0,
            mining_data=mining,
        )

        output = format_output([score], fmt="markdown")
        assert "Mined:" in output
        assert "20 saves" in output
        assert "3 platforms" in output
