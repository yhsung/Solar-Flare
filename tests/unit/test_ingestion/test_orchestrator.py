"""
Unit tests for ingestion orchestrator components.

Note: Full orchestrator tests require the complete memory module.
These tests focus on progress tracking and config components.
"""
import sys
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import directly from ingestion submodule to avoid full memory module import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from solar_flare.memory.ingestion.config import (
    IngestionConfig,
    LocalFSConfig,
    VersioningConfig,
    SourceType,
    SMBConfig,
    BasicAuth,
)
from solar_flare.memory.ingestion.progress import (
    IngestionResult,
    IngestionProgress,
    ProgressTracker,
)


class TestIngestionResult:
    """Tests for IngestionResult."""

    def test_initial_state(self):
        """Test initial result state."""
        result = IngestionResult()

        assert result.total_documents == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.skipped_unchanged == 0
        assert result.success_rate == 100.0

    def test_add_success(self):
        """Test adding successful ingestion."""
        result = IngestionResult()
        result.add_success("source1", "pdf")
        result.add_success("source1", "pdf")
        result.add_success("source2", "docx")

        assert result.successful == 3
        assert result.by_source["source1"] == 2
        assert result.by_source["source2"] == 1
        assert result.by_format["pdf"] == 2
        assert result.by_format["docx"] == 1

    def test_add_failure(self):
        """Test adding failed ingestion."""
        result = IngestionResult()
        result.add_failure("Error 1")
        result.add_failure("Error 2")

        assert result.failed == 2
        assert len(result.errors) == 2
        assert "Error 1" in result.errors

    def test_add_skipped(self):
        """Test adding skipped document."""
        result = IngestionResult()
        result.add_skipped()
        result.add_skipped()

        assert result.skipped_unchanged == 2

    def test_success_rate(self):
        """Test success rate calculation."""
        result = IngestionResult()
        result.add_success("s1", "pdf")
        result.add_success("s1", "pdf")
        result.add_success("s1", "pdf")
        result.add_failure("Error")

        assert result.success_rate == 75.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = IngestionResult()
        result.add_success("source", "pdf")
        result.duration_seconds = 10.5

        d = result.to_dict()

        assert d["successful"] == 1
        assert d["duration_seconds"] == 10.5
        assert "by_source" in d
        assert "by_format" in d


class TestIngestionProgress:
    """Tests for IngestionProgress."""

    def test_percent_complete(self):
        """Test completion percentage."""
        progress = IngestionProgress(
            source_name="test",
            total_documents=100,
            processed=50,
            successful=45,
            failed=5,
        )

        assert progress.percent_complete == 50.0

    def test_percent_complete_zero_total(self):
        """Test completion percentage with zero total."""
        progress = IngestionProgress(
            source_name="test",
            total_documents=0,
            processed=0,
            successful=0,
            failed=0,
        )

        assert progress.percent_complete == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = IngestionProgress(
            source_name="test",
            total_documents=100,
            processed=80,
            successful=72,
            failed=8,
        )

        assert progress.success_rate == 90.0

    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        progress = IngestionProgress(
            source_name="test",
            total_documents=100,
            processed=0,
            successful=0,
            failed=0,
        )

        # Should have some elapsed time (very small)
        assert progress.elapsed_seconds >= 0


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_start_source(self):
        """Test starting a source."""
        tracker = ProgressTracker()
        progress = tracker.start_source("test_source", total_documents=100)

        assert progress.source_name == "test_source"
        assert progress.total_documents == 100
        assert "test_source" in tracker.sources

    def test_update(self):
        """Test updating progress."""
        tracker = ProgressTracker()
        tracker.start_source("test", total_documents=100)

        tracker.update("test", processed=50, successful=45, failed=5)
        progress = tracker.get_progress("test")

        assert progress.processed == 50
        assert progress.successful == 45
        assert progress.failed == 5

    def test_increment(self):
        """Test incrementing progress."""
        tracker = ProgressTracker()
        tracker.start_source("test")

        tracker.increment("test", success=True)
        tracker.increment("test", success=True)
        tracker.increment("test", success=False)

        progress = tracker.get_progress("test")
        assert progress.processed == 3
        assert progress.successful == 2
        assert progress.failed == 1

    def test_callback_called(self):
        """Test callback is called on update."""
        callback = Mock()
        tracker = ProgressTracker(callback=callback)
        tracker.start_source("test")

        tracker.increment("test", success=True)

        callback.assert_called_once()

    def test_total_processed(self):
        """Test total processed across sources."""
        tracker = ProgressTracker()
        tracker.start_source("source1")
        tracker.start_source("source2")

        tracker.update("source1", processed=10)
        tracker.update("source2", processed=20)

        assert tracker.total_processed == 30

    def test_get_summary(self):
        """Test getting summary."""
        tracker = ProgressTracker()
        tracker.start_source("test", total_documents=100)
        tracker.update("test", processed=50, successful=45, failed=5)

        summary = tracker.get_summary()

        assert "sources" in summary
        assert "test" in summary["sources"]
        assert summary["total_processed"] == 50


class TestIngestionConfig:
    """Tests for IngestionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = IngestionConfig()

        assert config.sources == []
        assert config.batch_size == 50
        assert config.parallel_sources == 3

    def test_with_sources(self, tmp_path):
        """Test config with sources."""
        config = IngestionConfig(
            sources=[
                LocalFSConfig(
                    name="test",
                    root_path=tmp_path,
                )
            ]
        )

        assert len(config.sources) == 1
        assert config.sources[0].name == "test"

    def test_unique_source_names(self, tmp_path):
        """Test that source names must be unique."""
        with pytest.raises(ValueError, match="unique"):
            IngestionConfig(
                sources=[
                    LocalFSConfig(name="same", root_path=tmp_path),
                    LocalFSConfig(name="same", root_path=tmp_path),
                ]
            )


class TestLocalFSConfig:
    """Tests for LocalFSConfig."""

    def test_defaults(self, tmp_path):
        """Test default values."""
        config = LocalFSConfig(
            name="test",
            root_path=tmp_path,
        )

        assert config.source_type == SourceType.LOCAL_FS
        assert config.recursive is True
        assert config.follow_symlinks is False
        assert config.target_store == "working"

    def test_custom_extensions(self, tmp_path):
        """Test custom file extensions."""
        config = LocalFSConfig(
            name="test",
            root_path=tmp_path,
            file_extensions=[".pdf", ".docx"],
        )

        assert ".pdf" in config.file_extensions
        assert ".docx" in config.file_extensions


class TestVersioningConfig:
    """Tests for VersioningConfig."""

    def test_defaults(self):
        """Test default values."""
        config = VersioningConfig()

        assert config.enabled is True
        assert config.keep_all_versions is True
        assert config.db_path == Path("./data/ingestion/versions.db")

    def test_custom_db_path(self, tmp_path):
        """Test custom database path."""
        config = VersioningConfig(
            db_path=tmp_path / "custom.db"
        )

        assert config.db_path == tmp_path / "custom.db"
