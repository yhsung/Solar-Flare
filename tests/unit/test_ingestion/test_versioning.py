"""
Unit tests for version tracking.
"""
import sys
import pytest
from datetime import datetime, timedelta
from pathlib import Path

# Import directly from ingestion submodule to avoid full memory module import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from solar_flare.memory.ingestion.versioning import VersionStore, DocumentVersion


class TestVersionStore:
    """Tests for VersionStore."""

    @pytest.fixture
    def version_store(self, tmp_path):
        """Create a VersionStore instance."""
        db_path = tmp_path / "versions.db"
        return VersionStore(db_path)

    def test_init_creates_database(self, tmp_path):
        """Test database creation on init."""
        db_path = tmp_path / "test.db"
        store = VersionStore(db_path)

        assert db_path.exists()

    def test_init_creates_parent_dirs(self, tmp_path):
        """Test parent directory creation."""
        db_path = tmp_path / "subdir" / "nested" / "versions.db"
        store = VersionStore(db_path)

        assert db_path.exists()

    def test_add_first_version(self, version_store):
        """Test adding first version of a document."""
        version = version_store.add_version(
            source_path="/test/doc.pdf",
            content_hash="abc123",
            vector_store="internal",
            metadata={"author": "Test"},
        )

        assert version.version_number == 1
        assert version.source_path == "/test/doc.pdf"
        assert version.content_hash == "abc123"
        assert version.vector_store == "internal"
        assert version.is_current is True
        assert version.metadata["author"] == "Test"
        assert version.id is not None

    def test_add_subsequent_versions(self, version_store):
        """Test adding subsequent versions."""
        v1 = version_store.add_version("/doc.pdf", "hash1", "store1")
        v2 = version_store.add_version("/doc.pdf", "hash2", "store1")
        v3 = version_store.add_version("/doc.pdf", "hash3", "store1")

        assert v1.version_number == 1
        assert v2.version_number == 2
        assert v3.version_number == 3

    def test_only_latest_is_current(self, version_store):
        """Test that only the latest version is marked current."""
        version_store.add_version("/doc.pdf", "hash1", "store1")
        version_store.add_version("/doc.pdf", "hash2", "store1")
        version_store.add_version("/doc.pdf", "hash3", "store1")

        versions = version_store.get_all_versions("/doc.pdf")

        current_versions = [v for v in versions if v.is_current]
        assert len(current_versions) == 1
        assert current_versions[0].version_number == 3

    def test_get_latest_version(self, version_store):
        """Test getting latest version."""
        version_store.add_version("/doc.pdf", "hash1", "store1")
        version_store.add_version("/doc.pdf", "hash2", "store1")

        latest = version_store.get_latest_version("/doc.pdf")

        assert latest is not None
        assert latest.version_number == 2
        assert latest.content_hash == "hash2"

    def test_get_latest_version_not_found(self, version_store):
        """Test getting latest version for non-existent document."""
        latest = version_store.get_latest_version("/nonexistent.pdf")
        assert latest is None

    def test_get_all_versions(self, version_store):
        """Test getting all versions of a document."""
        version_store.add_version("/doc.pdf", "hash1", "store1")
        version_store.add_version("/doc.pdf", "hash2", "store1")
        version_store.add_version("/doc.pdf", "hash3", "store1")

        versions = version_store.get_all_versions("/doc.pdf")

        assert len(versions) == 3
        # Should be sorted by version number descending
        assert versions[0].version_number == 3
        assert versions[1].version_number == 2
        assert versions[2].version_number == 1

    def test_has_changed_new_document(self, version_store):
        """Test has_changed for new document."""
        result = version_store.has_changed("/new.pdf", "newhash")
        assert result is True

    def test_has_changed_same_hash(self, version_store):
        """Test has_changed with same content hash."""
        version_store.add_version("/doc.pdf", "samehash", "store1")

        result = version_store.has_changed("/doc.pdf", "samehash")
        assert result is False

    def test_has_changed_different_hash(self, version_store):
        """Test has_changed with different content hash."""
        version_store.add_version("/doc.pdf", "oldhash", "store1")

        result = version_store.has_changed("/doc.pdf", "newhash")
        assert result is True

    def test_get_documents_by_date_range(self, version_store):
        """Test getting documents by date range."""
        # Add some versions
        version_store.add_version("/doc1.pdf", "hash1", "store1")
        version_store.add_version("/doc2.pdf", "hash2", "store1")

        # Get documents from today
        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow() + timedelta(hours=1)

        docs = version_store.get_documents_by_date_range(start, end)

        assert len(docs) == 2

    def test_get_documents_by_date_range_empty(self, version_store):
        """Test date range with no results."""
        version_store.add_version("/doc.pdf", "hash1", "store1")

        # Query for past date
        start = datetime(2020, 1, 1)
        end = datetime(2020, 12, 31)

        docs = version_store.get_documents_by_date_range(start, end)

        assert len(docs) == 0

    def test_get_documents_by_store(self, version_store):
        """Test getting documents by vector store."""
        version_store.add_version("/doc1.pdf", "hash1", "store_a")
        version_store.add_version("/doc2.pdf", "hash2", "store_b")
        version_store.add_version("/doc3.pdf", "hash3", "store_a")

        docs_a = version_store.get_documents_by_store("store_a")
        docs_b = version_store.get_documents_by_store("store_b")

        assert len(docs_a) == 2
        assert len(docs_b) == 1

    def test_delete_document(self, version_store):
        """Test deleting all versions of a document."""
        version_store.add_version("/doc.pdf", "hash1", "store1")
        version_store.add_version("/doc.pdf", "hash2", "store1")
        version_store.add_version("/doc.pdf", "hash3", "store1")

        deleted = version_store.delete_document("/doc.pdf")

        assert deleted == 3
        assert version_store.get_latest_version("/doc.pdf") is None

    def test_delete_nonexistent_document(self, version_store):
        """Test deleting non-existent document."""
        deleted = version_store.delete_document("/nonexistent.pdf")
        assert deleted == 0

    def test_get_statistics(self, version_store):
        """Test getting statistics."""
        version_store.add_version("/doc1.pdf", "hash1", "store_a")
        version_store.add_version("/doc1.pdf", "hash2", "store_a")
        version_store.add_version("/doc2.pdf", "hash3", "store_b")

        stats = version_store.get_statistics()

        assert stats["total_versions"] == 3
        assert stats["unique_documents"] == 2
        assert stats["current_documents"] == 2
        assert stats["by_store"]["store_a"] == 1
        assert stats["by_store"]["store_b"] == 1

    def test_metadata_persistence(self, version_store):
        """Test metadata is properly stored and retrieved."""
        metadata = {
            "author": "Test Author",
            "department": "Engineering",
            "tags": ["safety", "design"],
        }

        version_store.add_version(
            "/doc.pdf",
            "hash1",
            "store1",
            metadata=metadata,
        )

        latest = version_store.get_latest_version("/doc.pdf")

        assert latest.metadata["author"] == "Test Author"
        assert latest.metadata["tags"] == ["safety", "design"]

    def test_empty_metadata(self, version_store):
        """Test handling of empty/null metadata."""
        version_store.add_version("/doc.pdf", "hash1", "store1", metadata=None)

        latest = version_store.get_latest_version("/doc.pdf")

        assert latest.metadata == {}


class TestDocumentVersion:
    """Tests for DocumentVersion dataclass."""

    def test_creation(self):
        """Test creating a DocumentVersion."""
        version = DocumentVersion(
            id=1,
            source_path="/test/doc.pdf",
            content_hash="abc123",
            version_number=1,
            ingested_at=datetime.utcnow(),
            metadata={"key": "value"},
            vector_store="internal",
            is_current=True,
        )

        assert version.id == 1
        assert version.source_path == "/test/doc.pdf"
        assert version.is_current is True
