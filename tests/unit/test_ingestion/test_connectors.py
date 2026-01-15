"""
Unit tests for source connectors.
"""
import sys
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import directly from ingestion submodule to avoid full memory module import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from solar_flare.memory.ingestion.connectors.base import (
    SourceDocument,
    ConnectorError,
    AuthenticationError,
)
from solar_flare.memory.ingestion.connectors.local_fs import LocalFSConnector
from solar_flare.memory.ingestion.config import LocalFSConfig


class TestSourceDocument:
    """Tests for SourceDocument dataclass."""

    def test_creation(self):
        """Test creating a SourceDocument."""
        doc = SourceDocument(
            source_path="/path/to/doc.pdf",
            filename="doc.pdf",
            modified_date=datetime(2024, 1, 15),
            size_bytes=1024,
            metadata={"key": "value"},
        )

        assert doc.source_path == "/path/to/doc.pdf"
        assert doc.filename == "doc.pdf"
        assert doc.size_bytes == 1024
        assert doc.metadata["key"] == "value"

    def test_default_metadata(self):
        """Test default empty metadata."""
        doc = SourceDocument(
            source_path="/path",
            filename="doc.txt",
            modified_date=datetime.now(),
            size_bytes=0,
        )

        assert doc.metadata == {}


class TestLocalFSConnector:
    """Tests for LocalFSConnector."""

    @pytest.fixture
    def sample_directory(self, tmp_path):
        """Create a sample directory structure."""
        # Create files
        (tmp_path / "doc1.md").write_text("# Document 1")
        (tmp_path / "doc2.txt").write_text("Plain text")
        (tmp_path / "data.pdf").write_bytes(b"%PDF-1.4 fake")

        # Create subdirectory
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("# Nested")

        # Create excluded file
        (tmp_path / "excluded.tmp").write_text("temp")

        return tmp_path

    @pytest.fixture
    def connector_config(self, sample_directory):
        """Create connector config."""
        return LocalFSConfig(
            name="test_local",
            root_path=sample_directory,
            recursive=True,
            file_extensions=[".md", ".txt", ".pdf"],
            exclude_patterns=["*.tmp"],
        )

    @pytest.mark.asyncio
    async def test_connect_success(self, connector_config):
        """Test successful connection."""
        connector = LocalFSConnector(connector_config)
        await connector.connect()

        assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_nonexistent_path(self, tmp_path):
        """Test connection to non-existent path."""
        config = LocalFSConfig(
            name="test",
            root_path=tmp_path / "nonexistent",
        )
        connector = LocalFSConnector(config)

        with pytest.raises(ConnectorError, match="does not exist"):
            await connector.connect()

    @pytest.mark.asyncio
    async def test_connect_file_not_directory(self, tmp_path):
        """Test connection to file instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        config = LocalFSConfig(
            name="test",
            root_path=file_path,
        )
        connector = LocalFSConnector(config)

        with pytest.raises(ConnectorError, match="not a directory"):
            await connector.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, connector_config):
        """Test disconnection."""
        connector = LocalFSConnector(connector_config)
        await connector.connect()
        await connector.disconnect()

        assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_list_documents(self, connector_config, sample_directory):
        """Test listing documents."""
        connector = LocalFSConnector(connector_config)
        await connector.connect()

        docs = []
        async for doc in connector.list_documents():
            docs.append(doc)

        filenames = [d.filename for d in docs]

        assert "doc1.md" in filenames
        assert "doc2.txt" in filenames
        assert "data.pdf" in filenames
        assert "nested.md" in filenames
        assert "excluded.tmp" not in filenames

    @pytest.mark.asyncio
    async def test_list_documents_non_recursive(self, sample_directory):
        """Test listing documents without recursion."""
        config = LocalFSConfig(
            name="test",
            root_path=sample_directory,
            recursive=False,
            file_extensions=[".md", ".txt", ".pdf"],
        )
        connector = LocalFSConnector(config)
        await connector.connect()

        docs = []
        async for doc in connector.list_documents():
            docs.append(doc)

        filenames = [d.filename for d in docs]

        # Should not include nested file
        assert "nested.md" not in filenames
        assert "doc1.md" in filenames

    @pytest.mark.asyncio
    async def test_list_documents_not_connected(self, connector_config):
        """Test listing documents when not connected."""
        connector = LocalFSConnector(connector_config)

        with pytest.raises(ConnectorError, match="Not connected"):
            async for _ in connector.list_documents():
                pass

    @pytest.mark.asyncio
    async def test_get_document_stream(self, connector_config, sample_directory):
        """Test getting document stream."""
        connector = LocalFSConnector(connector_config)
        await connector.connect()

        stream = await connector.get_document_stream(
            str(sample_directory / "doc1.md")
        )

        content = stream.read().decode()
        assert "Document 1" in content

    @pytest.mark.asyncio
    async def test_get_document_stream_not_found(self, connector_config):
        """Test getting stream for non-existent file."""
        connector = LocalFSConnector(connector_config)
        await connector.connect()

        with pytest.raises(ConnectorError, match="not found"):
            await connector.get_document_stream("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_context_manager(self, connector_config):
        """Test async context manager."""
        async with LocalFSConnector(connector_config) as connector:
            assert connector.is_connected is True

        assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_exclusion_patterns(self, sample_directory):
        """Test file exclusion patterns."""
        # Create more files
        (sample_directory / "test.log").write_text("log")
        (sample_directory / "build" ).mkdir()
        (sample_directory / "build" / "output.md").write_text("build")

        config = LocalFSConfig(
            name="test",
            root_path=sample_directory,
            recursive=True,
            file_extensions=[".md", ".txt", ".log"],
            exclude_patterns=["*.log", "build/*"],
        )
        connector = LocalFSConnector(config)
        await connector.connect()

        docs = []
        async for doc in connector.list_documents():
            docs.append(doc)

        filenames = [d.filename for d in docs]

        assert "test.log" not in filenames
        assert "output.md" not in filenames

    @pytest.mark.asyncio
    async def test_document_metadata(self, connector_config, sample_directory):
        """Test document metadata."""
        connector = LocalFSConnector(connector_config)
        await connector.connect()

        docs = []
        async for doc in connector.list_documents():
            docs.append(doc)

        # Find doc1.md
        doc1 = next(d for d in docs if d.filename == "doc1.md")

        assert doc1.metadata["source_type"] == "local_fs"
        assert "relative_path" in doc1.metadata
        assert doc1.size_bytes > 0
        assert isinstance(doc1.modified_date, datetime)
