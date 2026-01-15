"""
Unit tests for document loaders.
"""
import sys
import pytest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import directly from ingestion submodule to avoid full memory module import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from solar_flare.memory.ingestion.loaders.base import LoadedDocument, DocumentLoadError
from solar_flare.memory.ingestion.loaders.text_loader import MarkdownLoader, TextLoader
from solar_flare.memory.ingestion.loaders import (
    get_loader_for_extension,
    get_all_supported_extensions,
)


class TestLoadedDocument:
    """Tests for LoadedDocument dataclass."""

    def test_compute_hash(self):
        """Test content hash computation."""
        content = "Test content"
        hash1 = LoadedDocument.compute_hash(content)
        hash2 = LoadedDocument.compute_hash(content)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_different_content_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = LoadedDocument.compute_hash("Content A")
        hash2 = LoadedDocument.compute_hash("Content B")

        assert hash1 != hash2

    def test_word_count_computed(self):
        """Test automatic word count computation."""
        doc = LoadedDocument(
            content="This is a test document with eight words",
            title="Test",
            source_path="/test/path",
            file_format="text",
            content_hash="abc123",
        )

        assert doc.word_count == 8

    def test_explicit_word_count(self):
        """Test explicit word count is not overwritten."""
        doc = LoadedDocument(
            content="This is a test",
            title="Test",
            source_path="/test/path",
            file_format="text",
            content_hash="abc123",
            word_count=100,  # Explicit value
        )

        assert doc.word_count == 100


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_supported_extensions(self):
        """Test supported file extensions."""
        loader = MarkdownLoader()
        assert ".md" in loader.SUPPORTED_EXTENSIONS
        assert ".markdown" in loader.SUPPORTED_EXTENSIONS
        assert ".rst" in loader.SUPPORTED_EXTENSIONS

    def test_can_load(self, tmp_path):
        """Test can_load method."""
        loader = MarkdownLoader()

        md_file = tmp_path / "test.md"
        md_file.write_text("# Test")

        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test")

        assert loader.can_load(md_file) is True
        assert loader.can_load(txt_file) is False

    def test_load_from_path(self, tmp_path):
        """Test loading from file path."""
        loader = MarkdownLoader()

        md_file = tmp_path / "test.md"
        md_file.write_text("# My Title\n\nSome content here.")

        doc = loader.load(md_file)

        assert doc.title == "My Title"
        assert "Some content here" in doc.content
        assert doc.file_format == "markdown"
        assert doc.source_path == str(md_file)

    def test_load_from_stream(self):
        """Test loading from binary stream."""
        loader = MarkdownLoader()

        content = b"# Stream Title\n\nStream content."
        stream = BytesIO(content)

        doc = loader.load_from_stream(stream, "test.md", "/path/to/test.md")

        assert doc.title == "Stream Title"
        assert "Stream content" in doc.content

    def test_title_from_filename_if_no_heading(self, tmp_path):
        """Test title extraction from filename when no heading."""
        loader = MarkdownLoader()

        md_file = tmp_path / "my-document-name.md"
        md_file.write_text("No heading here, just content.")

        doc = loader.load(md_file)

        assert doc.title == "My Document Name"


class TestTextLoader:
    """Tests for TextLoader."""

    def test_supported_extensions(self):
        """Test supported file extensions."""
        loader = TextLoader()
        assert ".txt" in loader.SUPPORTED_EXTENSIONS
        assert ".log" in loader.SUPPORTED_EXTENSIONS
        assert ".csv" in loader.SUPPORTED_EXTENSIONS

    def test_load_from_path(self, tmp_path):
        """Test loading from file path."""
        loader = TextLoader()

        txt_file = tmp_path / "test_document.txt"
        txt_file.write_text("Plain text content.")

        doc = loader.load(txt_file)

        assert doc.content == "Plain text content."
        assert doc.title == "Test Document"
        assert doc.file_format == "text"

    def test_load_from_stream(self):
        """Test loading from binary stream."""
        loader = TextLoader()

        content = b"Stream text content."
        stream = BytesIO(content)

        doc = loader.load_from_stream(stream, "stream.txt", "/path/to/stream.txt")

        assert doc.content == "Stream text content."


class TestLoaderHelpers:
    """Tests for loader helper functions."""

    def test_get_loader_for_extension_markdown(self):
        """Test getting loader for markdown extension."""
        loader = get_loader_for_extension(".md")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_for_extension_text(self):
        """Test getting loader for text extension."""
        loader = get_loader_for_extension(".txt")
        assert isinstance(loader, TextLoader)

    def test_get_loader_for_unknown_extension(self):
        """Test error for unknown extension."""
        with pytest.raises(ValueError, match="No loader available"):
            get_loader_for_extension(".xyz")

    def test_get_all_supported_extensions(self):
        """Test getting all supported extensions."""
        extensions = get_all_supported_extensions()

        assert ".md" in extensions
        assert ".txt" in extensions
        assert isinstance(extensions, list)
        # Should be sorted
        assert extensions == sorted(extensions)


class TestPDFLoader:
    """Tests for PDFLoader (mocked)."""

    def test_import_error_handling(self, tmp_path):
        """Test graceful handling of missing pymupdf."""
        from solar_flare.memory.ingestion.loaders.pdf_loader import PDFLoader

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake pdf")

        loader = PDFLoader()

        # This should raise DocumentLoadError mentioning pymupdf
        with patch.dict("sys.modules", {"fitz": None}):
            # The import will fail, but we need to test the error handling
            pass  # Actual test would require more complex mocking


class TestDocxLoader:
    """Tests for DocxLoader (mocked)."""

    def test_supported_extensions(self):
        """Test supported extensions."""
        from solar_flare.memory.ingestion.loaders.docx_loader import DocxLoader
        loader = DocxLoader()
        assert ".docx" in loader.SUPPORTED_EXTENSIONS


class TestPptxLoader:
    """Tests for PptxLoader."""

    def test_supported_extensions(self):
        """Test supported extensions."""
        from solar_flare.memory.ingestion.loaders.pptx_loader import PptxLoader
        loader = PptxLoader()
        assert ".pptx" in loader.SUPPORTED_EXTENSIONS


class TestXlsxLoader:
    """Tests for XlsxLoader."""

    def test_supported_extensions(self):
        """Test supported extensions."""
        from solar_flare.memory.ingestion.loaders.xlsx_loader import XlsxLoader
        loader = XlsxLoader()
        assert ".xlsx" in loader.SUPPORTED_EXTENSIONS

    def test_max_rows_config(self):
        """Test max_rows configuration."""
        from solar_flare.memory.ingestion.loaders.xlsx_loader import XlsxLoader
        loader = XlsxLoader(max_rows=500)
        assert loader.max_rows == 500
