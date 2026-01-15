"""
PDF document loader using PyMuPDF (fitz).
"""
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from .base import BaseDocumentLoader, LoadedDocument, DocumentLoadError


class PDFLoader(BaseDocumentLoader):
    """Load PDF documents using PyMuPDF."""

    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(self, extract_images: bool = False, ocr_enabled: bool = False):
        """
        Initialize PDF loader.

        Args:
            extract_images: Whether to describe embedded images
            ocr_enabled: Whether to use OCR for scanned PDFs (requires tesseract)
        """
        self.extract_images = extract_images
        self.ocr_enabled = ocr_enabled

    def load(self, file_path: Path) -> LoadedDocument:
        """Load PDF from file path."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise DocumentLoadError(
                "PyMuPDF not installed. Run: pip install pymupdf",
                str(file_path)
            )

        try:
            doc = fitz.open(file_path)
            return self._extract_content(doc, str(file_path), file_path.name)
        except Exception as e:
            raise DocumentLoadError(f"Failed to load PDF: {e}", str(file_path), e)

    def load_from_stream(
        self,
        stream: BinaryIO,
        filename: str,
        source_path: str,
    ) -> LoadedDocument:
        """Load PDF from binary stream."""
        try:
            import fitz
        except ImportError:
            raise DocumentLoadError(
                "PyMuPDF not installed. Run: pip install pymupdf",
                source_path
            )

        try:
            data = stream.read()
            doc = fitz.open(stream=data, filetype="pdf")
            return self._extract_content(doc, source_path, filename)
        except Exception as e:
            raise DocumentLoadError(f"Failed to load PDF stream: {e}", source_path, e)

    def _extract_content(self, doc, source_path: str, filename: str) -> LoadedDocument:
        """Extract content from PyMuPDF document."""
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")

        content = "\n\n".join(text_parts)

        # Extract metadata
        metadata = {
            "page_count": len(doc),
            "filename": filename,
        }

        # Add PDF metadata if available
        if doc.metadata:
            pdf_meta = dict(doc.metadata)
            if pdf_meta.get("author"):
                metadata["author"] = pdf_meta["author"]
            if pdf_meta.get("subject"):
                metadata["subject"] = pdf_meta["subject"]
            if pdf_meta.get("keywords"):
                metadata["keywords"] = pdf_meta["keywords"]
            if pdf_meta.get("creationDate"):
                metadata["creation_date"] = pdf_meta["creationDate"]

        # Get title from PDF metadata or filename
        title = ""
        if doc.metadata:
            title = doc.metadata.get("title", "")
        if not title:
            title = self._generate_title(filename)

        return LoadedDocument(
            content=content,
            title=title,
            source_path=source_path,
            file_format="pdf",
            content_hash=LoadedDocument.compute_hash(content),
            metadata=metadata,
            page_count=len(doc),
        )
