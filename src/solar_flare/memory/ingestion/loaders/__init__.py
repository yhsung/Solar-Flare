"""
Document loaders for various file formats.
"""
from .base import BaseDocumentLoader, LoadedDocument, DocumentLoadError
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .pptx_loader import PptxLoader
from .xlsx_loader import XlsxLoader
from .text_loader import MarkdownLoader, TextLoader

__all__ = [
    "BaseDocumentLoader",
    "LoadedDocument",
    "DocumentLoadError",
    "PDFLoader",
    "DocxLoader",
    "PptxLoader",
    "XlsxLoader",
    "MarkdownLoader",
    "TextLoader",
]


def get_loader_for_extension(extension: str) -> BaseDocumentLoader:
    """
    Get the appropriate loader for a file extension.

    Args:
        extension: File extension (e.g., ".pdf", ".docx")

    Returns:
        Appropriate document loader instance

    Raises:
        ValueError: If no loader supports the extension
    """
    extension = extension.lower()

    loaders = [
        PDFLoader(),
        DocxLoader(),
        PptxLoader(),
        XlsxLoader(),
        MarkdownLoader(),
        TextLoader(),
    ]

    for loader in loaders:
        if extension in loader.SUPPORTED_EXTENSIONS:
            return loader

    raise ValueError(f"No loader available for extension: {extension}")


def get_all_supported_extensions() -> list[str]:
    """Get all supported file extensions."""
    extensions = set()
    loaders = [
        PDFLoader(),
        DocxLoader(),
        PptxLoader(),
        XlsxLoader(),
        MarkdownLoader(),
        TextLoader(),
    ]
    for loader in loaders:
        extensions.update(loader.SUPPORTED_EXTENSIONS)
    return sorted(extensions)
