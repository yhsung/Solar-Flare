"""
Base document loader interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO
import hashlib


@dataclass
class LoadedDocument:
    """Represents a loaded document ready for ingestion."""
    content: str
    title: str
    source_path: str
    file_format: str
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    page_count: Optional[int] = None
    word_count: Optional[int] = None

    @classmethod
    def compute_hash(cls, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def __post_init__(self):
        """Compute word count if not provided."""
        if self.word_count is None and self.content:
            self.word_count = len(self.content.split())


class DocumentLoadError(Exception):
    """Exception raised when document loading fails."""

    def __init__(self, message: str, file_path: str, cause: Optional[Exception] = None):
        self.file_path = file_path
        self.cause = cause
        super().__init__(f"{message}: {file_path}")


class BaseDocumentLoader(ABC):
    """Abstract base class for document format loaders."""

    # Class attribute: supported file extensions
    SUPPORTED_EXTENSIONS: List[str] = []

    @abstractmethod
    def load(self, file_path: Path) -> LoadedDocument:
        """
        Load a document from a file path.

        Args:
            file_path: Path to the document file

        Returns:
            LoadedDocument with extracted content and metadata

        Raises:
            DocumentLoadError: If loading fails
        """
        pass

    @abstractmethod
    def load_from_stream(
        self,
        stream: BinaryIO,
        filename: str,
        source_path: str,
    ) -> LoadedDocument:
        """
        Load a document from a binary stream.

        Useful for loading from remote sources without saving to disk.

        Args:
            stream: Binary stream of document content
            filename: Original filename
            source_path: Full source path for metadata

        Returns:
            LoadedDocument with extracted content and metadata
        """
        pass

    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _extract_metadata_from_path(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path."""
        return {
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "parent_dir": file_path.parent.name,
        }

    def _generate_title(self, filename: str) -> str:
        """Generate a title from filename."""
        stem = Path(filename).stem
        return stem.replace("_", " ").replace("-", " ").title()
