"""
Markdown and plain text document loaders.
"""
from pathlib import Path
from typing import BinaryIO, Dict, Any

from .base import BaseDocumentLoader, LoadedDocument, DocumentLoadError


class MarkdownLoader(BaseDocumentLoader):
    """Load Markdown documents."""

    SUPPORTED_EXTENSIONS = [".md", ".markdown", ".rst"]

    def load(self, file_path: Path) -> LoadedDocument:
        """Load Markdown from file path."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            raise DocumentLoadError(f"Failed to load Markdown: {e}", str(file_path), e)

        return self._create_document(content, str(file_path), file_path.name)

    def load_from_stream(
        self,
        stream: BinaryIO,
        filename: str,
        source_path: str,
    ) -> LoadedDocument:
        """Load Markdown from binary stream."""
        try:
            content = stream.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise DocumentLoadError(f"Failed to load Markdown stream: {e}", source_path, e)

        return self._create_document(content, source_path, filename)

    def _create_document(self, content: str, source_path: str, filename: str) -> LoadedDocument:
        """Create LoadedDocument from content."""
        # Try to extract title from first heading
        title = self._generate_title(filename)
        for line in content.split("\n")[:10]:
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                break

        metadata: Dict[str, Any] = {"filename": filename}

        return LoadedDocument(
            content=content,
            title=title,
            source_path=source_path,
            file_format="markdown",
            content_hash=LoadedDocument.compute_hash(content),
            metadata=metadata,
        )


class TextLoader(BaseDocumentLoader):
    """Load plain text documents."""

    SUPPORTED_EXTENSIONS = [".txt", ".log", ".csv", ".json", ".xml", ".yaml", ".yml"]

    def load(self, file_path: Path) -> LoadedDocument:
        """Load text from file path."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            raise DocumentLoadError(f"Failed to load text: {e}", str(file_path), e)

        title = self._generate_title(file_path.name)

        return LoadedDocument(
            content=content,
            title=title,
            source_path=str(file_path),
            file_format="text",
            content_hash=LoadedDocument.compute_hash(content),
            metadata={"filename": file_path.name},
        )

    def load_from_stream(
        self,
        stream: BinaryIO,
        filename: str,
        source_path: str,
    ) -> LoadedDocument:
        """Load text from binary stream."""
        try:
            content = stream.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise DocumentLoadError(f"Failed to load text stream: {e}", source_path, e)

        title = self._generate_title(filename)

        return LoadedDocument(
            content=content,
            title=title,
            source_path=source_path,
            file_format="text",
            content_hash=LoadedDocument.compute_hash(content),
            metadata={"filename": filename},
        )
