"""
Local filesystem source connector.
"""
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, BinaryIO
import fnmatch

from .base import BaseSourceConnector, SourceDocument, ConnectorError
from ..config import LocalFSConfig


class LocalFSConnector(BaseSourceConnector):
    """Connector for local filesystem sources."""

    def __init__(self, config: LocalFSConfig):
        super().__init__(config)
        self.config: LocalFSConfig = config

    async def connect(self) -> None:
        """Verify the root path exists."""
        if not self.config.root_path.exists():
            raise ConnectorError(f"Root path does not exist: {self.config.root_path}")
        if not self.config.root_path.is_dir():
            raise ConnectorError(f"Root path is not a directory: {self.config.root_path}")
        self.is_connected = True

    async def disconnect(self) -> None:
        """No cleanup needed for local filesystem."""
        self.is_connected = False

    async def list_documents(self) -> AsyncGenerator[SourceDocument, None]:
        """List all matching documents in the directory."""
        if not self.is_connected:
            raise ConnectorError("Not connected")

        pattern = "**/*" if self.config.recursive else "*"

        for file_path in self.config.root_path.glob(pattern):
            # Skip directories
            if file_path.is_dir():
                continue

            # Skip symlinks if not following
            if file_path.is_symlink() and not self.config.follow_symlinks:
                continue

            # Check extension
            if file_path.suffix.lower() not in self.config.file_extensions:
                continue

            # Check exclusion patterns
            if self._is_excluded(file_path):
                continue

            try:
                stat = file_path.stat()
                yield SourceDocument(
                    source_path=str(file_path),
                    filename=file_path.name,
                    modified_date=datetime.fromtimestamp(stat.st_mtime),
                    size_bytes=stat.st_size,
                    metadata={
                        "source_type": "local_fs",
                        "relative_path": str(file_path.relative_to(self.config.root_path)),
                    }
                )
            except (OSError, PermissionError):
                # Skip inaccessible files
                continue

    async def get_document_stream(self, source_path: str) -> BinaryIO:
        """Get file content as binary stream."""
        file_path = Path(source_path)
        if not file_path.exists():
            raise ConnectorError(f"File not found: {source_path}")

        # Return BytesIO with file content
        with open(file_path, "rb") as f:
            return BytesIO(f.read())

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if file matches any exclusion pattern."""
        try:
            relative_path = str(file_path.relative_to(self.config.root_path))
        except ValueError:
            relative_path = str(file_path)

        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(relative_path, pattern):
                return True
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
        return False
