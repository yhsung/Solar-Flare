"""
Base source connector interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional, BinaryIO


@dataclass
class SourceDocument:
    """Represents a document from a source before loading."""
    source_path: str
    filename: str
    modified_date: datetime
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectorError(Exception):
    """Base exception for connector errors."""
    pass


class AuthenticationError(ConnectorError):
    """Authentication failed."""
    pass


class ConnectionTimeoutError(ConnectorError):
    """Connection timed out."""
    pass


class BaseSourceConnector(ABC):
    """Abstract base class for source connectors."""

    def __init__(self, config: Any):
        """
        Initialize connector with configuration.

        Args:
            config: Source-specific configuration
        """
        self.config = config
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the source.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the source."""
        pass

    @abstractmethod
    async def list_documents(self) -> AsyncGenerator[SourceDocument, None]:
        """
        List all documents from the source.

        Yields:
            SourceDocument for each discoverable document
        """
        # This is a workaround for abstract async generators
        yield  # type: ignore
        raise NotImplementedError

    @abstractmethod
    async def get_document_stream(self, source_path: str) -> BinaryIO:
        """
        Get a binary stream for a document.

        Args:
            source_path: Path/identifier of the document

        Returns:
            Binary stream of document content
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
