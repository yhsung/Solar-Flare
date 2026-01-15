"""
Multi-source RAG data ingestion system.

This module provides a comprehensive data ingestion pipeline supporting:
- Multiple file formats: PDF, Word, PowerPoint, Excel, Markdown, Text
- Multiple data sources: Local filesystem, CIFS/SMB, Confluence, SharePoint
- Versioned ingestion with historical tracking
- Progress monitoring and error handling
"""
from .config import (
    SourceType,
    FileFormat,
    AuthConfig,
    APITokenAuth,
    BasicAuth,
    OAuth2Auth,
    RetryConfig,
    SourceConfig,
    LocalFSConfig,
    SMBConfig,
    ConfluenceConfig,
    SharePointConfig,
    VersioningConfig,
    IngestionConfig,
)
from .loaders import (
    BaseDocumentLoader,
    LoadedDocument,
    DocumentLoadError,
    PDFLoader,
    DocxLoader,
    PptxLoader,
    XlsxLoader,
    MarkdownLoader,
    TextLoader,
    get_loader_for_extension,
    get_all_supported_extensions,
)
from .connectors import (
    BaseSourceConnector,
    SourceDocument,
    ConnectorError,
    AuthenticationError,
    ConnectionTimeoutError,
    LocalFSConnector,
    SMBConnector,
    ConfluenceConnector,
    SharePointConnector,
)
from .versioning import VersionStore, DocumentVersion
from .progress import IngestionProgress, IngestionResult, ProgressTracker
from .orchestrator import IngestionOrchestrator, create_ingestion_orchestrator

__all__ = [
    # Config
    "SourceType",
    "FileFormat",
    "AuthConfig",
    "APITokenAuth",
    "BasicAuth",
    "OAuth2Auth",
    "RetryConfig",
    "SourceConfig",
    "LocalFSConfig",
    "SMBConfig",
    "ConfluenceConfig",
    "SharePointConfig",
    "VersioningConfig",
    "IngestionConfig",
    # Loaders
    "BaseDocumentLoader",
    "LoadedDocument",
    "DocumentLoadError",
    "PDFLoader",
    "DocxLoader",
    "PptxLoader",
    "XlsxLoader",
    "MarkdownLoader",
    "TextLoader",
    "get_loader_for_extension",
    "get_all_supported_extensions",
    # Connectors
    "BaseSourceConnector",
    "SourceDocument",
    "ConnectorError",
    "AuthenticationError",
    "ConnectionTimeoutError",
    "LocalFSConnector",
    "SMBConnector",
    "ConfluenceConnector",
    "SharePointConnector",
    # Versioning
    "VersionStore",
    "DocumentVersion",
    # Progress
    "IngestionProgress",
    "IngestionResult",
    "ProgressTracker",
    # Orchestrator
    "IngestionOrchestrator",
    "create_ingestion_orchestrator",
]
