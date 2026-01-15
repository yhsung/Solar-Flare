"""
Source connectors for various data sources.
"""
from .base import (
    BaseSourceConnector,
    SourceDocument,
    ConnectorError,
    AuthenticationError,
    ConnectionTimeoutError,
)
from .local_fs import LocalFSConnector
from .smb_connector import SMBConnector
from .confluence_connector import ConfluenceConnector
from .sharepoint_connector import SharePointConnector

__all__ = [
    "BaseSourceConnector",
    "SourceDocument",
    "ConnectorError",
    "AuthenticationError",
    "ConnectionTimeoutError",
    "LocalFSConnector",
    "SMBConnector",
    "ConfluenceConnector",
    "SharePointConnector",
]
