"""
SMB/CIFS shared folder source connector.
"""
from datetime import datetime
from io import BytesIO
from typing import AsyncGenerator, BinaryIO, Optional
import asyncio
import uuid

from .base import BaseSourceConnector, SourceDocument, ConnectorError, AuthenticationError
from ..config import SMBConfig


class SMBConnector(BaseSourceConnector):
    """Connector for CIFS/SMB shared folders."""

    def __init__(self, config: SMBConfig):
        super().__init__(config)
        self.config: SMBConfig = config
        self._connection = None
        self._session = None
        self._tree = None

    async def connect(self) -> None:
        """Establish SMB connection."""
        try:
            from smbprotocol.connection import Connection
            from smbprotocol.session import Session
            from smbprotocol.tree import TreeConnect
        except ImportError:
            raise ConnectorError(
                "smbprotocol not installed. Run: pip install smbprotocol"
            )

        try:
            # Create connection
            self._connection = Connection(
                uuid.uuid4(),
                self.config.server,
                self.config.port,
            )
            await asyncio.to_thread(self._connection.connect)

            # Create session with auth
            if self.config.auth:
                self._session = Session(
                    self._connection,
                    self.config.auth.username,
                    self.config.auth.password.get_secret_value(),
                )
            else:
                # Guest/anonymous access
                self._session = Session(self._connection, "", "")

            await asyncio.to_thread(self._session.connect)

            # Connect to share
            share_path = f"\\\\{self.config.server}\\{self.config.share}"
            self._tree = TreeConnect(self._session, share_path)
            await asyncio.to_thread(self._tree.connect)

            self.is_connected = True

        except Exception as e:
            error_str = str(e)
            if "STATUS_LOGON_FAILURE" in error_str or "STATUS_ACCESS_DENIED" in error_str:
                raise AuthenticationError(f"SMB authentication failed: {e}")
            raise ConnectorError(f"SMB connection failed: {e}")

    async def disconnect(self) -> None:
        """Close SMB connection."""
        try:
            if self._tree:
                await asyncio.to_thread(self._tree.disconnect)
            if self._session:
                await asyncio.to_thread(self._session.disconnect)
            if self._connection:
                await asyncio.to_thread(self._connection.disconnect)
        except Exception:
            pass  # Best effort cleanup
        finally:
            self._tree = None
            self._session = None
            self._connection = None
            self.is_connected = False

    async def list_documents(self) -> AsyncGenerator[SourceDocument, None]:
        """List documents from SMB share."""
        if not self.is_connected:
            raise ConnectorError("Not connected")

        async for doc in self._scan_directory(self.config.path.lstrip("/")):
            yield doc

    async def _scan_directory(self, path: str) -> AsyncGenerator[SourceDocument, None]:
        """Recursively scan directory."""
        try:
            from smbprotocol.open import (
                Open,
                CreateDisposition,
                FilePipePrinterAccessMask,
                ShareAccess,
                CreateOptions,
                FileAttributes,
                ImpersonationLevel,
            )
            from smbprotocol.file_info import FileDirectoryInformation

            # Open directory
            dir_open = Open(self._tree, path or "")
            await asyncio.to_thread(
                dir_open.create,
                ImpersonationLevel.Impersonation,
                FilePipePrinterAccessMask.FILE_LIST_DIRECTORY,
                FileAttributes.FILE_ATTRIBUTE_DIRECTORY,
                ShareAccess.FILE_SHARE_READ,
                CreateDisposition.FILE_OPEN,
                CreateOptions.FILE_DIRECTORY_FILE,
            )

            try:
                # Query directory contents
                entries = await asyncio.to_thread(
                    dir_open.query_directory,
                    "*",
                    FileDirectoryInformation,
                )

                for entry in entries:
                    name = entry["file_name"].get_value()
                    if name in (".", ".."):
                        continue

                    full_path = f"{path}\\{name}" if path else name
                    attrs = entry["file_attributes"].get_value()

                    if attrs & FileAttributes.FILE_ATTRIBUTE_DIRECTORY:
                        # Recurse into subdirectory
                        async for doc in self._scan_directory(full_path):
                            yield doc
                    else:
                        # Check extension
                        ext = f".{name.split('.')[-1].lower()}" if '.' in name else ""
                        if ext in self.config.file_extensions:
                            # Convert FILETIME to datetime
                            # FILETIME is 100-nanosecond intervals since 1601-01-01
                            filetime = entry["last_write_time"].get_value()
                            timestamp = (filetime / 10000000) - 11644473600
                            modified_date = datetime.fromtimestamp(timestamp)

                            yield SourceDocument(
                                source_path=full_path,
                                filename=name,
                                modified_date=modified_date,
                                size_bytes=entry["end_of_file"].get_value(),
                                metadata={
                                    "source_type": "smb",
                                    "server": self.config.server,
                                    "share": self.config.share,
                                }
                            )
            finally:
                await asyncio.to_thread(dir_open.close)

        except Exception as e:
            # Log and skip inaccessible directories
            pass

    async def get_document_stream(self, source_path: str) -> BinaryIO:
        """Get file content from SMB share."""
        try:
            from smbprotocol.open import (
                Open,
                CreateDisposition,
                FilePipePrinterAccessMask,
                ShareAccess,
                CreateOptions,
                FileAttributes,
                ImpersonationLevel,
            )

            file_open = Open(self._tree, source_path)
            await asyncio.to_thread(
                file_open.create,
                ImpersonationLevel.Impersonation,
                FilePipePrinterAccessMask.FILE_READ_DATA,
                FileAttributes.FILE_ATTRIBUTE_NORMAL,
                ShareAccess.FILE_SHARE_READ,
                CreateDisposition.FILE_OPEN,
                CreateOptions.FILE_NON_DIRECTORY_FILE,
            )

            try:
                # Read entire file
                file_size = file_open.end_of_file
                data = await asyncio.to_thread(file_open.read, 0, file_size)
                return BytesIO(data)
            finally:
                await asyncio.to_thread(file_open.close)

        except Exception as e:
            raise ConnectorError(f"Failed to read file {source_path}: {e}")
