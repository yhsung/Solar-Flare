"""
SharePoint Online source connector via Microsoft Graph API.
"""
from datetime import datetime
from io import BytesIO
from typing import AsyncGenerator, BinaryIO, Optional
import re

from .base import BaseSourceConnector, SourceDocument, ConnectorError, AuthenticationError
from ..config import SharePointConfig


class SharePointConnector(BaseSourceConnector):
    """Connector for SharePoint Online via Microsoft Graph API."""

    def __init__(self, config: SharePointConfig):
        super().__init__(config)
        self.config: SharePointConfig = config
        self._client = None
        self._access_token: Optional[str] = None
        self._site_id: Optional[str] = None

    async def connect(self) -> None:
        """Authenticate and get site ID."""
        try:
            import httpx
        except ImportError:
            raise ConnectorError("httpx not installed. Run: pip install httpx")

        try:
            from msal import ConfidentialClientApplication
        except ImportError:
            raise ConnectorError("msal not installed. Run: pip install msal")

        # Get OAuth2 token
        app = ConfidentialClientApplication(
            self.config.auth.client_id,
            authority=f"https://login.microsoftonline.com/{self.config.auth.tenant_id}",
            client_credential=self.config.auth.client_secret.get_secret_value(),
        )

        result = app.acquire_token_for_client(scopes=self.config.auth.scope)

        if "access_token" not in result:
            error_desc = result.get("error_description", "Unknown error")
            raise AuthenticationError(f"SharePoint authentication failed: {error_desc}")

        self._access_token = result["access_token"]

        # Initialize HTTP client
        self._client = httpx.AsyncClient(
            base_url="https://graph.microsoft.com/v1.0",
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
            },
            timeout=30.0,
        )

        # Get site ID from URL
        try:
            site_url = self.config.site_url.replace("https://", "").replace("http://", "")

            if "/sites/" in site_url:
                hostname, site_path = site_url.split("/sites/", 1)
                site_path = site_path.rstrip("/")
                response = await self._client.get(f"/sites/{hostname}:/sites/{site_path}")
            else:
                # Root site
                hostname = site_url.rstrip("/")
                response = await self._client.get(f"/sites/{hostname}")

            response.raise_for_status()
            self._site_id = response.json()["id"]
            self.is_connected = True

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise AuthenticationError("SharePoint authentication failed or insufficient permissions")
            raise ConnectorError(f"Failed to get SharePoint site: {e}")
        except Exception as e:
            raise ConnectorError(f"SharePoint connection failed: {e}")

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._access_token = None
        self._site_id = None
        self.is_connected = False

    async def list_documents(self) -> AsyncGenerator[SourceDocument, None]:
        """List documents from SharePoint document libraries."""
        if not self.is_connected:
            raise ConnectorError("Not connected")

        for library_name in self.config.document_libraries:
            async for doc in self._list_library_documents(library_name):
                yield doc

    async def _list_library_documents(
        self,
        library_name: str,
    ) -> AsyncGenerator[SourceDocument, None]:
        """List documents in a SharePoint document library."""
        try:
            # Get drive ID for the library
            response = await self._client.get(
                f"/sites/{self._site_id}/drives",
            )
            response.raise_for_status()
            drives = response.json().get("value", [])

            # Find matching drive by name
            drive_id = None
            for drive in drives:
                if drive.get("name", "").lower() == library_name.lower():
                    drive_id = drive["id"]
                    break

            if not drive_id:
                # Try partial match
                for drive in drives:
                    if library_name.lower() in drive.get("name", "").lower():
                        drive_id = drive["id"]
                        break

            if not drive_id:
                return

            # List items recursively
            async for doc in self._scan_folder(drive_id, "root"):
                yield doc

        except Exception as e:
            raise ConnectorError(f"Failed to list library {library_name}: {e}")

    async def _scan_folder(
        self,
        drive_id: str,
        folder_id: str,
    ) -> AsyncGenerator[SourceDocument, None]:
        """Recursively scan a SharePoint folder."""
        endpoint = f"/drives/{drive_id}/items/{folder_id}/children"

        while endpoint:
            try:
                response = await self._client.get(endpoint)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                return

            for item in data.get("value", []):
                if "folder" in item:
                    # Check if folder is in folder_paths filter (if configured)
                    folder_name = item.get("name", "")
                    if self.config.folder_paths:
                        # Only recurse if folder matches a configured path
                        parent_ref = item.get("parentReference", {})
                        item_path = parent_ref.get("path", "")
                        matches = any(
                            fp in item_path or fp == folder_name
                            for fp in self.config.folder_paths
                        )
                        if not matches:
                            continue

                    # Recurse into subfolder
                    async for doc in self._scan_folder(drive_id, item["id"]):
                        yield doc

                elif "file" in item:
                    # Check extension
                    name = item.get("name", "")
                    ext = ""
                    if '.' in name:
                        ext = f".{name.split('.')[-1].lower()}"

                    if ext in self.config.file_extensions:
                        # Parse modification date
                        modified_str = item.get("lastModifiedDateTime", "")
                        try:
                            modified_date = datetime.fromisoformat(
                                modified_str.replace("Z", "+00:00")
                            )
                        except (ValueError, AttributeError):
                            modified_date = datetime.utcnow()

                        yield SourceDocument(
                            source_path=f"sharepoint://{drive_id}/{item['id']}",
                            filename=name,
                            modified_date=modified_date,
                            size_bytes=item.get("size", 0),
                            metadata={
                                "source_type": "sharepoint",
                                "drive_id": drive_id,
                                "item_id": item["id"],
                                "web_url": item.get("webUrl", ""),
                                "created_by": (
                                    item.get("createdBy", {})
                                    .get("user", {})
                                    .get("displayName", "")
                                ),
                                "modified_by": (
                                    item.get("lastModifiedBy", {})
                                    .get("user", {})
                                    .get("displayName", "")
                                ),
                            }
                        )

            # Handle pagination
            next_link = data.get("@odata.nextLink")
            if next_link:
                # Extract path from full URL
                endpoint = next_link.replace("https://graph.microsoft.com/v1.0", "")
            else:
                endpoint = None

    async def get_document_stream(self, source_path: str) -> BinaryIO:
        """Download document content from SharePoint."""
        # Parse source path: sharepoint://DRIVE_ID/ITEM_ID
        match = re.match(r"sharepoint://([^/]+)/(.+)", source_path)
        if not match:
            raise ConnectorError(f"Invalid SharePoint source path: {source_path}")

        drive_id, item_id = match.groups()

        try:
            # Get download URL - Graph API returns 302 redirect
            response = await self._client.get(
                f"/drives/{drive_id}/items/{item_id}/content",
                follow_redirects=True,
            )

            if response.status_code == 200:
                return BytesIO(response.content)
            else:
                raise ConnectorError(
                    f"Failed to download file: HTTP {response.status_code}"
                )

        except Exception as e:
            raise ConnectorError(f"Failed to download file {source_path}: {e}")
