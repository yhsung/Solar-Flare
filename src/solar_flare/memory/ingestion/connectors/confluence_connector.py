"""
Confluence API source connector.
"""
from datetime import datetime
from io import BytesIO
from typing import AsyncGenerator, BinaryIO, Dict, Any, Optional
import base64
import re

from .base import BaseSourceConnector, SourceDocument, ConnectorError, AuthenticationError
from ..config import ConfluenceConfig


class ConfluenceConnector(BaseSourceConnector):
    """Connector for Confluence Cloud/Server via REST API."""

    def __init__(self, config: ConfluenceConfig):
        super().__init__(config)
        self.config: ConfluenceConfig = config
        self._client = None

    async def connect(self) -> None:
        """Initialize HTTP client for Confluence API."""
        try:
            import httpx
        except ImportError:
            raise ConnectorError("httpx not installed. Run: pip install httpx")

        # Build auth header
        auth_token = self.config.auth.token.get_secret_value()

        # Confluence Cloud uses Basic auth with email:api_token
        if self.config.auth.username:
            # Basic auth: base64(email:token)
            credentials = f"{self.config.auth.username}:{auth_token}"
            encoded = base64.b64encode(credentials.encode()).decode()
            auth_header = f"Basic {encoded}"
        else:
            # Bearer token
            auth_header = f"{self.config.auth.token_type} {auth_token}"

        headers = {
            "Authorization": auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
        )

        # Verify connection by testing API access
        try:
            response = await self._client.get("/wiki/rest/api/space", params={"limit": 1})
            response.raise_for_status()
            self.is_connected = True
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                raise AuthenticationError("Confluence authentication failed")
            raise ConnectorError(f"Confluence API error: {e}")
        except Exception as e:
            raise ConnectorError(f"Confluence connection failed: {e}")

    async def disconnect(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self.is_connected = False

    async def list_documents(self) -> AsyncGenerator[SourceDocument, None]:
        """List pages from configured Confluence spaces."""
        if not self.is_connected:
            raise ConnectorError("Not connected")

        for space_key in self.config.space_keys:
            async for doc in self._list_space_pages(space_key):
                yield doc

    async def _list_space_pages(self, space_key: str) -> AsyncGenerator[SourceDocument, None]:
        """List all pages in a Confluence space."""
        start = 0
        limit = 50
        total_fetched = 0

        while True:
            # Build query with optional label filter
            params: Dict[str, Any] = {
                "spaceKey": space_key,
                "start": start,
                "limit": limit,
                "expand": "version,metadata.labels",
            }

            try:
                response = await self._client.get("/wiki/rest/api/content", params=params)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise ConnectorError(f"Failed to list pages in space {space_key}: {e}")

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                # Apply label filter if configured
                if self.config.label_filters:
                    labels = [
                        label["name"]
                        for label in page.get("metadata", {}).get("labels", {}).get("results", [])
                    ]
                    if not any(label in labels for label in self.config.label_filters):
                        continue

                # Parse modification date
                when = page.get("version", {}).get("when", "")
                try:
                    modified_date = datetime.fromisoformat(when.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    modified_date = datetime.utcnow()

                yield SourceDocument(
                    source_path=f"confluence://{space_key}/{page['id']}",
                    filename=page["title"],
                    modified_date=modified_date,
                    size_bytes=0,  # Not available from API
                    metadata={
                        "source_type": "confluence",
                        "space_key": space_key,
                        "page_id": page["id"],
                        "version": page.get("version", {}).get("number", 1),
                        "content_type": page.get("type", "page"),
                        "url": f"{self.config.base_url}/wiki/spaces/{space_key}/pages/{page['id']}",
                    }
                )

                total_fetched += 1
                if self.config.page_limit and total_fetched >= self.config.page_limit:
                    return

            # Check for more pages
            if len(results) < limit:
                break
            start += limit

    async def get_document_stream(self, source_path: str) -> BinaryIO:
        """Get page content from Confluence."""
        # Parse source path: confluence://SPACE/PAGE_ID
        match = re.match(r"confluence://([^/]+)/(\d+)", source_path)
        if not match:
            raise ConnectorError(f"Invalid Confluence source path: {source_path}")

        space_key, page_id = match.groups()

        try:
            # Get page content with body
            response = await self._client.get(
                f"/wiki/rest/api/content/{page_id}",
                params={"expand": "body.storage,body.view"}
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise ConnectorError(f"Failed to get page {page_id}: {e}")

        # Extract HTML content and convert to plain text
        # Prefer storage format, fallback to view
        html_content = (
            data.get("body", {}).get("storage", {}).get("value", "") or
            data.get("body", {}).get("view", {}).get("value", "")
        )

        # Convert to text
        text_content = self._html_to_text(html_content)

        # Add page title at the top
        title = data.get("title", "")
        if title:
            text_content = f"# {title}\n\n{text_content}"

        return BytesIO(text_content.encode("utf-8"))

    def _html_to_text(self, html: str) -> str:
        """Convert Confluence HTML to plain text."""
        if not html:
            return ""

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up multiple newlines
            text = re.sub(r'\n{3,}', '\n\n', text)

            return text
        except ImportError:
            # Fallback: basic tag stripping
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
