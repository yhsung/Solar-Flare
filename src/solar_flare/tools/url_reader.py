"""
URL content reader tool for Solar-Flare agents.

This tool enables agents to fetch and read content from URLs,
useful for accessing documentation, standards, and technical articles.
"""

from typing import Dict, Any
import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class URLReaderInput(BaseModel):
    """Input schema for URL reader tool."""

    url: str = Field(description="URL to fetch and read content from")
    extract_text_only: bool = Field(
        default=True,
        description="If True, returns plain text; otherwise includes metadata",
    )


@tool(args_schema=URLReaderInput)
def read_url_content(url: str, extract_text_only: bool = True) -> Dict[str, Any]:
    """
    Fetch and read content from a URL.

    Use this tool to:
    - Access ISO 26262 and ASPICE documentation online
    - Read vendor documentation for MCUs and peripherals
    - Access technical articles and white papers
    - Read internal wiki pages (e.g., http://wiki.mediatek.inc)
    - Retrieve reference implementations and examples

    Args:
        url: The URL to fetch content from
        extract_text_only: If True, returns plain text; otherwise includes metadata

    Returns:
        Dictionary with content and metadata, or error information

    Example:
        >>> result = read_url_content("https://arm.com/documentation/...")
        >>> print(result["content"][:500])
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        return {
            "error": "Required packages not installed. Run: pip install httpx beautifulsoup4",
            "content": "",
            "url": url,
        }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SolarFlare/1.0; Automotive Safety Research)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            # Handle different content types
            if "application/json" in content_type:
                return {
                    "content": response.text,
                    "url": url,
                    "content_type": "json",
                    "status_code": response.status_code,
                }

            if "text/plain" in content_type:
                return {
                    "content": response.text,
                    "url": url,
                    "content_type": "text",
                    "status_code": response.status_code,
                }

            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            if extract_text_only:
                # Get text content
                text = soup.get_text(separator="\n", strip=True)

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = "\n".join(lines)

                return {
                    "content": content,
                    "url": url,
                    "content_type": "html",
                    "status_code": response.status_code,
                }
            else:
                # Include metadata
                title = soup.title.string if soup.title else ""
                meta_description = ""
                meta_tag = soup.find("meta", attrs={"name": "description"})
                if meta_tag:
                    meta_description = meta_tag.get("content", "")

                text = soup.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                content = "\n".join(lines)

                return {
                    "content": content,
                    "url": url,
                    "title": title,
                    "description": meta_description,
                    "content_type": "html",
                    "status_code": response.status_code,
                }

    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error {e.response.status_code}: {str(e)}",
            "content": "",
            "url": url,
        }
    except httpx.RequestError as e:
        return {
            "error": f"Request failed: {str(e)}",
            "content": "",
            "url": url,
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch URL: {str(e)}",
            "content": "",
            "url": url,
        }
