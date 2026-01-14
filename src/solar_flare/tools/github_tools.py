"""
GitHub integration tools for Solar-Flare agents.

These tools enable agents to access code repositories for reference
implementations, design documents, and test suites.
"""

from typing import Optional, List, Dict, Any
import os
import base64

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class GitHubFileInput(BaseModel):
    """Input schema for fetching a single file from GitHub."""

    owner: str = Field(description="Repository owner (username or organization)")
    repo: str = Field(description="Repository name")
    path: str = Field(description="File path within the repository")
    ref: Optional[str] = Field(
        default="main",
        description="Git reference (branch, tag, or commit SHA)",
    )


class GitHubDirectoryInput(BaseModel):
    """Input schema for listing directory contents on GitHub."""

    owner: str = Field(description="Repository owner (username or organization)")
    repo: str = Field(description="Repository name")
    path: str = Field(
        default="",
        description="Directory path (empty string for repository root)",
    )
    ref: Optional[str] = Field(
        default="main",
        description="Git reference (branch, tag, or commit SHA)",
    )


def _get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers with optional authentication."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "SolarFlare/1.0",
    }

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


@tool(args_schema=GitHubFileInput)
def github_get_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "main",
) -> Dict[str, Any]:
    """
    Retrieve a file from a GitHub repository.

    Use this tool to:
    - Access reference implementations in repositories
    - Review existing logging service code for compliance analysis
    - Retrieve code examples and design documents
    - Examine test suites and verification artifacts
    - Get C/C++ header files for embedded system designs

    Args:
        owner: Repository owner (e.g., "automotive-safety-org")
        repo: Repository name (e.g., "logging-service-reference")
        path: Path to file (e.g., "src/ring_buffer.c")
        ref: Git reference - branch name, tag, or commit SHA (default: "main")

    Returns:
        Dictionary with file content, encoding info, and metadata

    Example:
        >>> result = github_get_file("arm-software", "CMSIS", "CMSIS/Core/Include/core_cm4.h")
        >>> print(result["content"][:500])
    """
    try:
        import httpx
    except ImportError:
        return {
            "error": "httpx package not installed. Run: pip install httpx",
            "name": "",
            "path": path,
            "content": "",
        }

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    headers = _get_github_headers()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params, headers=headers)

            if response.status_code == 404:
                return {
                    "error": f"File not found: {owner}/{repo}/{path}",
                    "name": "",
                    "path": path,
                    "content": "",
                }

            if response.status_code == 403:
                return {
                    "error": "API rate limit exceeded. Set GITHUB_TOKEN for higher limits.",
                    "name": "",
                    "path": path,
                    "content": "",
                }

            response.raise_for_status()
            data = response.json()

            # Check if it's a file (not a directory)
            if isinstance(data, list):
                return {
                    "error": f"Path is a directory, not a file: {path}",
                    "name": "",
                    "path": path,
                    "content": "",
                }

            # Decode content if base64 encoded
            content = data.get("content", "")
            encoding = data.get("encoding", "")

            if encoding == "base64" and content:
                try:
                    content = base64.b64decode(content).decode("utf-8")
                except Exception:
                    content = base64.b64decode(content).decode("latin-1")

            return {
                "name": data.get("name", ""),
                "path": data.get("path", ""),
                "content": content,
                "size": data.get("size", 0),
                "sha": data.get("sha", ""),
                "url": data.get("html_url", ""),
                "download_url": data.get("download_url", ""),
            }

    except httpx.HTTPStatusError as e:
        return {
            "error": f"GitHub API error {e.response.status_code}: {str(e)}",
            "name": "",
            "path": path,
            "content": "",
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch file: {str(e)}",
            "name": "",
            "path": path,
            "content": "",
        }


@tool(args_schema=GitHubDirectoryInput)
def github_list_directory(
    owner: str,
    repo: str,
    path: str = "",
    ref: str = "main",
) -> List[Dict[str, Any]]:
    """
    List contents of a directory in a GitHub repository.

    Use this tool to:
    - Explore repository structure before fetching specific files
    - Find relevant source files, headers, and documentation
    - Locate test files and verification artifacts
    - Discover available examples and reference implementations

    Args:
        owner: Repository owner (e.g., "automotive-safety-org")
        repo: Repository name (e.g., "logging-service-reference")
        path: Directory path (empty string for root)
        ref: Git reference - branch name, tag, or commit SHA (default: "main")

    Returns:
        List of files and directories with their metadata

    Example:
        >>> contents = github_list_directory("arm-software", "CMSIS", "CMSIS/Core/Include")
        >>> for item in contents:
        ...     print(f"{item['type']}: {item['name']}")
    """
    try:
        import httpx
    except ImportError:
        return [
            {
                "error": "httpx package not installed. Run: pip install httpx",
                "name": "",
                "type": "",
            }
        ]

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": ref}
    headers = _get_github_headers()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params, headers=headers)

            if response.status_code == 404:
                return [
                    {
                        "error": f"Path not found: {owner}/{repo}/{path}",
                        "name": "",
                        "type": "",
                    }
                ]

            if response.status_code == 403:
                return [
                    {
                        "error": "API rate limit exceeded. Set GITHUB_TOKEN for higher limits.",
                        "name": "",
                        "type": "",
                    }
                ]

            response.raise_for_status()
            data = response.json()

            # Handle single file response
            if isinstance(data, dict):
                return [
                    {
                        "name": data.get("name", ""),
                        "path": data.get("path", ""),
                        "type": data.get("type", "file"),
                        "size": data.get("size", 0),
                        "url": data.get("html_url", ""),
                    }
                ]

            # Handle directory listing
            results = []
            for item in data:
                results.append(
                    {
                        "name": item.get("name", ""),
                        "path": item.get("path", ""),
                        "type": item.get("type", ""),  # "file" or "dir"
                        "size": item.get("size", 0),
                        "url": item.get("html_url", ""),
                    }
                )

            # Sort: directories first, then files alphabetically
            results.sort(key=lambda x: (x["type"] != "dir", x["name"].lower()))

            return results

    except httpx.HTTPStatusError as e:
        return [
            {
                "error": f"GitHub API error {e.response.status_code}: {str(e)}",
                "name": "",
                "type": "",
            }
        ]
    except Exception as e:
        return [
            {
                "error": f"Failed to list directory: {str(e)}",
                "name": "",
                "type": "",
            }
        ]
