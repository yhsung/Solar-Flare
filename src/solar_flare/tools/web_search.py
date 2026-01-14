"""
Tavily web search tool implementation for Solar-Flare agents.

This tool enables agents to search for ISO 26262, ASPICE, and automotive
embedded systems information from authoritative sources.
"""

from typing import Optional, List, Dict, Any
import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        description="Search query for automotive safety standards and embedded systems"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of results to return",
    )
    search_depth: str = Field(
        default="advanced",
        pattern="^(basic|advanced)$",
        description="Search depth: 'basic' for quick, 'advanced' for comprehensive",
    )
    include_domains: Optional[List[str]] = Field(
        default=None,
        description="Optional list of domains to prioritize (e.g., iso.org, arm.com)",
    )


# Default domains for automotive safety searches
DEFAULT_AUTOMOTIVE_DOMAINS = [
    "iso.org",
    "automotivespice.com",
    "arm.com",
    "infineon.com",
    "renesas.com",
    "nxp.com",
    "ti.com",
    "autosar.org",
]


@tool(args_schema=WebSearchInput)
def tavily_web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    include_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search the web for ISO 26262, ASPICE, and automotive embedded systems information.

    Use this tool to:
    - Research ISO 26262 and ASPICE best practices
    - Find reference implementations and design patterns
    - Look up MCU-specific documentation (ARM Cortex, AURIX, RH850, S32)
    - Search for safety analysis techniques (FMEA, FTA)
    - Find automotive industry standards and guidelines

    Args:
        query: Search query focused on automotive safety standards
        max_results: Maximum number of results (1-10)
        search_depth: "basic" for quick search, "advanced" for comprehensive
        include_domains: Optional list of domains to prioritize

    Returns:
        List of search results with title, URL, and content snippet

    Example:
        >>> results = tavily_web_search("ISO 26262 ring buffer safety requirements ASIL D")
        >>> for r in results:
        ...     print(f"{r['title']}: {r['url']}")
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        return [
            {
                "error": "Tavily package not installed. Run: pip install tavily-python",
                "title": "Installation Required",
                "url": "",
                "content": "",
            }
        ]

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [
            {
                "error": "TAVILY_API_KEY environment variable not set",
                "title": "Configuration Required",
                "url": "",
                "content": "",
            }
        ]

    client = TavilyClient(api_key=api_key)

    # Use provided domains or defaults
    domains = include_domains or DEFAULT_AUTOMOTIVE_DOMAINS

    try:
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=domains,
        )

        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                }
            )

        return results

    except Exception as e:
        return [
            {
                "error": f"Search failed: {str(e)}",
                "title": "Search Error",
                "url": "",
                "content": "",
            }
        ]
