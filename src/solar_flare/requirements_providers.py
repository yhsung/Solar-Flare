"""
Requirements providers for importing requirements from external issue trackers.

Supports Redmine and Jira with a unified interface for fetching requirements
and converting them to the session state format.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class RequirementsProviderType(str, Enum):
    """Supported requirements providers."""
    REDMINE = "redmine"
    JIRA = "jira"


@dataclass
class FieldMapping:
    """Mapping configuration for provider fields to session fields."""
    id_field: str = "id"
    title_field: str = "title"
    description_field: str = "description"
    priority_field: str = "priority"
    asil_field: Optional[str] = None  # Custom field name for ASIL level
    asil_default: str = "QM"


class RequirementsProvider(ABC):
    """
    Base class for requirements providers.
    
    Subclasses implement provider-specific logic for connecting to
    issue tracking systems and fetching requirements.
    """
    
    def __init__(self, field_mapping: Optional[FieldMapping] = None):
        self.field_mapping = field_mapping or FieldMapping()
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the provider."""
        pass
    
    @abstractmethod
    def fetch_requirements(
        self,
        project: str,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch requirements from the provider.
        
        Args:
            project: Project identifier
            query: Optional filter query (provider-specific)
            limit: Maximum number of requirements to fetch
            
        Returns:
            List of requirement dictionaries in session format
        """
        pass
    
    def _normalize_requirement(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a raw requirement to session format.
        
        Args:
            raw: Raw requirement data from provider
            
        Returns:
            Normalized requirement dictionary
        """
        mapping = self.field_mapping
        
        return {
            "id": str(raw.get(mapping.id_field, "")),
            "title": str(raw.get(mapping.title_field, "")),
            "description": str(raw.get(mapping.description_field, "")),
            "priority": str(raw.get(mapping.priority_field, "medium")),
            "asil_level": str(raw.get(mapping.asil_field, mapping.asil_default)) if mapping.asil_field else mapping.asil_default,
        }


class RedmineProvider(RequirementsProvider):
    """
    Fetch requirements from Redmine issues.
    
    Requires: pip install python-redmine
    
    Environment variables:
        REDMINE_URL: Redmine instance URL
        REDMINE_API_KEY: API key for authentication
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        field_mapping: Optional[FieldMapping] = None,
    ):
        super().__init__(field_mapping)
        self.url = url or os.getenv("REDMINE_URL")
        self.api_key = api_key or os.getenv("REDMINE_API_KEY")
        self._client = None
        
        # Default Redmine field mapping
        if field_mapping is None:
            self.field_mapping = FieldMapping(
                id_field="id",
                title_field="subject",
                description_field="description",
                priority_field="priority",
            )
    
    def connect(self) -> None:
        """Connect to Redmine using python-redmine library."""
        if not self.url:
            raise ValueError("Redmine URL not configured. Set REDMINE_URL environment variable.")
        
        try:
            from redminelib import Redmine
        except ImportError:
            raise ImportError(
                "python-redmine package required for Redmine provider. "
                "Install with: pip install python-redmine"
            )
        
        self._client = Redmine(self.url, key=self.api_key)
    
    def fetch_requirements(
        self,
        project: str,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch requirements from Redmine.
        
        Args:
            project: Project identifier (slug or ID)
            query: Tracker name to filter by (e.g., "Requirement", "User Story")
            limit: Maximum number of issues to fetch
            
        Returns:
            List of requirements in session format
        """
        if self._client is None:
            self.connect()
        
        # Build filter parameters
        params = {
            "project_id": project,
            "limit": limit,
            "status_id": "open",  # Only open issues
        }
        
        # Filter by tracker if specified
        if query:
            params["tracker_id"] = query
        
        issues = self._client.issue.filter(**params)
        
        requirements = []
        for issue in issues:
            raw = {
                "id": f"REQ-{issue.id}",
                "subject": issue.subject,
                "description": getattr(issue, "description", ""),
                "priority": getattr(issue.priority, "name", "medium"),
            }
            
            # Try to get ASIL level from custom field
            if self.field_mapping.asil_field:
                for cf in getattr(issue, "custom_fields", []):
                    if cf.name == self.field_mapping.asil_field:
                        raw["asil_level"] = cf.value
                        break
            
            requirements.append(self._normalize_requirement(raw))
        
        return requirements


class JiraProvider(RequirementsProvider):
    """
    Fetch requirements from Jira issues.
    
    Requires: pip install atlassian-python-api
    
    Environment variables:
        JIRA_URL: Jira instance URL (e.g., https://company.atlassian.net)
        JIRA_USERNAME: Username or email
        JIRA_API_TOKEN: API token for authentication
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        field_mapping: Optional[FieldMapping] = None,
    ):
        super().__init__(field_mapping)
        self.url = url or os.getenv("JIRA_URL")
        self.username = username or os.getenv("JIRA_USERNAME")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        self._client = None
        
        # Default Jira field mapping
        if field_mapping is None:
            self.field_mapping = FieldMapping(
                id_field="key",
                title_field="summary",
                description_field="description",
                priority_field="priority",
            )
    
    def connect(self) -> None:
        """Connect to Jira using atlassian-python-api."""
        if not self.url:
            raise ValueError("Jira URL not configured. Set JIRA_URL environment variable.")
        
        try:
            from atlassian import Jira
        except ImportError:
            raise ImportError(
                "atlassian-python-api package required for Jira provider. "
                "Install with: pip install atlassian-python-api"
            )
        
        self._client = Jira(
            url=self.url,
            username=self.username,
            password=self.api_token,
            cloud=True,  # Assume Jira Cloud by default
        )
    
    def fetch_requirements(
        self,
        project: str,
        query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch requirements from Jira.
        
        Args:
            project: Project key (e.g., "LOG", "REQ")
            query: JQL query to filter issues (e.g., "type = Requirement")
            limit: Maximum number of issues to fetch
            
        Returns:
            List of requirements in session format
        """
        if self._client is None:
            self.connect()
        
        # Build JQL query
        jql = f"project = {project}"
        if query:
            jql += f" AND {query}"
        jql += " ORDER BY created DESC"
        
        issues = self._client.jql(jql, limit=limit)
        
        requirements = []
        for issue in issues.get("issues", []):
            fields = issue.get("fields", {})
            
            raw = {
                "key": issue.get("key", ""),
                "summary": fields.get("summary", ""),
                "description": fields.get("description", "") or "",
                "priority": fields.get("priority", {}).get("name", "medium") if fields.get("priority") else "medium",
            }
            
            # Try to get ASIL level from custom field or labels
            if self.field_mapping.asil_field:
                # Check custom fields
                asil_value = fields.get(self.field_mapping.asil_field)
                if asil_value:
                    raw["asil_level"] = asil_value
            else:
                # Try to extract from labels
                labels = fields.get("labels", [])
                for label in labels:
                    if label.upper().startswith("ASIL"):
                        raw["asil_level"] = label
                        break
            
            requirements.append(self._normalize_requirement(raw))
        
        return requirements


def create_requirements_provider(
    provider: str | RequirementsProviderType,
    **kwargs,
) -> RequirementsProvider:
    """
    Factory function to create a requirements provider.
    
    Args:
        provider: Provider type ('redmine' or 'jira')
        **kwargs: Provider-specific configuration
        
    Returns:
        Configured RequirementsProvider instance
    """
    if isinstance(provider, str):
        provider = RequirementsProviderType(provider.lower())
    
    if provider == RequirementsProviderType.REDMINE:
        return RedmineProvider(**kwargs)
    elif provider == RequirementsProviderType.JIRA:
        return JiraProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def load_requirements_from_redmine(
    project: str,
    tracker: Optional[str] = None,
    limit: int = 100,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to load requirements from Redmine.
    
    Args:
        project: Project identifier
        tracker: Tracker name to filter (e.g., "Requirement")
        limit: Maximum requirements to fetch
        url: Redmine URL (defaults to REDMINE_URL env var)
        api_key: API key (defaults to REDMINE_API_KEY env var)
        
    Returns:
        List of requirements in session format
    """
    provider = RedmineProvider(url=url, api_key=api_key)
    return provider.fetch_requirements(project, query=tracker, limit=limit)


def load_requirements_from_jira(
    project: str,
    jql: Optional[str] = None,
    limit: int = 100,
    url: Optional[str] = None,
    username: Optional[str] = None,
    api_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to load requirements from Jira.
    
    Args:
        project: Project key (e.g., "LOG")
        jql: Additional JQL filter (e.g., "type = Requirement")
        limit: Maximum requirements to fetch
        url: Jira URL (defaults to JIRA_URL env var)
        username: Username (defaults to JIRA_USERNAME env var)
        api_token: API token (defaults to JIRA_API_TOKEN env var)
        
    Returns:
        List of requirements in session format
    """
    provider = JiraProvider(url=url, username=username, api_token=api_token)
    return provider.fetch_requirements(project, query=jql, limit=limit)
