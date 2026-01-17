"""
Unit tests for requirements_providers module.

Tests cover:
- Base provider interface
- RedmineProvider with mocked API
- JiraProvider with mocked API
- Convenience functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from solar_flare.requirements_providers import (
    RequirementsProvider,
    RedmineProvider,
    JiraProvider,
    FieldMapping,
    create_requirements_provider,
    load_requirements_from_redmine,
    load_requirements_from_jira,
    RequirementsProviderType,
)


class TestFieldMapping:
    """Tests for FieldMapping dataclass."""
    
    def test_default_mapping(self):
        mapping = FieldMapping()
        assert mapping.id_field == "id"
        assert mapping.title_field == "title"
        assert mapping.description_field == "description"
        assert mapping.priority_field == "priority"
        assert mapping.asil_default == "QM"
    
    def test_custom_mapping(self):
        mapping = FieldMapping(
            id_field="key",
            title_field="summary",
            asil_field="custom_asil",
            asil_default="ASIL-B",
        )
        assert mapping.id_field == "key"
        assert mapping.title_field == "summary"
        assert mapping.asil_field == "custom_asil"
        assert mapping.asil_default == "ASIL-B"


class TestRedmineProvider:
    """Tests for RedmineProvider."""
    
    def test_init_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("REDMINE_URL", "https://redmine.example.com")
        monkeypatch.setenv("REDMINE_API_KEY", "test-key")
        
        provider = RedmineProvider()
        assert provider.url == "https://redmine.example.com"
        assert provider.api_key == "test-key"
    
    def test_init_with_explicit_values(self):
        provider = RedmineProvider(
            url="https://custom.redmine.com",
            api_key="custom-key",
        )
        assert provider.url == "https://custom.redmine.com"
        assert provider.api_key == "custom-key"
    
    def test_connect_missing_url(self, monkeypatch):
        monkeypatch.delenv("REDMINE_URL", raising=False)
        provider = RedmineProvider(url=None)
        
        with pytest.raises(ValueError, match="REDMINE_URL"):
            provider.connect()
    
    @patch("solar_flare.requirements_providers.RedmineProvider.connect")
    def test_fetch_requirements(self, mock_connect):
        """Test fetching requirements with mocked Redmine client."""
        provider = RedmineProvider(url="https://test.com", api_key="key")
        
        # Mock the client and issues
        mock_issue = Mock()
        mock_issue.id = 123
        mock_issue.subject = "Test Requirement"
        mock_issue.description = "Description text"
        mock_issue.priority = Mock(name="High")
        mock_issue.custom_fields = []
        
        provider._client = Mock()
        provider._client.issue.filter.return_value = [mock_issue]
        
        requirements = provider.fetch_requirements("test-project")
        
        assert len(requirements) == 1
        assert requirements[0]["id"] == "REQ-123"
        assert requirements[0]["title"] == "Test Requirement"
    
    def test_normalize_requirement(self):
        provider = RedmineProvider(url="https://test.com", api_key="key")
        
        raw = {
            "id": "REQ-001",
            "subject": "My Requirement",
            "description": "Details here",
            "priority": "High",
        }
        
        normalized = provider._normalize_requirement(raw)
        
        assert normalized["id"] == "REQ-001"
        assert normalized["title"] == "My Requirement"
        assert normalized["priority"] == "High"
        assert normalized["asil_level"] == "QM"  # Default


class TestJiraProvider:
    """Tests for JiraProvider."""
    
    def test_init_with_env_vars(self, monkeypatch):
        monkeypatch.setenv("JIRA_URL", "https://company.atlassian.net")
        monkeypatch.setenv("JIRA_USERNAME", "user@example.com")
        monkeypatch.setenv("JIRA_API_TOKEN", "jira-token")
        
        provider = JiraProvider()
        assert provider.url == "https://company.atlassian.net"
        assert provider.username == "user@example.com"
        assert provider.api_token == "jira-token"
    
    def test_init_with_explicit_values(self):
        provider = JiraProvider(
            url="https://custom.atlassian.net",
            username="admin",
            api_token="secret",
        )
        assert provider.url == "https://custom.atlassian.net"
        assert provider.username == "admin"
    
    def test_connect_missing_url(self, monkeypatch):
        monkeypatch.delenv("JIRA_URL", raising=False)
        provider = JiraProvider(url=None)
        
        with pytest.raises(ValueError, match="JIRA_URL"):
            provider.connect()
    
    @patch("solar_flare.requirements_providers.JiraProvider.connect")
    def test_fetch_requirements(self, mock_connect):
        """Test fetching requirements with mocked Jira client."""
        provider = JiraProvider(url="https://test.atlassian.net", username="u", api_token="t")
        
        # Mock the client and issues
        mock_response = {
            "issues": [
                {
                    "key": "LOG-123",
                    "fields": {
                        "summary": "Logging Feature",
                        "description": "Implement logging",
                        "priority": {"name": "High"},
                        "labels": ["ASIL-D"],
                    }
                }
            ]
        }
        
        provider._client = Mock()
        provider._client.jql.return_value = mock_response
        
        requirements = provider.fetch_requirements("LOG")
        
        assert len(requirements) == 1
        assert requirements[0]["id"] == "LOG-123"
        assert requirements[0]["title"] == "Logging Feature"
        assert requirements[0]["asil_level"] == "QM"  # Default since asil_field not configured


class TestFactoryFunction:
    """Tests for create_requirements_provider factory."""
    
    def test_create_redmine_provider(self):
        provider = create_requirements_provider(
            "redmine",
            url="https://redmine.example.com",
            api_key="key",
        )
        assert isinstance(provider, RedmineProvider)
    
    def test_create_jira_provider(self):
        provider = create_requirements_provider(
            RequirementsProviderType.JIRA,
            url="https://jira.example.com",
            username="user",
            api_token="token",
        )
        assert isinstance(provider, JiraProvider)
    
    def test_invalid_provider(self):
        with pytest.raises(ValueError):
            create_requirements_provider("unknown")


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @patch.object(RedmineProvider, "fetch_requirements")
    @patch.object(RedmineProvider, "connect")
    def test_load_requirements_from_redmine(self, mock_connect, mock_fetch):
        mock_fetch.return_value = [{"id": "REQ-1", "title": "Test"}]
        
        requirements = load_requirements_from_redmine(
            project="test",
            url="https://redmine.example.com",
            api_key="key",
        )
        
        assert len(requirements) == 1
        mock_fetch.assert_called_once()
    
    @patch.object(JiraProvider, "fetch_requirements")
    @patch.object(JiraProvider, "connect")
    def test_load_requirements_from_jira(self, mock_connect, mock_fetch):
        mock_fetch.return_value = [{"id": "LOG-1", "title": "Test"}]
        
        requirements = load_requirements_from_jira(
            project="LOG",
            url="https://jira.example.com",
            username="user",
            api_token="token",
        )
        
        assert len(requirements) == 1
        mock_fetch.assert_called_once()
