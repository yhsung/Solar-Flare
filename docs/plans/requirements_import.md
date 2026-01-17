# Requirements Import from Redmine/Jira

## Overview

Add ability to fetch requirements from issue tracking systems instead of defining them manually. Support Redmine and Jira with a unified provider interface.

---

## Architecture

```
┌─────────────────────────────────────────┐
│       requirements_providers.py         │
├─────────────────────────────────────────┤
│ RequirementsProvider (base class)       │
│ ├── RedmineProvider                     │
│ └── JiraProvider                        │
│                                         │
│ load_requirements(provider) → List[Req] │
└─────────────────────────────────────────┘
```

---

## Proposed Changes

### New Module

#### [NEW] requirements_providers.py

```python
class RequirementsProvider(ABC):
    """Base class for requirements providers."""
    @abstractmethod
    def fetch_requirements(self, project: str, query: str = None) -> List[Dict]
    
class RedmineProvider(RequirementsProvider):
    """Fetch requirements from Redmine issues."""
    # Uses python-redmine library
    # Maps: issue.id → REQ-XXX, subject → title, tracker → asil_level
    
class JiraProvider(RequirementsProvider):
    """Fetch requirements from Jira issues."""
    # Uses atlassian-python-api
    # Maps: issue.key → REQ-XXX, summary → title, priority → asil_level
```

---

### Configuration

#### .env additions
```bash
# Redmine
REDMINE_URL=https://redmine.example.com
REDMINE_API_KEY=your-api-key

# Jira
JIRA_URL=https://company.atlassian.net
JIRA_USERNAME=email@example.com
JIRA_API_TOKEN=your-api-token
```

---

### Field Mapping

| Session Field | Redmine | Jira |
|---------------|---------|------|
| `id` | `REQ-{issue.id}` | `issue.key` |
| `title` | `issue.subject` | `issue.summary` |
| `description` | `issue.description` | `issue.description` |
| `priority` | `issue.priority.name` | `issue.priority.name` |
| `asil_level` | Custom field or tracker | Custom field or label |

---

### Integration with Session State

```python
from solar_flare import load_requirements_from_redmine

# Option 1: Load directly
requirements = load_requirements_from_redmine(
    project="logging-service",
    tracker="Requirement"  # Filter by tracker type
)

# Option 2: Create session from provider
session = create_session_from_provider(
    provider="redmine",
    project="logging-service"
)
```

---

### Dependencies

```toml
# pyproject.toml - optional dependencies
[project.optional-dependencies]
redmine = ["python-redmine>=2.4"]
jira = ["atlassian-python-api>=3.40"]
```

---

## Verification Plan

### Unit Tests
- Mock API responses for both providers
- Test field mapping

### Integration Tests
- Test against local Redmine/Jira instances (Docker)
