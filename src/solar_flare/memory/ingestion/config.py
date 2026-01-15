"""
Configuration models for multi-source RAG ingestion.
"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, SecretStr, field_validator


class SourceType(str, Enum):
    """Supported source types."""
    LOCAL_FS = "local_fs"
    SMB = "smb"
    CONFLUENCE = "confluence"
    SHAREPOINT = "sharepoint"


class FileFormat(str, Enum):
    """Supported file formats."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"


class AuthConfig(BaseModel):
    """Base authentication configuration."""
    auth_type: str = Field(description="Authentication type identifier")


class APITokenAuth(AuthConfig):
    """API token authentication for cloud sources."""
    auth_type: str = "api_token"
    token: SecretStr = Field(description="API token/key")
    token_type: str = Field(default="Bearer", description="Token type (Bearer, Basic, etc.)")

    # For Confluence basic auth (email:token)
    username: Optional[str] = Field(default=None, description="Username/email for basic auth")


class BasicAuth(AuthConfig):
    """Username/password authentication."""
    auth_type: str = "basic"
    username: str = Field(description="Username")
    password: SecretStr = Field(description="Password")
    domain: Optional[str] = Field(default=None, description="Domain for NTLM auth")


class OAuth2Auth(AuthConfig):
    """OAuth2 authentication for SharePoint/Graph API."""
    auth_type: str = "oauth2"
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: SecretStr = Field(description="OAuth2 client secret")
    tenant_id: str = Field(description="Azure AD tenant ID")
    scope: List[str] = Field(
        default_factory=lambda: ["https://graph.microsoft.com/.default"],
        description="OAuth2 scopes"
    )


class RetryConfig(BaseModel):
    """Retry configuration for error handling."""
    max_retries: int = Field(default=3, ge=0, le=10)
    initial_delay_seconds: float = Field(default=1.0, ge=0.1)
    max_delay_seconds: float = Field(default=60.0, ge=1.0)
    exponential_base: float = Field(default=2.0, ge=1.0)


class SourceConfig(BaseModel):
    """Base configuration for all source connectors."""
    name: str = Field(description="Unique source identifier")
    source_type: SourceType = Field(description="Type of source")
    target_store: str = Field(
        default="working",
        description="Target vector store (standards, internal, working)"
    )
    enabled: bool = Field(default=True, description="Whether this source is enabled")
    file_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".md", ".txt", ".docx", ".pptx", ".xlsx"],
        description="File extensions to process"
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Glob patterns to exclude"
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to attach to all documents"
    )


class LocalFSConfig(SourceConfig):
    """Configuration for local filesystem source."""
    source_type: SourceType = SourceType.LOCAL_FS
    root_path: Path = Field(description="Root directory to scan")
    recursive: bool = Field(default=True, description="Scan subdirectories")
    follow_symlinks: bool = Field(default=False, description="Follow symbolic links")


class SMBConfig(SourceConfig):
    """Configuration for CIFS/SMB share source."""
    source_type: SourceType = SourceType.SMB
    server: str = Field(description="SMB server hostname or IP")
    share: str = Field(description="Share name")
    path: str = Field(default="/", description="Path within share")
    port: int = Field(default=445, description="SMB port")
    auth: Optional[BasicAuth] = Field(default=None, description="Authentication")
    timeout_seconds: int = Field(default=30, description="Connection timeout")


class ConfluenceConfig(SourceConfig):
    """Configuration for Confluence API source."""
    source_type: SourceType = SourceType.CONFLUENCE
    base_url: str = Field(description="Confluence base URL (e.g., https://company.atlassian.net)")
    space_keys: List[str] = Field(description="Space keys to ingest")
    auth: APITokenAuth = Field(description="API token authentication")
    include_attachments: bool = Field(default=True, description="Include page attachments")
    include_comments: bool = Field(default=False, description="Include page comments")
    page_limit: Optional[int] = Field(default=None, description="Max pages per space")
    label_filters: List[str] = Field(
        default_factory=list,
        description="Only include pages with these labels"
    )


class SharePointConfig(SourceConfig):
    """Configuration for SharePoint API source."""
    source_type: SourceType = SourceType.SHAREPOINT
    site_url: str = Field(description="SharePoint site URL")
    document_libraries: List[str] = Field(
        default_factory=lambda: ["Documents"],
        description="Document library names to scan"
    )
    auth: OAuth2Auth = Field(description="OAuth2 authentication")
    include_versions: bool = Field(default=True, description="Include document versions")
    folder_paths: List[str] = Field(
        default_factory=list,
        description="Specific folder paths (empty = all)"
    )


class VersioningConfig(BaseModel):
    """Configuration for document versioning."""
    enabled: bool = Field(default=True, description="Enable versioned ingestion")
    db_path: Path = Field(
        default=Path("./data/ingestion/versions.db"),
        description="SQLite database path for version tracking"
    )
    keep_all_versions: bool = Field(
        default=True,
        description="Keep all versions (vs. only latest)"
    )
    version_comparison_fields: List[str] = Field(
        default_factory=lambda: ["content_hash", "modified_date"],
        description="Fields to compare for version detection"
    )


class IngestionConfig(BaseModel):
    """Master configuration for the ingestion system."""
    sources: List[Union[LocalFSConfig, SMBConfig, ConfluenceConfig, SharePointConfig]] = Field(
        default_factory=list,
        description="List of source configurations"
    )
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    batch_size: int = Field(default=50, ge=1, le=500, description="Documents per batch")
    parallel_sources: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max parallel source connections"
    )
    progress_callback_interval: int = Field(
        default=10,
        description="Progress callback every N documents"
    )

    @field_validator("sources", mode="before")
    @classmethod
    def validate_unique_names(cls, v):
        """Ensure all source names are unique."""
        if not v:
            return v
        names = []
        for s in v:
            if isinstance(s, dict):
                names.append(s.get("name", ""))
            else:
                names.append(getattr(s, "name", ""))
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")
        return v
