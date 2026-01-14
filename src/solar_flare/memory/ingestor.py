"""
Working materials ingestor for batch document ingestion.

This module provides functionality for ingesting working materials
(meetings, emails, discussions, drafts) from a directory structure.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from solar_flare.memory.multi_store import MultiStoreRAG, StoreConfig


class WorkingMaterialsIngestor:
    """
    Ingest working materials from filesystem.

    Expected directory structure:
    working_materials/
    ├── meetings/          -> document_type: "meeting_notes"
    ├── emails/            -> document_type: "email_thread"
    ├── discussions/       -> document_type: "discussion_notes"
    ├── drafts/            -> document_type: "design_draft"
    └── reviews/           -> document_type: "review_notes"
    """

    # Directory name to document type mapping
    DIR_TYPE_MAP: Dict[str, str] = {
        "meetings": "meeting_notes",
        "emails": "email_thread",
        "email": "email_thread",
        "discussions": "discussion_notes",
        "discussion": "discussion_notes",
        "drafts": "design_draft",
        "draft": "design_draft",
        "reviews": "review_notes",
        "review": "review_notes",
    }

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".log"}

    def __init__(self, multi_store: MultiStoreRAG, target_store: str = "working"):
        """
        Initialize the ingestor.

        Args:
            multi_store: MultiStoreRAG instance to add documents to
            target_store: Name of the store to add working materials to
        """
        self.multi_store = multi_store
        self.target_store = target_store

    def ingest_directory(
        self,
        root_dir: Path,
        recursive: bool = True,
    ) -> Dict[str, int]:
        """
        Scan directory and ingest all found documents.

        Args:
            root_dir: Root directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            Dict mapping document type to count ingested
        """
        root_path = Path(root_dir)
        if not root_path.exists():
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        counts: Dict[str, int] = {}

        if recursive:
            # Scan directory tree and categorize by directory name
            for file_path in root_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    doc_type = self._detect_document_type(file_path)
                    self._ingest_file(file_path, doc_type)
                    counts[doc_type] = counts.get(doc_type, 0) + 1
        else:
            # Scan only top-level files in categorized subdirectories
            for subdir in root_path.iterdir():
                if subdir.is_dir():
                    doc_type = self._detect_document_type(subdir)
                    for file_path in subdir.iterdir():
                        if file_path.is_file() and file_path.suffix in self.SUPPORTED_EXTENSIONS:
                            self._ingest_file(file_path, doc_type)
                            counts[doc_type] = counts.get(doc_type, 0) + 1

        return counts

    def _detect_document_type(self, path: Path) -> str:
        """
        Detect document type from path.

        Args:
            path: File or directory path

        Returns:
            Document type string (e.g., "meeting_notes", "email_thread")
        """
        # Check if path is a directory (for recursive mode)
        if path.is_dir():
            dir_name = path.name.lower()
        else:
            # For files, check parent directory name
            dir_name = path.parent.name.lower()

        # Look up in directory type map
        for key, doc_type in self.DIR_TYPE_MAP.items():
            if key in dir_name:
                return doc_type

        # Default to generic "working_material"
        return "working_material"

    def _extract_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Extract metadata from file path and content.

        Args:
            path: File path

        Returns:
            Metadata dictionary
        """
        metadata: Dict[str, Any] = {
            "document_type": self._detect_document_type(path),
            "source_file": str(path),
            "file_name": path.name,
        }

        # Try to extract date from filename
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if date_match:
            try:
                metadata["date"] = datetime.strptime(date_match.group(1), "%Y-%m-%d").isoformat()
            except ValueError:
                pass

        # Try to extract date from other formats
        for pattern in [r"(\d{8})", r"(\d{6})"]:
            date_match = re.search(pattern, path.stem)
            if date_match:
                date_str = date_match.group(1)
                try:
                    if len(date_str) == 8:
                        metadata["date"] = datetime.strptime(date_str, "%Y%m%d").isoformat()
                    elif len(date_str) == 6:
                        metadata["date"] = datetime.strptime(date_str, "%y%m%d").isoformat()
                except ValueError:
                    pass

        # Extract attendees from filename (for meeting notes)
        if "attendees:" in path.stem.lower():
            attendees_match = re.search(r"attendees[:\s-]+(.+)", path.stem.lower())
            if attendees_match:
                metadata["attendees"] = [
                    a.strip() for a in attendees_match.group(1).split(",")
                ]

        return metadata

    def _ingest_file(self, file_path: Path, doc_type: str) -> None:
        """
        Ingest a single file.

        Args:
            file_path: Path to file
            doc_type: Document type
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not content.strip():
                return

            metadata = self._extract_metadata(file_path)
            metadata["document_type"] = doc_type

            # Create title from filename
            title = file_path.stem.replace("_", " ").replace("-", " ").title()

            # Add to multi-store
            self.multi_store.add_text(
                store_name=self.target_store,
                text=content,
                title=title,
                standard="working_material",
                **metadata,
            )

        except Exception as e:
            print(f"Warning: Failed to ingest {file_path}: {e}")

    def ingest_single_file(
        self,
        file_path: Path,
        doc_type: Optional[str] = None,
        title: Optional[str] = None,
        **extra_metadata,
    ) -> None:
        """
        Ingest a single file with optional override metadata.

        Args:
            file_path: Path to file
            doc_type: Override document type
            title: Override title
            **extra_metadata: Additional metadata fields
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return

        # Detect or override document type
        metadata = self._extract_metadata(file_path)
        if doc_type:
            metadata["document_type"] = doc_type
        metadata.update(extra_metadata)

        # Use provided title or generate from filename
        final_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()

        self.multi_store.add_text(
            store_name=self.target_store,
            text=content,
            title=final_title,
            standard="working_material",
            **metadata,
        )

    def ingest_text(
        self,
        text: str,
        title: str,
        doc_type: str,
        **metadata,
    ) -> None:
        """
        Ingest text directly.

        Args:
            text: Text content
            title: Document title
            doc_type: Document type
            **metadata: Additional metadata fields
        """
        metadata["document_type"] = doc_type
        metadata["source"] = "direct_ingestion"

        self.multi_store.add_text(
            store_name=self.target_store,
            text=text,
            title=title,
            standard="working_material",
            **metadata,
        )


class IngestionResult(BaseModel):
    """Result of a batch ingestion operation."""

    total_files: int = 0
    successful: int = 0
    failed: int = 0
    by_type: Dict[str, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

    def add_success(self, doc_type: str) -> None:
        """Record a successful ingestion."""
        self.successful += 1
        self.by_type[doc_type] = self.by_type.get(doc_type, 0) + 1

    def add_failure(self, error: str) -> None:
        """Record a failed ingestion."""
        self.failed += 1
        self.errors.append(error)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.successful / self.total_files) * 100
