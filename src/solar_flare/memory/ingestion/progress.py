"""
Progress tracking for ingestion operations.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Callable


@dataclass
class IngestionProgress:
    """Progress update for ingestion operations."""
    source_name: str
    total_documents: int
    processed: int
    successful: int
    failed: int
    skipped: int = 0
    current_file: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def percent_complete(self) -> float:
        """Calculate completion percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed / self.total_documents) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.processed == 0:
            return 100.0
        return (self.successful / self.processed) * 100

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time."""
        return (datetime.utcnow() - self.started_at).total_seconds()

    @property
    def documents_per_second(self) -> float:
        """Calculate processing rate."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.processed / elapsed

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time."""
        rate = self.documents_per_second
        if rate == 0:
            return None
        remaining = self.total_documents - self.processed
        return remaining / rate


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    skipped_unchanged: int = 0
    by_source: Dict[str, int] = field(default_factory=dict)
    by_format: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    versions_created: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        processed = self.successful + self.failed
        if processed == 0:
            return 100.0
        return (self.successful / processed) * 100

    def add_success(self, source_name: str, file_format: str) -> None:
        """Record a successful ingestion."""
        self.successful += 1
        self.by_source[source_name] = self.by_source.get(source_name, 0) + 1
        self.by_format[file_format] = self.by_format.get(file_format, 0) + 1

    def add_failure(self, error: str) -> None:
        """Record a failed ingestion."""
        self.failed += 1
        self.errors.append(error)

    def add_skipped(self) -> None:
        """Record a skipped (unchanged) document."""
        self.skipped_unchanged += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_documents": self.total_documents,
            "successful": self.successful,
            "failed": self.failed,
            "skipped_unchanged": self.skipped_unchanged,
            "success_rate": self.success_rate,
            "by_source": self.by_source,
            "by_format": self.by_format,
            "errors": self.errors[:10],  # Limit errors in output
            "total_errors": len(self.errors),
            "duration_seconds": self.duration_seconds,
            "versions_created": self.versions_created,
        }


class ProgressTracker:
    """Track progress across multiple sources."""

    def __init__(self, callback: Optional[Callable[[IngestionProgress], None]] = None):
        """
        Initialize progress tracker.

        Args:
            callback: Optional callback for progress updates
        """
        self.sources: Dict[str, IngestionProgress] = {}
        self.started_at = datetime.utcnow()
        self.callback = callback

    def start_source(self, source_name: str, total_documents: int = 0) -> IngestionProgress:
        """Start tracking a new source."""
        progress = IngestionProgress(
            source_name=source_name,
            total_documents=total_documents,
            processed=0,
            successful=0,
            failed=0,
        )
        self.sources[source_name] = progress
        return progress

    def update(
        self,
        source_name: str,
        processed: Optional[int] = None,
        successful: Optional[int] = None,
        failed: Optional[int] = None,
        skipped: Optional[int] = None,
        current_file: Optional[str] = None,
        total_documents: Optional[int] = None,
    ) -> Optional[IngestionProgress]:
        """Update progress for a source."""
        if source_name not in self.sources:
            return None

        progress = self.sources[source_name]

        if processed is not None:
            progress.processed = processed
        if successful is not None:
            progress.successful = successful
        if failed is not None:
            progress.failed = failed
        if skipped is not None:
            progress.skipped = skipped
        if current_file is not None:
            progress.current_file = current_file
        if total_documents is not None:
            progress.total_documents = total_documents

        # Fire callback
        if self.callback:
            self.callback(progress)

        return progress

    def increment(
        self,
        source_name: str,
        success: bool = True,
        current_file: Optional[str] = None,
    ) -> Optional[IngestionProgress]:
        """Increment progress for a source."""
        if source_name not in self.sources:
            return None

        progress = self.sources[source_name]
        progress.processed += 1

        if success:
            progress.successful += 1
        else:
            progress.failed += 1

        if current_file is not None:
            progress.current_file = current_file

        # Fire callback
        if self.callback:
            self.callback(progress)

        return progress

    def get_progress(self, source_name: str) -> Optional[IngestionProgress]:
        """Get progress for a specific source."""
        return self.sources.get(source_name)

    @property
    def total_processed(self) -> int:
        """Total documents processed across all sources."""
        return sum(p.processed for p in self.sources.values())

    @property
    def total_successful(self) -> int:
        """Total successful ingestions across all sources."""
        return sum(p.successful for p in self.sources.values())

    @property
    def total_failed(self) -> int:
        """Total failed ingestions across all sources."""
        return sum(p.failed for p in self.sources.values())

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time since start."""
        return (datetime.utcnow() - self.started_at).total_seconds()

    def get_summary(self) -> Dict:
        """Get summary of all sources."""
        return {
            "sources": {
                name: {
                    "processed": p.processed,
                    "successful": p.successful,
                    "failed": p.failed,
                    "skipped": p.skipped,
                    "percent_complete": p.percent_complete,
                }
                for name, p in self.sources.items()
            },
            "total_processed": self.total_processed,
            "total_successful": self.total_successful,
            "total_failed": self.total_failed,
            "elapsed_seconds": self.elapsed_seconds,
        }
