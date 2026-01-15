"""
Document version tracking using SQLite.
"""
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json


@dataclass
class DocumentVersion:
    """Represents a version of a document."""
    id: Optional[int]
    source_path: str
    content_hash: str
    version_number: int
    ingested_at: datetime
    metadata: Dict[str, Any]
    vector_store: str
    is_current: bool = True


class VersionStore:
    """SQLite-based version tracking for ingested documents."""

    def __init__(self, db_path: Path):
        """
        Initialize version store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    ingested_at TIMESTAMP NOT NULL,
                    metadata TEXT,
                    vector_store TEXT NOT NULL,
                    is_current BOOLEAN DEFAULT 1,
                    UNIQUE(source_path, version_number)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_path
                ON document_versions(source_path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON document_versions(content_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ingested_at
                ON document_versions(ingested_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_current
                ON document_versions(is_current)
            """)

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_latest_version(self, source_path: str) -> Optional[DocumentVersion]:
        """Get the latest version of a document."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE source_path = ? AND is_current = 1
                ORDER BY version_number DESC LIMIT 1
                """,
                (source_path,)
            ).fetchone()

            if row:
                return self._row_to_version(row)
            return None

    def get_all_versions(self, source_path: str) -> List[DocumentVersion]:
        """Get all versions of a document."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE source_path = ?
                ORDER BY version_number DESC
                """,
                (source_path,)
            ).fetchall()

            return [self._row_to_version(row) for row in rows]

    def add_version(
        self,
        source_path: str,
        content_hash: str,
        vector_store: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentVersion:
        """
        Add a new version of a document.

        Args:
            source_path: Unique identifier/path for the document
            content_hash: SHA-256 hash of the content
            vector_store: Name of the target vector store
            metadata: Optional metadata dictionary

        Returns:
            The created DocumentVersion
        """
        with self._get_connection() as conn:
            # Get next version number
            row = conn.execute(
                """
                SELECT MAX(version_number) as max_ver
                FROM document_versions
                WHERE source_path = ?
                """,
                (source_path,)
            ).fetchone()

            next_version = (row["max_ver"] or 0) + 1

            # Mark previous versions as not current
            conn.execute(
                """
                UPDATE document_versions
                SET is_current = 0
                WHERE source_path = ?
                """,
                (source_path,)
            )

            # Insert new version
            now = datetime.utcnow()
            metadata_json = json.dumps(metadata) if metadata else None

            cursor = conn.execute(
                """
                INSERT INTO document_versions
                (source_path, content_hash, version_number, ingested_at,
                 metadata, vector_store, is_current)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (source_path, content_hash, next_version, now,
                 metadata_json, vector_store)
            )

            return DocumentVersion(
                id=cursor.lastrowid,
                source_path=source_path,
                content_hash=content_hash,
                version_number=next_version,
                ingested_at=now,
                metadata=metadata or {},
                vector_store=vector_store,
                is_current=True,
            )

    def has_changed(self, source_path: str, content_hash: str) -> bool:
        """
        Check if document content has changed since last ingestion.

        Args:
            source_path: Document path/identifier
            content_hash: Current content hash

        Returns:
            True if content is new or changed, False if unchanged
        """
        latest = self.get_latest_version(source_path)
        if not latest:
            return True  # New document
        return latest.content_hash != content_hash

    def get_documents_by_date_range(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        current_only: bool = True,
    ) -> List[DocumentVersion]:
        """
        Get all document versions ingested within a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range (default: now)
            current_only: Only return current versions

        Returns:
            List of DocumentVersion objects
        """
        end_date = end_date or datetime.utcnow()

        with self._get_connection() as conn:
            if current_only:
                rows = conn.execute(
                    """
                    SELECT * FROM document_versions
                    WHERE ingested_at BETWEEN ? AND ?
                    AND is_current = 1
                    ORDER BY ingested_at DESC
                    """,
                    (start_date, end_date)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM document_versions
                    WHERE ingested_at BETWEEN ? AND ?
                    ORDER BY ingested_at DESC
                    """,
                    (start_date, end_date)
                ).fetchall()

            return [self._row_to_version(row) for row in rows]

    def get_documents_by_store(
        self,
        vector_store: str,
        current_only: bool = True,
    ) -> List[DocumentVersion]:
        """
        Get all documents in a specific vector store.

        Args:
            vector_store: Name of the vector store
            current_only: Only return current versions

        Returns:
            List of DocumentVersion objects
        """
        with self._get_connection() as conn:
            if current_only:
                rows = conn.execute(
                    """
                    SELECT * FROM document_versions
                    WHERE vector_store = ? AND is_current = 1
                    ORDER BY ingested_at DESC
                    """,
                    (vector_store,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM document_versions
                    WHERE vector_store = ?
                    ORDER BY ingested_at DESC
                    """,
                    (vector_store,)
                ).fetchall()

            return [self._row_to_version(row) for row in rows]

    def delete_document(self, source_path: str) -> int:
        """
        Delete all versions of a document.

        Args:
            source_path: Document path/identifier

        Returns:
            Number of versions deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM document_versions
                WHERE source_path = ?
                """,
                (source_path,)
            )
            return cursor.rowcount

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the version store."""
        with self._get_connection() as conn:
            total_versions = conn.execute(
                "SELECT COUNT(*) as count FROM document_versions"
            ).fetchone()["count"]

            unique_documents = conn.execute(
                "SELECT COUNT(DISTINCT source_path) as count FROM document_versions"
            ).fetchone()["count"]

            current_documents = conn.execute(
                "SELECT COUNT(*) as count FROM document_versions WHERE is_current = 1"
            ).fetchone()["count"]

            by_store = {}
            rows = conn.execute(
                """
                SELECT vector_store, COUNT(*) as count
                FROM document_versions
                WHERE is_current = 1
                GROUP BY vector_store
                """
            ).fetchall()
            for row in rows:
                by_store[row["vector_store"]] = row["count"]

            return {
                "total_versions": total_versions,
                "unique_documents": unique_documents,
                "current_documents": current_documents,
                "by_store": by_store,
            }

    def _row_to_version(self, row: sqlite3.Row) -> DocumentVersion:
        """Convert database row to DocumentVersion."""
        ingested_at = row["ingested_at"]
        if isinstance(ingested_at, str):
            ingested_at = datetime.fromisoformat(ingested_at)

        return DocumentVersion(
            id=row["id"],
            source_path=row["source_path"],
            content_hash=row["content_hash"],
            version_number=row["version_number"],
            ingested_at=ingested_at,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            vector_store=row["vector_store"],
            is_current=bool(row["is_current"]),
        )
