"""
Unified ingestion orchestrator for multi-source RAG.
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Type, Union
import logging

from solar_flare.memory.multi_store import MultiStoreRAG
from solar_flare.memory.vector_store import StandardDocument

from .config import (
    IngestionConfig,
    SourceConfig,
    LocalFSConfig,
    SMBConfig,
    ConfluenceConfig,
    SharePointConfig,
    SourceType,
)
from .loaders.base import BaseDocumentLoader, LoadedDocument, DocumentLoadError
from .loaders.pdf_loader import PDFLoader
from .loaders.docx_loader import DocxLoader
from .loaders.pptx_loader import PptxLoader
from .loaders.xlsx_loader import XlsxLoader
from .loaders.text_loader import MarkdownLoader, TextLoader
from .connectors.base import BaseSourceConnector, SourceDocument, ConnectorError
from .connectors.local_fs import LocalFSConnector
from .connectors.smb_connector import SMBConnector
from .connectors.confluence_connector import ConfluenceConnector
from .connectors.sharepoint_connector import SharePointConnector
from .versioning import VersionStore
from .progress import ProgressTracker, IngestionProgress, IngestionResult


logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """
    Orchestrates multi-source document ingestion.

    Coordinates source connectors, document loaders, version tracking,
    and MultiStoreRAG integration.
    """

    # Connector registry
    CONNECTORS: Dict[SourceType, Type[BaseSourceConnector]] = {
        SourceType.LOCAL_FS: LocalFSConnector,
        SourceType.SMB: SMBConnector,
        SourceType.CONFLUENCE: ConfluenceConnector,
        SourceType.SHAREPOINT: SharePointConnector,
    }

    # Loader registry by extension
    LOADERS: Dict[str, BaseDocumentLoader] = {}

    def __init__(
        self,
        config: IngestionConfig,
        multi_store: MultiStoreRAG,
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
    ):
        """
        Initialize the ingestion orchestrator.

        Args:
            config: Ingestion configuration
            multi_store: Target MultiStoreRAG instance
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.multi_store = multi_store
        self.progress_callback = progress_callback

        # Initialize version store
        self.version_store = VersionStore(config.versioning.db_path)

        # Initialize loaders
        self._init_loaders()

        # Progress tracker
        self.progress = ProgressTracker(callback=progress_callback)

    def _init_loaders(self) -> None:
        """Initialize document loaders."""
        loaders = [
            PDFLoader(),
            DocxLoader(),
            PptxLoader(),
            XlsxLoader(),
            MarkdownLoader(),
            TextLoader(),
        ]

        for loader in loaders:
            for ext in loader.SUPPORTED_EXTENSIONS:
                self.LOADERS[ext] = loader

    async def ingest_all(self) -> IngestionResult:
        """
        Ingest documents from all configured sources.

        Returns:
            IngestionResult with statistics
        """
        start_time = datetime.utcnow()
        result = IngestionResult()

        # Filter enabled sources
        enabled_sources = [s for s in self.config.sources if s.enabled]

        if not enabled_sources:
            logger.warning("No enabled sources configured")
            return result

        # Create semaphore for parallel source limiting
        semaphore = asyncio.Semaphore(self.config.parallel_sources)

        # Run sources in parallel with semaphore
        tasks = [
            self._ingest_source_with_semaphore(source, semaphore, result)
            for source in enabled_sources
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate duration
        result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        # Save all stores
        self.multi_store.save_all()

        return result

    async def ingest_source(self, source_name: str) -> IngestionResult:
        """
        Ingest documents from a specific source.

        Args:
            source_name: Name of the source to ingest

        Returns:
            IngestionResult with statistics
        """
        source = next(
            (s for s in self.config.sources if s.name == source_name),
            None
        )

        if not source:
            raise ValueError(f"Source not found: {source_name}")

        result = IngestionResult()
        start_time = datetime.utcnow()

        await self._ingest_source(source, result)

        result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        # Save all stores
        self.multi_store.save_all()

        return result

    async def _ingest_source_with_semaphore(
        self,
        source: SourceConfig,
        semaphore: asyncio.Semaphore,
        result: IngestionResult,
    ) -> None:
        """Ingest source with semaphore for parallel limiting."""
        async with semaphore:
            await self._ingest_source(source, result)

    async def _ingest_source(
        self,
        source: SourceConfig,
        result: IngestionResult,
    ) -> None:
        """Ingest all documents from a source."""
        connector_class = self.CONNECTORS.get(source.source_type)
        if not connector_class:
            error = f"No connector for source type: {source.source_type}"
            result.add_failure(error)
            logger.error(error)
            return

        # Create connector with the right config type
        connector = connector_class(source)

        # Start progress tracking
        self.progress.start_source(source.name)

        try:
            async with connector:
                # Collect documents in batches
                batch: List[tuple] = []

                async for source_doc in connector.list_documents():
                    result.total_documents += 1

                    try:
                        # Get loader for this file
                        ext = Path(source_doc.filename).suffix.lower()
                        loader = self.LOADERS.get(ext)

                        if not loader:
                            logger.debug(f"No loader for extension: {ext}")
                            continue

                        # Get document stream
                        stream = await connector.get_document_stream(source_doc.source_path)

                        # Load document
                        loaded_doc = loader.load_from_stream(
                            stream,
                            source_doc.filename,
                            source_doc.source_path,
                        )

                        # Check if changed (version tracking)
                        if self.config.versioning.enabled:
                            if not self.version_store.has_changed(
                                source_doc.source_path,
                                loaded_doc.content_hash,
                            ):
                                result.add_skipped()
                                self.progress.update(
                                    source.name,
                                    skipped=result.skipped_unchanged,
                                )
                                continue

                        # Add to batch
                        batch.append((source_doc, loaded_doc))

                        # Process batch if full
                        if len(batch) >= self.config.batch_size:
                            await self._process_batch(batch, source, result)
                            batch = []

                        # Update progress
                        self.progress.increment(
                            source.name,
                            success=True,
                            current_file=source_doc.filename,
                        )

                    except DocumentLoadError as e:
                        result.add_failure(str(e))
                        self.progress.increment(source.name, success=False)
                        logger.warning(f"Failed to load document: {e}")
                    except Exception as e:
                        result.add_failure(f"Unexpected error: {e}")
                        self.progress.increment(source.name, success=False)
                        logger.exception(f"Unexpected error processing {source_doc.source_path}")

                # Process remaining batch
                if batch:
                    await self._process_batch(batch, source, result)

        except ConnectorError as e:
            error = f"Connector error for {source.name}: {e}"
            result.add_failure(error)
            logger.error(error)

    async def _process_batch(
        self,
        batch: List[tuple],
        source: SourceConfig,
        result: IngestionResult,
    ) -> None:
        """Process a batch of documents."""
        for source_doc, loaded_doc in batch:
            try:
                # Build metadata
                metadata = {
                    **loaded_doc.metadata,
                    **source.metadata,
                    "source_name": source.name,
                    "source_type": source.source_type.value,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "content_hash": loaded_doc.content_hash,
                    "original_path": source_doc.source_path,
                    "modified_date": source_doc.modified_date.isoformat(),
                }

                # Create StandardDocument
                doc = StandardDocument(
                    title=loaded_doc.title,
                    standard=source.source_type.value,
                    content=loaded_doc.content,
                    metadata=metadata,
                )

                # Add to vector store
                self.multi_store.add_document(source.target_store, doc)

                # Track version
                if self.config.versioning.enabled:
                    self.version_store.add_version(
                        source_path=source_doc.source_path,
                        content_hash=loaded_doc.content_hash,
                        vector_store=source.target_store,
                        metadata=metadata,
                    )
                    result.versions_created += 1

                result.add_success(source.name, loaded_doc.file_format)

            except Exception as e:
                result.add_failure(f"Failed to ingest {source_doc.source_path}: {e}")
                logger.exception("Failed to ingest document")

    def get_version_history(self, source_path: str) -> List:
        """Get version history for a document."""
        return self.version_store.get_all_versions(source_path)

    def get_ingestion_stats(self) -> Dict:
        """Get overall ingestion statistics."""
        return self.version_store.get_statistics()


def create_ingestion_orchestrator(
    multi_store: MultiStoreRAG,
    sources: Optional[List[Union[LocalFSConfig, SMBConfig, ConfluenceConfig, SharePointConfig]]] = None,
    versioning_db_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
) -> IngestionOrchestrator:
    """
    Factory function to create an IngestionOrchestrator with common defaults.

    Args:
        multi_store: Target MultiStoreRAG instance
        sources: List of source configurations
        versioning_db_path: Path for version tracking database
        progress_callback: Optional progress callback

    Returns:
        Configured IngestionOrchestrator
    """
    from .config import IngestionConfig, VersioningConfig

    versioning_config = VersioningConfig(
        enabled=True,
        db_path=versioning_db_path or Path("./data/ingestion/versions.db"),
        keep_all_versions=True,
    )

    config = IngestionConfig(
        sources=sources or [],
        versioning=versioning_config,
    )

    return IngestionOrchestrator(
        config=config,
        multi_store=multi_store,
        progress_callback=progress_callback,
    )
