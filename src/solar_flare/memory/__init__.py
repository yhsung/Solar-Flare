"""
Memory and persistence layer for Solar-Flare.

This module provides:
- Conversation memory management
- State checkpointing for workflows
- Vector store for RAG on ISO 26262/ASPICE standards
- Multi-store RAG system for different document types
"""

from solar_flare.memory.conversation import (
    ConversationSession,
    ConversationMemory,
)
from solar_flare.memory.checkpointer import (
    CheckpointConfig,
    SolarFlareCheckpointer,
    CheckpointMetadata,
    FileCheckpointSaver,
    CheckpointInspector,
    create_checkpointer,
)
from solar_flare.memory.vector_store import (
    VectorStoreConfig,
    StandardDocument,
    StandardsVectorStore,
    EmbeddedStandardsProvider,
    create_default_vector_store,
)
from solar_flare.memory.multi_store import (
    StoreConfig,
    MultiStoreConfig,
    MultiStoreRAG,
    DEFAULT_STORES,
    create_multi_store_rag,
)
from solar_flare.memory.ingestor import (
    WorkingMaterialsIngestor,
    IngestionResult,
)

__all__ = [
    # Conversation
    "ConversationSession",
    "ConversationMemory",
    # Checkpointing
    "CheckpointConfig",
    "SolarFlareCheckpointer",
    "CheckpointMetadata",
    "FileCheckpointSaver",
    "CheckpointInspector",
    "create_checkpointer",
    # Vector Store
    "VectorStoreConfig",
    "StandardDocument",
    "StandardsVectorStore",
    "EmbeddedStandardsProvider",
    "create_default_vector_store",
    # Multi-Store RAG
    "StoreConfig",
    "MultiStoreConfig",
    "MultiStoreRAG",
    "DEFAULT_STORES",
    "create_multi_store_rag",
    # Ingestion
    "WorkingMaterialsIngestor",
    "IngestionResult",
]
