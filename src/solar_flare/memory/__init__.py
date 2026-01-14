"""
Memory and persistence layer for Solar-Flare.

This module provides:
- Conversation memory management
- State checkpointing for workflows
- Vector store for RAG on ISO 26262/ASPICE standards
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
]
