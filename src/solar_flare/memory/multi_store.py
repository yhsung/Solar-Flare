"""
Multi-store RAG system for managing separate vector stores by document type.

This module provides a unified interface for managing multiple vector stores,
each optimized for different document types (standards, internal docs, working materials).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from solar_flare.memory.vector_store import (
    VectorStoreConfig,
    StandardDocument,
    StandardsVectorStore,
)


class StoreConfig(BaseModel):
    """Configuration for a single vector store."""

    name: str = Field(description="Unique store identifier")
    document_types: List[str] = Field(
        default_factory=list,
        description="Types of documents this store holds",
    )
    persist_directory: str = Field(
        description="Directory to persist this vector store",
    )
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Chunk size for document splitting",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    use_local_embeddings: bool = Field(
        default=False,
        description="Use local HuggingFace embeddings",
    )
    store_type: Literal["faiss", "chroma"] = Field(
        default="faiss",
        description="Vector store backend type",
    )


class MultiStoreConfig(BaseModel):
    """Configuration for multiple vector stores."""

    base_directory: str = Field(
        default="./data/vector_stores",
        description="Base directory for all stores",
    )
    stores: Dict[str, StoreConfig] = Field(
        default_factory=dict,
        description="Named store configurations",
    )

    def model_post_init(self, __context: Any) -> None:
        """Ensure base directory exists."""
        Path(self.base_directory).mkdir(parents=True, exist_ok=True)


# Default store configurations for Solar-Flare
DEFAULT_STORES: Dict[str, StoreConfig] = {
    "standards": StoreConfig(
        name="standards",
        document_types=["ISO 26262", "ASPICE", "GB/T 34590"],
        persist_directory="./data/vector_stores/standards",
        chunk_size=500,
        chunk_overlap=50,
    ),
    "internal": StoreConfig(
        name="internal",
        document_types=["design_spec", "architecture", "api", "hardware"],
        persist_directory="./data/vector_stores/internal",
        chunk_size=600,
        chunk_overlap=100,
    ),
    "working": StoreConfig(
        name="working",
        document_types=[
            "meeting_notes",
            "email_thread",
            "discussion_notes",
            "design_draft",
            "review_notes",
        ],
        persist_directory="./data/vector_stores/working",
        chunk_size=400,
        chunk_overlap=50,
    ),
}


class MultiStoreRAG:
    """
    Unified interface to multiple vector stores.

    Manages separate vector stores for different document types,
    each with optimized chunking and embedding strategies.

    Attributes:
        config: Multi-store configuration
        _stores: Internal dict of store name to StandardsVectorStore
    """

    def __init__(self, config: Optional[MultiStoreConfig] = None):
        """
        Initialize the multi-store RAG system.

        Args:
            config: Multi-store configuration. Defaults to DEFAULT_STORES.
        """
        self.config = config or MultiStoreConfig(stores=DEFAULT_STORES.copy())
        self._stores: Dict[str, StandardsVectorStore] = {}
        self._initialize_stores()

    def _initialize_stores(self) -> None:
        """Initialize each store using StandardsVectorStore."""
        for name, store_config in self.config.stores.items():
            # Update persist directory to be relative to base
            full_path = str(
                Path(self.config.base_directory) / Path(store_config.persist_directory).name
            )

            vector_config = VectorStoreConfig(
                store_type=store_config.store_type,
                persist_directory=full_path,
                embedding_model=store_config.embedding_model,
                use_local_embeddings=store_config.use_local_embeddings,
                chunk_size=store_config.chunk_size,
                chunk_overlap=store_config.chunk_overlap,
            )

            self._stores[name] = StandardsVectorStore(vector_config)

    def add_document(
        self,
        store_name: str,
        document: StandardDocument,
    ) -> None:
        """
        Add a document to a specific store.

        Args:
            store_name: Name of the target store
            document: Standard document to add

        Raises:
            KeyError: If store_name not found
        """
        if store_name not in self._stores:
            raise KeyError(
                f"Store '{store_name}' not found. Available: {list(self._stores.keys())}"
            )
        self._stores[store_name].add_documents([document])

    def add_text(
        self,
        store_name: str,
        text: str,
        title: str,
        standard: str = "custom",
        **metadata,
    ) -> None:
        """
        Add text to a specific store with automatic chunking.

        Args:
            store_name: Name of the target store
            text: Text content to add
            title: Document title
            standard: Standard identifier
            **metadata: Additional metadata fields
        """
        doc = StandardDocument(
            title=title,
            standard=standard,
            content=text,
            metadata=metadata,
        )
        self.add_document(store_name, doc)

    def search_all(
        self,
        query: str,
        top_k: int = 3,
        stores: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across specified stores, return results grouped by store.

        Args:
            query: Search query
            top_k: Number of results per store
            stores: List of store names to search. None = all stores

        Returns:
            Dict mapping store name to list of results
        """
        target_stores = stores or list(self._stores.keys())
        results: Dict[str, List[Dict[str, Any]]] = {}

        for store_name in target_stores:
            if store_name in self._stores:
                store_results = self._stores[store_name].search(query, top_k=top_k)
                results[store_name] = store_results

        return results

    def search_unified(
        self,
        query: str,
        top_k: int = 5,
        stores: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search across stores and return merged, ranked results.

        Args:
            query: Search query
            top_k: Total number of results to return
            stores: List of store names to search. None = all stores

        Returns:
            List of merged results from all stores
        """
        all_results: List[Dict[str, Any]] = []
        target_stores = stores or list(self._stores.keys())

        for store_name in target_stores:
            if store_name in self._stores:
                store_results = self._stores[store_name].search(query, top_k=top_k)
                # Add source store to each result
                for result in store_results:
                    result["source_store"] = store_name
                all_results.extend(store_results)

        # Sort by score if available, otherwise maintain order
        if all_results and "score" in all_results[0]:
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return all_results[:top_k]

    def retrieve_for_context(
        self,
        query: str,
        agent_type: Optional[str] = None,
        asil_level: Optional[str] = None,
        include_working_materials: bool = True,
    ) -> str:
        """
        Build unified context from all relevant sources.

        Args:
            query: Search query
            agent_type: Type of agent requesting context
            asil_level: ASIL level for filtering
            include_working_materials: Whether to include working materials

        Returns:
            Formatted context string for agent prompts
        """
        # Determine which stores to query
        stores_to_search = ["standards", "internal"]
        if include_working_materials:
            stores_to_search.append("working")

        results = self.search_all(query, top_k=3, stores=stores_to_search)

        context_parts: List[str] = []

        # Group results by store type
        for store_name in stores_to_search:
            if store_name in results and results[store_name]:
                context_parts.append(f"## {store_name.upper()} CONTEXT\n")
                for i, result in enumerate(results[store_name], 1):
                    title = result.get("title", "Unknown")
                    content = result.get("content", "")[:500]
                    context_parts.append(f"{i}. {title}\n   {content}...\n")

        # Add ASIL filter context if provided
        if asil_level:
            context_parts.insert(0, f"Querying for ASIL level: {asil_level}\n")

        return "\n".join(context_parts) if context_parts else "No relevant context found."

    def save_all(self) -> None:
        """Persist all stores to disk."""
        for store in self._stores.values():
            store.save()

    def load_all(self) -> None:
        """Load all stores from disk."""
        for store in self._stores.values():
            store.load()

    def get_store(self, store_name: str) -> StandardsVectorStore:
        """
        Get a specific store by name.

        Args:
            store_name: Name of the store

        Returns:
            The requested StandardsVectorStore

        Raises:
            KeyError: If store_name not found
        """
        if store_name not in self._stores:
            raise KeyError(
                f"Store '{store_name}' not found. Available: {list(self._stores.keys())}"
            )
        return self._stores[store_name]

    def list_stores(self) -> List[str]:
        """Return list of available store names."""
        return list(self._stores.keys())

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all stores.

        Returns:
            Dict mapping store name to its statistics
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for name, store in self._stores.items():
            stats[name] = store.get_statistics()
        return stats


def create_multi_store_rag(
    base_directory: str = "./data/vector_stores",
    embedding_model: str = "text-embedding-3-small",
    use_local_embeddings: bool = False,
) -> MultiStoreRAG:
    """
    Create a MultiStoreRAG with default configuration.

    Creates three stores optimized for Solar-Flare:
    - standards: ISO 26262, ASPICE, GB/T 34590 (chunk=500, overlap=50)
    - internal: Design specs, APIs, hardware (chunk=600, overlap=100)
    - working: Meetings, emails, discussions (chunk=400, overlap=50)

    Args:
        base_directory: Base directory for all stores
        embedding_model: Embedding model to use
        use_local_embeddings: Whether to use local HuggingFace embeddings

    Returns:
        Configured MultiStoreRAG instance
    """
    # Create store configs with updated settings
    stores: Dict[str, StoreConfig] = {}

    for name, default_config in DEFAULT_STORES.items():
        # Get the default config dict and update with custom settings
        config_dict = default_config.model_dump()
        config_dict.update({
            "embedding_model": embedding_model,
            "use_local_embeddings": use_local_embeddings,
        })
        stores[name] = StoreConfig(**config_dict)

    config = MultiStoreConfig(base_directory=base_directory, stores=stores)
    return MultiStoreRAG(config)
