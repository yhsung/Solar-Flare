"""
Unit tests for MultiStoreRAG system.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

from solar_flare.memory.multi_store import (
    StoreConfig,
    MultiStoreConfig,
    MultiStoreRAG,
    DEFAULT_STORES,
    create_multi_store_rag,
)
from solar_flare.memory.vector_store import StandardDocument


class TestStoreConfig:
    """Tests for StoreConfig model."""

    def test_store_config_creation(self):
        """Test creating a StoreConfig."""
        config = StoreConfig(
            name="test_store",
            document_types=["test"],
            persist_directory="./test",
            chunk_size=600,
            chunk_overlap=100,
        )
        assert config.name == "test_store"
        assert config.document_types == ["test"]
        assert config.chunk_size == 600
        assert config.chunk_overlap == 100

    def test_store_config_defaults(self):
        """Test StoreConfig default values."""
        config = StoreConfig(name="test", persist_directory="./test")
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.embedding_model == "text-embedding-3-small"
        assert config.use_local_embeddings is False
        assert config.store_type == "faiss"


class TestMultiStoreConfig:
    """Tests for MultiStoreConfig model."""

    def test_multi_store_config_creation(self):
        """Test creating a MultiStoreConfig."""
        stores = {
            "standards": StoreConfig(
                name="standards",
                document_types=["ISO 26262"],
                persist_directory="./standards",
            )
        }
        config = MultiStoreConfig(base_directory="./test", stores=stores)
        assert config.base_directory == "./test"
        assert len(config.stores) == 1
        assert "standards" in config.stores

    def test_default_stores_structure(self):
        """Test DEFAULT_STORES has expected keys."""
        assert "standards" in DEFAULT_STORES
        assert "internal" in DEFAULT_STORES
        assert "working" in DEFAULT_STORES


class TestMultiStoreRAG:
    """Tests for MultiStoreRAG class."""

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_initialization(self, mock_store_class):
        """Test MultiStoreRAG initialization."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        config = MultiStoreConfig(stores=stores)

        rag = MultiStoreRAG(config)

        assert "test" in rag._stores
        assert rag.list_stores() == ["test"]

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_add_document(self, mock_store_class):
        """Test adding a document to a store."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        doc = StandardDocument(
            title="Test Doc",
            standard="TEST",
            content="Test content",
        )
        rag.add_document("test", doc)

        mock_store.add_documents.assert_called_once()

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_add_document_invalid_store(self, mock_store_class):
        """Test adding document to non-existent store raises error."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        doc = StandardDocument(
            title="Test Doc",
            standard="TEST",
            content="Test content",
        )

        with pytest.raises(KeyError, match="Store 'nonexistent' not found"):
            rag.add_document("nonexistent", doc)

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_add_text(self, mock_store_class):
        """Test adding text to a store."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        rag.add_text("test", "Test content", "Test Title", standard="CUSTOM")

        mock_store.add_documents.assert_called_once()

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_search_all(self, mock_store_class):
        """Test searching across all stores."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"title": "Result 1", "content": "Content 1"},
            {"title": "Result 2", "content": "Content 2"},
        ]
        mock_store_class.return_value = mock_store

        stores = {
            "standards": StoreConfig(
                name="standards",
                document_types=["ISO 26262"],
                persist_directory="./standards",
            ),
            "internal": StoreConfig(
                name="internal",
                document_types=["design"],
                persist_directory="./internal",
            ),
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        results = rag.search_all("test query", top_k=3)

        assert "standards" in results
        assert "internal" in results
        assert len(results["standards"]) == 2
        assert len(results["internal"]) == 2

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_search_all_specific_stores(self, mock_store_class):
        """Test searching only specific stores."""
        mock_store = MagicMock()
        mock_store.search.return_value = [{"title": "Result 1"}]
        mock_store_class.return_value = mock_store

        stores = {
            "standards": StoreConfig(
                name="standards",
                document_types=["ISO 26262"],
                persist_directory="./standards",
            ),
            "internal": StoreConfig(
                name="internal",
                document_types=["design"],
                persist_directory="./internal",
            ),
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        results = rag.search_all("test query", stores=["standards"])

        assert "standards" in results
        assert "internal" not in results

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_search_unified(self, mock_store_class):
        """Test unified search across stores."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"title": "Result 1", "score": 0.9},
            {"title": "Result 2", "score": 0.7},
        ]
        mock_store_class.return_value = mock_store

        stores = {
            "standards": StoreConfig(
                name="standards",
                document_types=["ISO 26262"],
                persist_directory="./standards",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        results = rag.search_unified("test query", top_k=5)

        assert len(results) <= 5
        assert all("source_store" in r for r in results)

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_retrieve_for_context(self, mock_store_class):
        """Test retrieving formatted context."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {"title": "Result 1", "content": "Content 1"},
        ]
        mock_store_class.return_value = mock_store

        stores = {
            "standards": StoreConfig(
                name="standards",
                document_types=["ISO 26262"],
                persist_directory="./standards",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        context = rag.retrieve_for_context(
            "test query",
            agent_type="test_agent",
            asil_level="ASIL_D",
        )

        assert "STANDARDS CONTEXT" in context
        assert "ASIL level: ASIL_D" in context

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_save_all(self, mock_store_class):
        """Test saving all stores."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        rag.save_all()

        mock_store.save.assert_called_once()

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_load_all(self, mock_store_class):
        """Test loading all stores."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        rag.load_all()

        mock_store.load.assert_called_once()

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_get_store(self, mock_store_class):
        """Test getting a specific store."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        store = rag.get_store("test")

        assert store is not None

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_get_store_invalid(self, mock_store_class):
        """Test getting non-existent store raises error."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        with pytest.raises(KeyError, match="Store 'invalid' not found"):
            rag.get_store("invalid")

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_get_statistics(self, mock_store_class):
        """Test getting statistics for all stores."""
        mock_store = MagicMock()
        mock_store.get_statistics.return_value = {"total_documents": 10}
        mock_store_class.return_value = mock_store

        stores = {
            "test": StoreConfig(
                name="test",
                document_types=["test"],
                persist_directory="./test",
            )
        }
        rag = MultiStoreRAG(MultiStoreConfig(stores=stores))

        stats = rag.get_statistics()

        assert "test" in stats
        assert stats["test"]["total_documents"] == 10


class TestCreateMultiStoreRag:
    """Tests for create_multi_store_rag factory function."""

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_factory_creates_all_default_stores(self, mock_store_class):
        """Test factory creates all three default stores."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            rag = create_multi_store_rag()

        assert "standards" in rag.list_stores()
        assert "internal" in rag.list_stores()
        assert "working" in rag.list_stores()

    @patch("solar_flare.memory.multi_store.StandardsVectorStore")
    def test_factory_with_custom_settings(self, mock_store_class):
        """Test factory with custom base directory and embedding model."""
        mock_store = MagicMock()
        mock_store_class.return_value = mock_store

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            rag = create_multi_store_rag(
                base_directory="./custom_stores",
                embedding_model="text-embedding-3-large",
            )

        assert rag.config.base_directory == "./custom_stores"
        # All stores should have the updated embedding model
        for store_config in rag.config.stores.values():
            assert store_config.embedding_model == "text-embedding-3-large"
