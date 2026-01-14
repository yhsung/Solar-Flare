# MultiStoreRAG Implementation Plan

## Overview

Implement a multi-store RAG system for Solar-Flare that manages separate vector stores for different document types (standards, internal docs, working materials) while providing a unified query interface.

## Motivation

Different document types have different characteristics:
- **Standards** (ISO 26262, ASPICE): Change rarely, need larger chunks, formal structure
- **Internal Docs** (design specs, APIs): Change occasionally, need medium chunks
- **Working Materials** (meetings, emails, discussions): Change frequently, need smaller chunks for conversational content

Separate stores enable:
- Different chunking strategies per document type
- Independent re-indexing when documents change
- Store-specific metadata schemas
- Different embedding models if needed

## Current Architecture

- `StandardsVectorStore`: Single vector store in `src/solar_flare/memory/vector_store.py`
- `VectorStoreConfig`: Pydantic configuration model
- `StandardDocument`: Document model with title, standard, part, version, content, metadata
- `AgentRegistry`: Pattern for registering and creating agents
- `BaseWorkerAgent`: Abstract base for all agents with `llm`, `tools`, `agent_name`, `hardware_constraints`

## Implementation Plan

### 1. Create Multi-Store Configuration

**File**: `src/solar_flare/memory/multi_store.py`

**Classes**:
```python
class StoreConfig(BaseModel):
    """Configuration for a single vector store."""
    name: str                           # Store identifier
    document_types: List[str]           # Types of documents this store holds
    persist_directory: str              # Where to store this index
    chunk_size: int = 500               # Chunk size for this store
    chunk_overlap: int = 50             # Overlap for this store
    embedding_model: str = "text-embedding-3-small"
    use_local_embeddings: bool = False

class MultiStoreConfig(BaseModel):
    """Configuration for multiple vector stores."""
    base_directory: str = "./data/vector_stores"
    stores: Dict[str, StoreConfig] = {}  # Named store configurations
```

### 2. Create MultiStoreRAG Class

**File**: `src/solar_flare/memory/multi_store.py`

**Key Methods**:
```python
class MultiStoreRAG:
    """Unified interface to multiple vector stores."""

    def __init__(self, config: MultiStoreConfig):
        # Initialize each store with its config
        # Store references in self._stores dict

    def search_all(
        self, query: str, top_k: int = 3,
        stores: Optional[List[str]] = None
    ) -> Dict[str, List[Dict]]:
        """Search across specified stores, return results grouped by store."""

    def search_unified(
        self, query: str, top_k: int = 5,
        stores: Optional[List[str]] = None
    ) -> List[Dict]:
        """Search across stores and return merged, ranked results."""

    def retrieve_for_context(
        self, query: str, agent_type: str,
        asil_level: Optional[str] = None,
        include_working_materials: bool = True
    ) -> str:
        """Build unified context from all relevant sources."""

    def ingest_working_materials(
        self, directory: Path,
        doc_types: Dict[str, str] = None
    ) -> int:
        """Ingest working materials from directory structure."""
```

### 3. Create Working Materials Ingestor

**File**: `src/solar_flare/memory/ingestor.py`

**Purpose**: Batch ingestion of working materials with automatic type detection

```python
class WorkingMaterialsIngestor:
    """Ingest working materials from filesystem."""

    # Directory structure:
    # working_materials/
    # ├── meetings/          -> meeting_notes
    # ├── emails/            -> email_thread
    # ├── discussions/       -> discussion_notes
    # └── drafts/            -> design_draft

    def ingest_directory(self, root_dir: Path) -> int:
        """Scan directory and ingest all found documents."""

    def _detect_document_type(self, path: Path) -> str:
        """Detect document type from path and content."""

    def _extract_metadata(self, path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from file (dates, attendees, etc.)."""
```

### 4. Update Memory Module Exports

**File**: `src/solar_flare/memory/__init__.py`

**Add exports**:
```python
from solar_flare.memory.multi_store import (
    MultiStoreRAG,
    MultiStoreConfig,
    StoreConfig,
    create_multi_store_rag,
)
```

### 5. Create Factory Function

**File**: `src/solar_flare/memory/multi_store.py`

```python
def create_multi_store_rag(
    base_directory: str = "./data/vector_stores",
    use_local_embeddings: bool = False,
) -> MultiStoreRAG:
    """
    Create a MultiStoreRAG with default configuration.

    Stores created:
    - standards: ISO 26262, ASPICE, GB/T 34590 (large chunks)
    - internal: Design specs, architecture docs (medium chunks)
    - working: Meeting notes, emails, discussions (small chunks)
    """
```

### 6. Integration with Agents (Future)

**Not in initial implementation**, but design for future:

```python
# Option A: Pass to agent init
agent = ISO26262AnalyzerAgent(
    llm=llm,
    tools=tools,
    agent_name="iso_26262_analyzer",
    hardware_constraints=constraints,
    vector_store=multi_store  # NEW parameter
)

# Option B: Global registry pattern (similar to AgentRegistry)
VectorStoreRegistry.register("default", multi_store)
# Agents access via: VectorStoreRegistry.get("default")
```

## File Structure

```
src/solar_flare/memory/
├── __init__.py              # Update exports
├── vector_store.py          # Existing, no changes
├── multi_store.py           # NEW: MultiStoreRAG, StoreConfig, factory
└── ingestor.py              # NEW: WorkingMaterialsIngestor

examples/
└── multi_store_usage.py      # NEW: Usage examples

tests/unit/
└── test_multi_store.py      # NEW: Unit tests
```

## Default Configuration

```python
DEFAULT_STORES = {
    "standards": StoreConfig(
        name="standards",
        document_types=["ISO 26262", "ASPICE", "GB/T 34590"],
        persist_directory="./data/vector_stores/standards",
        chunk_size=500,
        chunk_overlap=50,
    ),
    "internal": StoreConfig(
        name="internal",
        document_types=["design_spec", "architecture", "api"],
        persist_directory="./data/vector_stores/internal",
        chunk_size=600,
        chunk_overlap=100,
    ),
    "working": StoreConfig(
        name="working",
        document_types=["meeting_notes", "email_thread", "discussion_notes", "design_draft"],
        persist_directory="./data/vector_stores/working",
        chunk_size=400,
        chunk_overlap=50,
    ),
}
```

## Testing Strategy

1. **Unit Tests** (`tests/unit/test_multi_store.py`):
   - Test `MultiStoreRAG` initialization
   - Test `search_all()` returns results from correct stores
   - Test `search_unified()` merges and ranks properly
   - Test `retrieve_for_context()` formatting
   - Test `WorkingMaterialsIngestor` file detection

2. **Integration Tests**:
   - Test ingestion from actual directory structure
   - Test cross-store queries
   - Test persistence (save/load each store)

## Example Usage

```python
from solar_flare.memory import create_multi_store_rag
from pathlib import Path

# Create multi-store system
rag = create_multi_store_rag()

# Ingest working materials
rag.ingest_working_materials(Path("docs/working_materials"))

# Search across specific stores
results = rag.search_all(
    "overflow policy decision",
    stores=["working"],
    top_k=5
)

# Cross-reference standards with internal docs
combined = rag.search_all(
    "ASIL-D ring buffer requirements",
    stores=["standards", "internal"],
    top_k=3
)

# Get unified context for agents
context = rag.retrieve_for_context(
    query="ring buffer overflow handling",
    agent_type="embedded_designer",
    asil_level="ASIL_D"
)
```

## Implementation Phases

### Phase 1: Core Configuration Models
**File**: `src/solar_flare/memory/multi_store.py`

Create configuration classes that extend the existing `VectorStoreConfig` pattern:

```python
class StoreConfig(BaseModel):
    """Configuration for a single vector store."""
    name: str
    document_types: List[str]
    persist_directory: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-small"
    use_local_embeddings: bool = False
    store_type: Literal["faiss", "chroma"] = "faiss"

class MultiStoreConfig(BaseModel):
    """Configuration for multiple vector stores."""
    base_directory: str = "./data/vector_stores"
    stores: Dict[str, StoreConfig] = {}
```

### Phase 2: MultiStoreRAG Class
**File**: `src/solar_flare/memory/multi_store.py`

Key implementation details:
- Reuses existing `StandardsVectorStore` for each store
- Manages `Dict[str, StandardsVectorStore]` internally
- Provides unified search across stores

```python
class MultiStoreRAG:
    def __init__(self, config: MultiStoreConfig):
        self.config = config
        self._stores: Dict[str, StandardsVectorStore] = {}
        self._initialize_stores()

    def _initialize_stores(self):
        """Initialize each store using StandardsVectorStore."""
        for name, store_config in self.config.stores.items():
            vector_config = VectorStoreConfig(
                store_type=store_config.store_type,
                persist_directory=store_config.persist_directory,
                embedding_model=store_config.embedding_model,
                use_local_embeddings=store_config.use_local_embeddings,
                chunk_size=store_config.chunk_size,
                chunk_overlap=store_config.chunk_overlap,
            )
            # Load existing or create new
            self._stores[name] = StandardsVectorStore(vector_config)

    def add_document(self, store_name: str, document: StandardDocument):
        """Add document to specific store."""

    def add_text(self, store_name: str, text: str, **kwargs):
        """Add text to specific store with chunking."""

    def search_all(self, query: str, top_k: int = 3,
                   stores: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Search across stores, return grouped results."""

    def search_unified(self, query: str, top_k: int = 5,
                      stores: Optional[List[str]] = None) -> List[Dict]:
        """Search across stores, return merged ranked results."""

    def retrieve_for_context(self, query: str, agent_type: str,
                            asil_level: Optional[str] = None) -> str:
        """Build unified context for agent prompts."""

    def save_all(self):
        """Persist all stores."""

    def load_all(self):
        """Load all stores from disk."""
```

### Phase 3: Working Materials Ingestor
**File**: `src/solar_flare/memory/ingestor.py`

Directory structure expected:
```
working_materials/
├── meetings/          -> document_type: "meeting_notes"
├── emails/            -> document_type: "email_thread"
├── discussions/       -> document_type: "discussion_notes"
├── drafts/            -> document_type: "design_draft"
└── reviews/           -> document_type: "review_notes"
```

```python
class WorkingMaterialsIngestor:
    def __init__(self, multi_store: MultiStoreRAG):
        self.multi_store = multi_store
        self.dir_type_map = {
            "meetings": "meeting_notes",
            "emails": "email_thread",
            "discussions": "discussion_notes",
            "drafts": "design_draft",
            "reviews": "review_notes",
        }

    def ingest_directory(self, root_dir: Path) -> Dict[str, int]:
        """Scan and ingest, return counts by type."""

    def _detect_document_type(self, path: Path) -> str:
        """Detect type from directory name."""

    def _extract_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract date from filename, attendees from content, etc."""
```

### Phase 4: Factory Function
**File**: `src/solar_flare/memory/multi_store.py`

```python
def create_multi_store_rag(
    base_directory: str = "./data/vector_stores",
    embedding_model: str = "text-embedding-3-small",
    use_local_embeddings: bool = False,
) -> MultiStoreRAG:
    """
    Create MultiStoreRAG with defaults optimized for Solar-Flare.

    Stores:
    - standards: ISO 26262, ASPICE, GB/T 34590 (chunk=500, overlap=50)
    - internal: Design specs, APIs, hardware (chunk=600, overlap=100)
    - working: Meetings, emails, discussions (chunk=400, overlap=50)
    """
```

### Phase 5: Update Module Exports
**File**: `src/solar_flare/memory/__init__.py`

Add new exports while keeping existing ones.

### Phase 6: Tests and Examples

**Unit tests**: `tests/unit/test_multi_store.py`
**Example script**: `examples/multi_store_usage.py`

## Critical Files Reference

| File | Purpose |
|------|---------|
| `src/solar_flare/memory/vector_store.py:1` | Existing `StandardsVectorStore` to reuse |
| `src/solar_flare/memory/vector_store.py:45` | `VectorStoreConfig` pattern to follow |
| `src/solar_flare/memory/vector_store.py:85` | `StandardDocument` model for docs |
| `src/solar_flare/agents/base.py:164` | `AgentRegistry` pattern for reference |
| `src/solar_flare/memory/__init__.py:1` | Module exports to update |

## Verification Checklist

- [ ] `StoreConfig` and `MultiStoreConfig` models created
- [ ] `MultiStoreRAG` class with `_initialize_stores()` using existing `StandardsVectorStore`
- [ ] `search_all()` returns Dict[str, List[Dict]] grouped by store
- [ ] `search_unified()` merges and re-ranks results across stores
- [ ] `retrieve_for_context()` formats output for agent prompts
- [ ] `WorkingMaterialsIngestor` with directory scanning
- [ ] `create_multi_store_rag()` factory with default configurations
- [ ] `__init__.py` exports updated
- [ ] Unit tests in `tests/unit/test_multi_store.py`
- [ ] Example script in `examples/multi_store_usage.py`
- [ ] RAG_GUIDE.md updated with multi-store section
