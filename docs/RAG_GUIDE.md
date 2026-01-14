# RAG (Retrieval-Augmented Generation) Guide

This guide explains how to prepare, ingest, and use automotive safety standards (ISO 26262, ASPICE) for retrieval-augmented generation in the Solar-Flare multi-agent system.

## Overview

Solar-Flare includes a vector store that enables RAG capabilities, allowing agents to retrieve relevant sections from ISO 26262 and ASPICE standards during design and analysis tasks.

### Benefits of RAG

- **Accurate Standards Citations**: Agents reference specific standard clauses
- **Up-to-date Information**: Easy to update with new standard versions
- **Reduced Hallucination**: Grounded responses in actual document content
- **Improved Compliance**: Better alignment with ISO 26262/ASPICE requirements

---

## Quick Start

### Using Pre-loaded Standards

Solar-Flare comes with sample ISO 26262 and ASPICE documents pre-loaded:

```python
from solar_flare.memory import create_default_vector_store

# Create vector store with embedded standards
store = create_default_vector_store()

# Search for relevant content
results = store.search(
    query="ASIL D diagnostic coverage requirements",
    top_k=3
)

for result in results:
    print(f"{result['title']}: {result['content'][:200]}...")
```

### Retrieving Context for Agents

```python
# Get RAG context for agent prompts
context = store.retrieve_for_context(
    query="ring buffer lock-free implementation",
    agent_type="embedded_designer",
    asil_level="ASIL_D"
)

# Use context in agent prompt
prompt = f"""
Design a lock-free ring buffer with the following standards context:

{context}

Please provide implementation details...
"""
```

---

## Document Preparation

### Supported Document Formats

| Format | Description |
|--------|-------------|
| **JSON** | Structured document import via `add_from_json()` |
| **Text Chunks** | Direct text via `add_text_chunks()` |
| **StandardDocument** | Programmatic document creation |

### Document Schema

```python
from solar_flare.memory import StandardDocument

doc = StandardDocument(
    title="ISO 26262-6: Software Level Requirements",
    standard="ISO 26262",
    part="6",                      # Optional: Part/section identifier
    version="2018",                # Optional: Document version
    content="Full document text...",  # The content to index
    metadata={                     # Additional metadata
        "keywords": ["software", "verification", "ASIL"],
        "domain": "functional_safety",
    }
)
```

### JSON Format for Bulk Import

Create a JSON file for bulk document import:

```json
{
  "version": "1.0",
  "documents": [
    {
      "title": "Part 1: Vocabulary",
      "standard": "ISO 26262",
      "part": "1",
      "version": "2018",
      "content": "ISO 26262 defines the following vocabulary...",
      "metadata": {
        "keywords": ["ASIL", "safety goal", "functional safety"]
      }
    },
    {
      "title": "SWE.1: Requirements Elicitation",
      "standard": "ASPICE",
      "part": "SWE.1",
      "version": "v3.1",
      "content": "SWE.1 Requirements Elicitation - Base Practices...",
      "metadata": {
        "process_area": "requirements",
        "capability_level": "performed"
      }
    }
  ]
}
```

Then import:

```python
from solar_flare.memory import StandardsVectorStore, VectorStoreConfig

config = VectorStoreConfig()
store = StandardsVectorStore(config)
store.add_from_json("path/to/documents.json")
store.save()
```

---

## Vector Store Configuration

### Configuration Options

```python
from solar_flare.memory import VectorStoreConfig

config = VectorStoreConfig(
    # Vector store backend
    store_type="faiss",  # or "chroma"

    # Persistence location
    persist_directory="./data/vector_store",

    # Embedding model
    embedding_model="text-embedding-3-small",  # OpenAI
    # or for local models:
    # embedding_model="sentence-transformers/all-MiniLM-L6-v2",

    # Use local embeddings (no API key required)
    use_local_embeddings=False,

    # Chunking settings
    chunk_size=500,
    chunk_overlap=50,

    # Retrieval settings
    top_k=3,
)
```

### Backend Comparison

| Backend | Pros | Cons | Use Case |
|---------|------|------|----------|
| **FAISS** | Fast, in-memory | Not persistent by default | Development, testing |
| **ChromaDB** | Persistent, scalable | Slower than FAISS | Production, large document sets |

### Embedding Models

| Model | Type | Size | Quality | API Required |
|-------|------|------|---------|--------------|
| `text-embedding-3-small` | OpenAI | Small | High | Yes |
| `text-embedding-3-large` | OpenAI | Large | Very High | Yes |
| `all-MiniLM-L6-v2` | Local | Small | Medium | No |
| `all-mpnet-base-v2` | Local | Medium | High | No |

---

## Ingestion Workflow

### Step 1: Prepare Documents

Organize your standards documents:

```
standards/
├── iso_26262/
│   ├── part1_vocabulary.txt
│   ├── part4_system_level.txt
│   ├── part5_hardware_level.txt
│   └── part6_software_level.txt
├── aspice/
│   ├── swe1_requirements.json
│   ├── swe2_architecture.json
│   └── swe3_detailed_design.json
└── hardware/
    └── constraints.json
```

### Step 2: Create Ingestion Script

```python
"""
Ingestion script for ISO 26262 and ASPICE standards.
"""
import asyncio
from pathlib import Path
from solar_flare.memory import (
    StandardsVectorStore,
    VectorStoreConfig,
    StandardDocument,
)

async def ingest_standards():
    # Configure vector store
    config = VectorStoreConfig(
        store_type="faiss",
        persist_directory="./data/vector_store",
        use_local_embeddings=False,
    )

    store = StandardsVectorStore(config)

    # Load ISO 26262 documents
    iso_dir = Path("standards/iso_26262")
    for file_path in iso_dir.glob("*.txt"):
        with open(file_path, "r") as f:
            content = f.read()

        # Extract metadata from filename
        part_num = file_path.stem.split("_")[1]

        store.add_text_chunks(
            text=content,
            title=f"ISO 26262-{part_num.upper()}",
            standard="ISO 26262",
            metadata={"part": part_num, "version": "2018"},
        )

    # Load ASPICE documents
    aspice_dir = Path("standards/aspice")
    for file_path in aspice_dir.glob("*.json"):
        store.add_from_json(str(file_path))

    # Save for persistence
    store.save()

    # Print statistics
    stats = store.get_statistics()
    print(f"Ingestion complete:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Store location: {stats['persist_directory']}")

if __name__ == "__main__":
    asyncio.run(ingest_standards())
```

### Step 3: Run Ingestion

```bash
python scripts/ingest_standards.py
```

---

## Advanced Usage

### Filtering by Standard

```python
# Search only ISO 26262 documents
results = store.search(
    query="safety mechanisms",
    filter_standard="ISO 26262"
)

# Search only ASPICE documents
aspice_results = store.search(
    query="requirements traceability",
    filter_standard="ASPICE"
)
```

### Retrieving for Specific Agents

```python
# ISO 26262 Analyzer context
iso_context = store.retrieve_for_context(
    query="ASIL D requirements",
    agent_type="iso_26262_analyzer",
    asil_level="ASIL_D"
)

# Embedded Designer context
design_context = store.retrieve_for_context(
    query="lock-free ring buffer",
    agent_type="embedded_designer",
)

# ASPICE Assessor context
aspice_context = store.retrieve_for_context(
    query="capability level 3 process",
    agent_type="aspice_assessor",
    capability_level=3
)
```

### Custom Chunking Strategies

```python
from solar_flare.memory import StandardsVectorStore, VectorStoreConfig

config = VectorStoreConfig(
    chunk_size=1000,      # Larger chunks for more context
    chunk_overlap=200,    # More overlap for continuity
)

store = StandardsVectorStore(config)

# Manual chunking for structured content
sections = [
    ("Section 1: Safety Goals", "..."),
    ("Section 2: Functional Safety", "..."),
    ("Section 3: Technical Safety", "..."),
]

for title, content in sections:
    store.add_text_chunks(
        text=content,
        title=title,
        standard="ISO 26262",
        metadata={"section": title},
    )
```

---

## Persistence and Loading

### Saving Vector Store

```python
store = create_default_vector_store()
store.save()  # Persists to configured directory
```

### Loading Existing Store

```python
from solar_flare.memory import StandardsVectorStore, VectorStoreConfig

config = VectorStoreConfig(
    persist_directory="./data/vector_store",
)

store = StandardsVectorStore(config)

# Load existing store
if store.load():
    print("Loaded existing vector store")
else:
    print("No existing store found")
```

### Store Statistics

```python
stats = store.get_statistics()
print(f"Backend: {stats['store_type']}")
print(f"Documents: {stats['total_documents']}")
print(f"Embeddings: {stats['embedding_model']}")
print(f"Local: {stats['use_local_embeddings']}")
```

---

## Integration with Agents

### Adding RAG to Agent Prompts

```python
from solar_flare.agents import BaseAgent
from solar_flare.memory import StandardsVectorStore

class RAGEnabledAgent(BaseAgent):
    def __init__(self, llm, vector_store: StandardsVectorStore, **kwargs):
        super().__init__(llm, **kwargs)
        self.vector_store = vector_store

    async def execute(self, context: dict, messages: list):
        # Retrieve relevant standards
        rag_context = self.vector_store.retrieve_for_context(
            query=messages[-1].content,
            agent_type=self.name,
            asil_level=context.get("asil_level"),
        )

        # Add RAG context to prompt
        enhanced_prompt = self._build_prompt(context) + "\n\n" + rag_context

        # Execute with enhanced prompt
        response = await self.llm.ainvoke(enhanced_prompt)
        return self._parse_response(response)
```

### Example: ISO 26262 Analyzer with RAG

```python
from solar_flare.memory import create_default_vector_store
from solar_flare.agents.iso_26262_analyzer import ISO26262AnalyzerAgent

# Create vector store
vector_store = create_default_vector_store()

# Create agent with RAG (custom implementation)
agent = ISO26262AnalyzerAgent(
    llm=llm,
    tools=[],
    hardware_constraints=constraints,
    vector_store=vector_store,  # Pass vector store
)

# Agent will use RAG context automatically
result = await agent.execute(
    context={"component": "ring_buffer", "asil_level": "ASIL_D"},
    messages=[HumanMessage("Analyze for ASIL-D compliance")],
)
```

---

## Best Practices

### Document Preparation

1. **Use Clean Text**: Remove headers, footers, page numbers
2. **Preserve Structure**: Keep section headings and numbering
3. **Include Metadata**: Add part numbers, versions, keywords
4. **Chunk Appropriately**: Balance between context size and precision

### Chunking Guidelines

| Document Type | Chunk Size | Overlap |
|---------------|------------|---------|
| Standards (ISO 26262) | 500-800 | 50-100 |
| Process (ASPICE) | 300-500 | 50 |
| Technical specs | 800-1000 | 100-200 |

### Retrieval Strategies

```python
# For specific questions (high precision)
results = store.search(query, top_k=1)

# For exploratory analysis (high recall)
results = store.search(query, top_k=5)

# For comprehensive review
results = store.search(query, top_k=10)
```

---

## Troubleshooting

### Empty Search Results

```python
results = store.search("my query")
if not results:
    # Check if documents are loaded
    stats = store.get_statistics()
    if stats['total_documents'] == 0:
        print("No documents in store. Run ingestion first.")
    else:
        print("Documents exist but no matches. Try different query terms.")
```

### Low Quality Retrieval

1. **Adjust chunk size**: Smaller chunks may be more precise
2. **Add metadata**: Filter by standard or part
3. **Rephrase query**: Use terms from the original documents
4. **Increase top_k**: Retrieve more candidates

### Memory Issues

```python
# For large document sets, use ChromaDB with disk persistence
config = VectorStoreConfig(
    store_type="chroma",
    persist_directory="./data/vector_store_chroma",
)

# Or use local embeddings to reduce memory footprint
config = VectorStoreConfig(
    use_local_embeddings=True,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
)
```

---

## API Reference

### `StandardsVectorStore`

```python
class StandardsVectorStore:
    def __init__(self, config: VectorStoreConfig)

    def add_documents(self, documents: List[StandardDocument]) -> None
    def add_document(self, title: str, standard: str, content: str,
                    part: str = None, version: str = None) -> None
    def add_from_json(self, json_path: str) -> None
    def add_text_chunks(self, text: str, title: str, standard: str,
                       metadata: dict = None) -> None

    def search(self, query: str, top_k: int = None,
               filter_standard: str = None) -> List[Dict]

    def retrieve_for_context(self, query: str, agent_type: str,
                            asil_level: str = None,
                            capability_level: int = None) -> str

    def save(self) -> None
    def load(self) -> bool
    def get_statistics(self) -> Dict
```

### `VectorStoreConfig`

```python
class VectorStoreConfig(BaseModel):
    store_type: str = "faiss"           # Backend choice
    persist_directory: str = "./data/vector_store"
    embedding_model: str = "text-embedding-3-small"
    use_local_embeddings: bool = False
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 3
```

### `StandardDocument`

```python
class StandardDocument(BaseModel):
    title: str                           # Document title
    standard: str                        # Standard name (ISO 26262, ASPICE)
    part: Optional[str]                   # Part/section ID
    version: Optional[str]                # Version
    content: str                         # Full text content
    metadata: Dict[str, Any]             # Additional fields
```

---

## Examples

### Example 1: Ingest ISO 26262 Part 6

```python
from solar_flare.memory import StandardsVectorStore, StandardDocument

store = StandardsVectorStore()

with open("standards/iso_26262_part6.txt", "r") as f:
    content = f.read()

store.add_text_chunks(
    text=content,
    title="ISO 26262-6: Software Level Requirements",
    standard="ISO 26262",
    metadata={"part": "6", "version": "2018"}
)

store.save()
```

### Example 2: Search and Retrieve

```python
# Search for ASIL D requirements
results = store.search(
    query="ASIL D software verification requirements",
    top_k=3
)

for i, result in enumerate(results, 1):
    print(f"\n[Result {i}]")
    print(f"Title: {result['title']}")
    print(f"Standard: {result['standard']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:300]}...")
```

### Example 3: Cross-Standard Search

```python
# Search across all standards
all_results = store.search(
    query="traceability requirements",
    top_k=5
)

# Group by standard
by_standard = {}
for result in all_results:
    std = result["standard"]
    if std not in by_standard:
        by_standard[std] = []
    by_standard[std].append(result)

# Display by standard
for standard, results in by_standard.items():
    print(f"\n{standard}: {len(results)} results")
```

---

## Appendix: Pre-loaded Standards

Solar-Flare includes the following pre-loaded sample documents:

### ISO 26262

| Part | Title | Content Summary |
|------|-------|-----------------|
| 1 | Vocabulary | ASIL definitions, safety goal terminology |
| 4 | System Level | Technical safety concept, HSR, FTTI |
| 5 | Hardware Level | Hardware architecture, diagnostic coverage, ECC |
| 6 | Software Level | Software requirements, coding standards, verification |

### ASPICE

| Process | Title | Content Summary |
|---------|-------|-----------------|
| SWE.1 | Requirements Elicitation | Stakeholder requirements, traceability |
| SWE.2 | System Architecture | Hardware-software allocation, interfaces |
| SWE.3 | Detailed Design | Component design, interfaces |
| SWE.4 | Unit Construction | Coding standards, static analysis |
| SWE.5 | Unit Verification | Testing, coverage by ASIL |

### Hardware Constraints

| Document | Content |
|----------|---------|
| HW-001 | Mailbox, DMA, Timer, CPU/Bandwidth limits |

To replace or expand these samples, use the ingestion workflow described above.
