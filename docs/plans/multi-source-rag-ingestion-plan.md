# Multi-Source RAG Data Ingestion System

## Overview

Implement a comprehensive multi-source RAG data ingestion system that extends the existing `MultiStoreRAG` to support multiple file formats and data sources with versioned ingestion.

## Requirements Summary

- **File Formats**: PDF, Markdown, Plain text, Word (.docx), PowerPoint (.pptx), Excel (.xlsx)
- **Data Sources**: Local filesystem, CIFS/SMB shares, Confluence API, SharePoint API
- **Versioning**: Keep all versions with timestamps for historical queries
- **Authentication**: API tokens for cloud sources

## Architecture

```
┌─────────────────────────────────────┐
│       IngestionOrchestrator         │
│  - Coordinates all connectors       │
│  - Progress tracking & retry        │
└───────────────┬─────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───┴───┐  ┌────┴────┐  ┌───┴───┐
│Source │  │Document │  │Version│
│Connect│  │Loaders  │  │Store  │
└───────┘  └─────────┘  └───────┘
```

## Files to Create

```
src/solar_flare/memory/ingestion/
├── __init__.py                    # Module exports
├── config.py                      # Configuration models
├── loaders/
│   ├── __init__.py
│   ├── base.py                    # BaseDocumentLoader ABC
│   ├── pdf_loader.py              # PyMuPDF-based
│   ├── markdown_loader.py         # Markdown/RST
│   ├── text_loader.py             # Plain text
│   ├── docx_loader.py             # python-docx
│   ├── pptx_loader.py             # python-pptx
│   └── xlsx_loader.py             # openpyxl
├── connectors/
│   ├── __init__.py
│   ├── base.py                    # BaseSourceConnector ABC
│   ├── local_fs.py                # Local filesystem
│   ├── smb_connector.py           # CIFS/SMB
│   ├── confluence_connector.py    # Confluence REST API
│   └── sharepoint_connector.py    # Microsoft Graph API
├── versioning.py                  # SQLite version tracking
├── orchestrator.py                # Main orchestrator
└── progress.py                    # Progress tracking

tests/unit/test_ingestion/
├── __init__.py
├── test_loaders.py
├── test_connectors.py
├── test_versioning.py
└── test_orchestrator.py
```

## Implementation Steps

### Step 1: Configuration Models
**File**: `src/solar_flare/memory/ingestion/config.py`

- `SourceType` enum: LOCAL_FS, SMB, CONFLUENCE, SHAREPOINT
- `FileFormat` enum: PDF, MARKDOWN, TEXT, DOCX, PPTX, XLSX
- `AuthConfig` base + `APITokenAuth`, `BasicAuth`, `OAuth2Auth`
- `SourceConfig` base + `LocalFSConfig`, `SMBConfig`, `ConfluenceConfig`, `SharePointConfig`
- `VersioningConfig`: db_path, keep_all_versions
- `IngestionConfig`: master config with sources list

### Step 2: Document Loaders
**Files**: `src/solar_flare/memory/ingestion/loaders/*.py`

Base interface (`LoadedDocument` dataclass):
- content, title, source_path, file_format
- content_hash (SHA-256), metadata, extracted_at

Loaders to implement:
| Loader | Library | Extensions |
|--------|---------|------------|
| PDFLoader | pymupdf | .pdf |
| DocxLoader | python-docx | .docx, .doc |
| PptxLoader | python-pptx | .pptx, .ppt |
| XlsxLoader | openpyxl | .xlsx, .xls |
| MarkdownLoader | built-in | .md, .markdown, .rst |
| TextLoader | built-in | .txt, .log, .csv |

### Step 3: Source Connectors
**Files**: `src/solar_flare/memory/ingestion/connectors/*.py`

Base interface methods:
- `async connect()` / `async disconnect()`
- `async list_documents()` → Generator[SourceDocument]
- `async get_document_stream(path)` → BinaryIO

Connectors:
| Connector | Library | Auth |
|-----------|---------|------|
| LocalFSConnector | pathlib | None |
| SMBConnector | smbprotocol | BasicAuth (NTLM) |
| ConfluenceConnector | httpx | APITokenAuth |
| SharePointConnector | httpx + msal | OAuth2Auth |

### Step 4: Version Tracking
**File**: `src/solar_flare/memory/ingestion/versioning.py`

SQLite schema:
```sql
CREATE TABLE document_versions (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    ingested_at TIMESTAMP NOT NULL,
    metadata TEXT,
    vector_store TEXT NOT NULL,
    is_current BOOLEAN DEFAULT 1
);
```

Key methods:
- `add_version(source_path, content_hash, vector_store, metadata)`
- `has_changed(source_path, content_hash)` → bool
- `get_all_versions(source_path)` → List[DocumentVersion]
- `get_documents_by_date_range(start, end)` → List[DocumentVersion]

### Step 5: Ingestion Orchestrator
**File**: `src/solar_flare/memory/ingestion/orchestrator.py`

`IngestionOrchestrator` class:
- Initializes with `IngestionConfig` and `MultiStoreRAG`
- Registers all loaders by extension
- Registers all connectors by source type
- `async ingest_all()` → IngestionResult
- `async ingest_source(name)` → IngestionResult
- Batch processing with configurable batch_size
- Progress callbacks every N documents
- Automatic retry with exponential backoff

### Step 6: Update Module Exports
**File**: `src/solar_flare/memory/__init__.py`

Add exports:
```python
from solar_flare.memory.ingestion import (
    IngestionOrchestrator,
    IngestionConfig,
    LocalFSConfig,
    SMBConfig,
    ConfluenceConfig,
    SharePointConfig,
    VersionStore,
)
```

### Step 7: Update Dependencies
**File**: `pyproject.toml`

Add to dependencies:
```toml
"pymupdf>=1.23.0",
"python-docx>=1.1.0",
"python-pptx>=0.6.23",
"openpyxl>=3.1.0",
"beautifulsoup4>=4.12.0",
"smbprotocol>=1.12.0",
"msal>=1.26.0",
```

### Step 8: Unit Tests
**File**: `tests/unit/test_ingestion/`

Test coverage:
- Each loader: load from path, load from stream, error handling
- Each connector: connect, list documents, get stream, auth errors
- Version store: add version, detect changes, historical queries
- Orchestrator: batch processing, progress callbacks, error recovery

## Key Integration Points

### Existing Files to Reference (no changes needed)
- [multi_store.py](src/solar_flare/memory/multi_store.py) - `MultiStoreRAG.add_document()` method
- [vector_store.py](src/solar_flare/memory/vector_store.py) - `StandardDocument` model
- [ingestor.py](src/solar_flare/memory/ingestor.py) - Pattern for `WorkingMaterialsIngestor`

### Document Flow
1. Connector lists documents from source
2. Loader extracts content from document stream
3. Version store checks if content changed
4. If changed: create `StandardDocument`, add to `MultiStoreRAG`, record version
5. Save vector store after each batch

## Usage Example

```python
from solar_flare.memory import create_multi_store_rag
from solar_flare.memory.ingestion import (
    IngestionOrchestrator, IngestionConfig,
    LocalFSConfig, ConfluenceConfig, APITokenAuth
)

async def main():
    multi_store = create_multi_store_rag()

    config = IngestionConfig(
        sources=[
            LocalFSConfig(
                name="design_docs",
                root_path=Path("./docs"),
                target_store="internal",
            ),
            ConfluenceConfig(
                name="wiki",
                base_url="https://company.atlassian.net",
                space_keys=["SAFETY"],
                target_store="internal",
                auth=APITokenAuth(token="..."),
            ),
        ]
    )

    orchestrator = IngestionOrchestrator(config, multi_store)
    result = await orchestrator.ingest_all()
    print(f"Ingested {result.successful} documents")
```

## Verification

1. **Unit tests**: Run `pytest tests/unit/test_ingestion/`
2. **Local FS test**: Ingest a folder with mixed file types
3. **Version test**: Re-ingest same folder, verify skipped unchanged
4. **Cloud test**: Configure Confluence/SharePoint with test space
5. **Historical query**: Query documents by date range
