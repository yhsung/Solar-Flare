"""
Vector Store implementation for RAG on ISO 26262 and ASPICE standards.

This module provides retrieval-augmented generation capabilities by
indexing and querying automotive safety standard documents.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


class VectorStoreConfig(BaseModel):
    """Configuration for vector store initialization."""

    store_type: str = Field(
        default="faiss",
        description="Vector store type: 'faiss' or 'chroma'",
    )
    persist_directory: str = Field(
        default="./data/vector_store",
        description="Directory to persist vector store",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model or HuggingFace model path",
    )
    use_local_embeddings: bool = Field(
        default=False,
        description="Use local HuggingFace embeddings instead of OpenAI",
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
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant chunks to retrieve",
    )


class StandardDocument(BaseModel):
    """Represents a standard document (ISO 26262, ASPICE, etc.)."""

    title: str = Field(description="Document title")
    standard: str = Field(description="Standard name (e.g., ISO 26262, ASPICE)")
    part: Optional[str] = Field(default=None, description="Part or section identifier")
    version: Optional[str] = Field(default=None, description="Document version")
    content: str = Field(description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        metadata = {
            "title": self.title,
            "standard": self.standard,
            "part": self.part,
            "version": self.version,
            **self.metadata,
        }
        return Document(page_content=self.content, metadata=metadata)


class StandardsVectorStore:
    """
    Vector store for automotive safety standards.

    Features:
    - Index ISO 26262 and ASPICE documents
    - Semantic search for relevant clauses
    - RAG integration for agent prompts
    - Support for both FAISS and ChromaDB backends

    Attributes:
        config: Vector store configuration
        vector_store: Underlying vector store instance
        embeddings: Embedding function
    """

    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
    ):
        """
        Initialize the standards vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self.embeddings = self._create_embeddings()
        self.vector_store: Optional[VectorStore] = None
        self._documents: List[Document] = []

        # Create persist directory if needed
        Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

    def _create_embeddings(self) -> Embeddings:
        """
        Create embedding function based on configuration.

        Returns:
            Embeddings instance
        """
        if self.config.use_local_embeddings:
            # Use local HuggingFace model
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            # Use OpenAI embeddings
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY required for OpenAI embeddings. "
                    "Set use_local_embeddings=True to use local models."
                )
            return OpenAIEmbeddings(model=self.config.embedding_model)

    def add_documents(self, documents: List[StandardDocument]) -> None:
        """
        Add standard documents to the vector store.

        Args:
            documents: List of standard documents to add
        """
        langchain_docs = [doc.to_document() for doc in documents]
        self._documents.extend(langchain_docs)
        self._rebuild_store()

    def add_document(
        self,
        title: str,
        standard: str,
        content: str,
        part: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single document to the vector store.

        Args:
            title: Document title
            standard: Standard name
            content: Document content
            part: Optional part/section identifier
            version: Optional version
            metadata: Additional metadata
        """
        doc = StandardDocument(
            title=title,
            standard=standard,
            part=part,
            version=version,
            content=content,
            metadata=metadata or {},
        )
        self.add_documents([doc])

    def add_from_json(self, json_path: str) -> None:
        """
        Load and add documents from a JSON file.

        JSON format:
        {
            "documents": [
                {
                    "title": "Part 1: Vocabulary",
                    "standard": "ISO 26262",
                    "part": "1",
                    "version": "2018",
                    "content": "...",
                    "metadata": {}
                }
            ]
        }

        Args:
            json_path: Path to JSON file
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        documents = []
        for doc_data in data.get("documents", []):
            doc = StandardDocument(**doc_data)
            documents.append(doc)

        self.add_documents(documents)

    def add_text_chunks(
        self,
        text: str,
        title: str,
        standard: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add text as chunked documents.

        Args:
            text: Full text to chunk
            title: Document title
            standard: Standard name
            metadata: Additional metadata
        """
        chunks = self._chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc = StandardDocument(
                title=f"{title} (Chunk {i+1}/{len(chunks)})",
                standard=standard,
                content=chunk,
                metadata={
                    "chunk": i,
                    "total_chunks": len(chunks),
                    **(metadata or {}),
                },
            )
            self._documents.append(doc.to_document())

        self._rebuild_store()

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.config.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.config.chunk_size // 2:
                    chunk = text[start : start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - self.config.chunk_overlap

        return [c for c in chunks if c]

    def _rebuild_store(self) -> None:
        """Rebuild the vector store with current documents."""
        if not self._documents:
            return

        if self.config.store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                self._documents,
                self.embeddings,
            )
        elif self.config.store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                self._documents,
                self.embeddings,
                persist_directory=self.config.persist_directory,
            )
        else:
            raise ValueError(f"Unknown store type: {self.config.store_type}")

    def save(self) -> None:
        """Persist the vector store to disk."""
        if self.vector_store is None:
            return

        if self.config.store_type == "faiss":
            save_path = os.path.join(self.config.persist_directory, "faiss_index")
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save_local(save_path)
        elif self.config.store_type == "chroma":
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()

    def load(self) -> bool:
        """
        Load vector store from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.config.store_type == "faiss":
                save_path = os.path.join(self.config.persist_directory, "faiss_index")
                if os.path.exists(save_path):
                    self.vector_store = FAISS.load_local(
                        save_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    return True
            elif self.config.store_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=self.config.persist_directory,
                    embedding_function=self.embeddings,
                )
                return True
        except Exception as e:
            print(f"Failed to load vector store: {e}")

        return False

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_standard: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant standard documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_standard: Optional filter by standard name

        Returns:
            List of relevant documents with metadata
        """
        if self.vector_store is None:
            return []

        k = top_k or self.config.top_k

        try:
            if filter_standard:
                # Use metadata filter
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k * 2,  # Get more to filter
                )
                # Filter by standard
                filtered = [
                    (doc, score)
                    for doc, score in results
                    if doc.metadata.get("standard") == filter_standard
                ][:k]
                results = filtered
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                )

            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                    "title": doc.metadata.get("title", ""),
                    "standard": doc.metadata.get("standard", ""),
                    "part": doc.metadata.get("part"),
                }
                for doc, score in results
            ]

        except Exception as e:
            print(f"Search failed: {e}")
            return []

    def retrieve_for_context(
        self,
        query: str,
        agent_type: str,
        asil_level: Optional[str] = None,
        capability_level: Optional[int] = None,
    ) -> str:
        """
        Retrieve relevant context for an agent's prompt.

        Args:
            query: Agent's current query/task
            agent_type: Type of agent (iso_26262_analyzer, embedded_designer, etc.)
            asil_level: Optional ASIL level for filtering
            capability_level: Optional ASPICE capability level

        Returns:
            Formatted context string with retrieved standards
        """
        # Build search query with context
        search_query = f"{query} {agent_type}"
        if asil_level:
            search_query += f" ASIL {asil_level}"
        if capability_level:
            search_query += f" ASPICE level {capability_level}"

        results = self.search(search_query, top_k=3)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            part_str = f" Part {result['part']}" if result.get('part') else ''
            context_parts.append(
                f"[Reference {i}] {result['title']} "
                f"({result['standard']}{part_str})\n"
                f"{result['content'][:500]}..."
            )

        return (
            "\n\n## Relevant Standards References\n\n"
            + "\n\n".join(context_parts)
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        return {
            "store_type": self.config.store_type,
            "total_documents": len(self._documents),
            "persist_directory": self.config.persist_directory,
            "embedding_model": self.config.embedding_model,
            "use_local_embeddings": self.config.use_local_embeddings,
            "loaded": self.vector_store is not None,
        }


class EmbeddedStandardsProvider:
    """
    Provider for pre-loaded automotive safety standards.

    Includes sample content from ISO 26262 and ASPICE for demonstration.
    """

    @staticmethod
    def get_iso_26262_samples() -> List[StandardDocument]:
        """Get sample ISO 26262 documents."""
        return [
            StandardDocument(
                title="ISO 26262-1: Vocabulary",
                standard="ISO 26262",
                part="1",
                version="2018",
                content="""
                ISO 26262 defines the following vocabulary:

                ASIL (Automotive Safety Integrity Level): One of four levels
                (QM, A, B, C, D) to specify the necessary requirements for
                achieving safety goals. ASIL D is the highest integrity level.

                Safety Goal: Top-level safety requirement derived from the
                hazard analysis and risk assessment.

                Functional Safety Concept: Specification of the safety
                mechanisms to achieve the safety goals.

                Hardware Architecture Metrics: Diagnostic Coverage (DC),
                Latent Fault Metric (LFM), and Single Point Fault Metric (SPFM).

                For ASIL D: DC > 99%, LFM > 90%, SPFM > 99%.
                """,
                metadata={"keywords": ["ASIL", "safety goal", "metrics"]},
            ),
            StandardDocument(
                title="ISO 26262-4: System Level Design",
                standard="ISO 26262",
                part="4",
                version="2018",
                content="""
                System level design requirements:

                1. Technical Safety Concept: Must specify the safety mechanisms
                   to implement the functional safety concept at the system level.

                2. Hardware-Software Interface: HSR specification must define
                   all interfaces between hardware and software components.

                3. Integration: Must define integration requirements for all
                   system elements, including timing constraints.

                4. Safe State: The system must have a defined safe state that
                   can be reached within the fault tolerant time interval (FTTI).

                5. Fault Tolerant Time Interval (FTTI): Maximum time between
                   fault occurrence and occurrence of a hazardous event if
                   no safety mechanism is activated.

                6. Freedom from Interference: Must demonstrate that lower
                   ASIL components do not interfere with higher ASIL components.
                """,
                metadata={
                    "keywords": [
                        "technical safety concept",
                        "HSR",
                        "integration",
                        "safe state",
                        "FTTI",
                    ]
                },
            ),
            StandardDocument(
                title="ISO 26262-5: Hardware Level Design",
                standard="ISO 26262",
                part="5",
                version="2018",
                content="""
                Hardware level design requirements:

                1. Hardware Safety Requirements: Must be derived from the
                   technical safety concept and include diagnostic mechanisms.

                2. Hardware Architecture: Must support the safety mechanisms
                   with appropriate diagnostic coverage.

                3. Safety Analysis: Must perform FMEA (Failure Mode and
                   Effects Analysis) and FTA (Fault Tree Analysis).

                4. Hardware Design: Must use development process compliant
                   with ASPICE (typically capability level 3 or higher).

                5. Diagnostic Coverage: For ASIL D, must achieve >99%
                   diagnostic coverage of hardware faults.

                6. Memory Protection: Must implement ECC or parity for
                   safety-critical memory.

                7. Timing: Must guarantee timing constraints are met under
                   all operating conditions.
                """,
                metadata={
                    "keywords": [
                        "hardware safety",
                        "diagnostic coverage",
                        "FMEA",
                        "FTA",
                        "ECC",
                        "timing",
                    ]
                },
            ),
            StandardDocument(
                title="ISO 26262-6: Software Level Design",
                standard="ISO 26262",
                part="6",
                version="2018",
                content="""
                Software level design requirements:

                1. Software Safety Requirements: Must be derived from the
                   system safety requirements and technical safety concept.

                2. Software Architecture: Must support freedom from
                   interference and error detection.

                3. Development Environment: Must be qualified for safety-critical
                   development (compilers, build tools, etc.).

                4. Coding Standards: Must follow defined coding standards
                   (MISRA C/C++ typically required for ASIL B and above).

                5. Verification: Must perform unit testing, integration testing,
                   and requirements-based testing with defined coverage metrics.

                6. Data Flow Analysis: Must perform static analysis to detect
                   data flow and control flow anomalies.

                7. Resource Management: Must verify memory usage, stack usage,
                   and timing constraints are within budget.
                """,
                metadata={
                    "keywords": [
                        "software safety",
                        "architecture",
                        "coding standards",
                        "verification",
                        "testing",
                        "MISRA",
                    ]
                },
            ),
        ]

    @staticmethod
    def get_aspice_samples() -> List[StandardDocument]:
        """Get sample ASPICE documents."""
        return [
            StandardDocument(
                title="ASPICE SWE.1: Requirements elicitation",
                standard="ASPICE",
                part="SWE.1",
                version="v3.1",
                content="""
                SWE.1 Requirements Elicitation - Capability Level 1 (Performed):

                Base Practices:
                1. Identify and analyze stakeholder requirements
                2. Specify the system requirements
                3. Define the requirements attributes
                4. Ensure requirements are consistent, complete, and verifiable
                5. Maintain traceability of requirements

                Work Products:
                - System Requirements Specification
                - Stakeholder Requirements Specification

                For Capability Level 2 (Managed), also requires:
                - Requirements management process with change control
                - Resource allocation for requirements engineering
                - Training for personnel
                """,
                metadata={"keywords": ["requirements", "stakeholders", "traceability"]},
            ),
            StandardDocument(
                title="ASPICE SWE.2: System Architecture",
                standard="ASPICE",
                part="SWE.2",
                version="v3.1",
                content="""
                SWE.2 System Architecture - Capability Level 1 (Performed):

                Base Practices:
                1. Identify and analyze architectural requirements
                2. Define the system architecture
                3. Define the hardware-software allocation
                4. Define interfaces between components
                5. Ensure architecture supports all requirements

                Work Products:
                - System Architecture Description
                - Hardware-Software Allocation Specification
                - Interface Specification

                For Capability Level 3 (Established), also requires:
                - Defined process for architecture development
                - Architecture patterns and guidelines
                - Architecture reviews with documented outcomes
                """,
                metadata={"keywords": ["architecture", "interfaces", "allocation"]},
            ),
            StandardDocument(
                title="ASPICE SWE.3: Detailed Design",
                standard="ASPICE",
                part="SWE.3",
                version="v3.1",
                content="""
                SWE.3 Detailed Design - Capability Level 1 (Performed):

                Base Practices:
                1. Design detailed components and units
                2. Define interfaces between units
                3. Define data structures and algorithms
                4. Ensure design supports all requirements

                Work Products:
                - Detailed Design Specification
                - Unit Interface Specification

                For Capability Level 4 (Predictable), also requires:
                - Quantitative process management
                - Process performance baselines
                - Predictable achievement of quality objectives
                """,
                metadata={"keywords": ["detailed design", "units", "interfaces"]},
            ),
            StandardDocument(
                title="ASPICE SWE.4: Unit Construction",
                standard="ASPICE",
                part="SWE.4",
                version="v3.1",
                content="""
                SWE.4 Unit Construction - Capability Level 1 (Performed):

                Base Practices:
                1. Implement units according to detailed design
                2. Follow coding standards
                3. Perform static analysis
                4. Maintain traceability from design to code

                Work Products:
                - Source Code
                - Static Analysis Results
                - Code Review Reports

                For Capability Level 5 (Innovating), also requires:
                - Continuous process improvement
                - Innovation in development practices
                - Industry-leading capabilities
                """,
                metadata={"keywords": ["coding", "static analysis", "implementation"]},
            ),
            StandardDocument(
                title="ASPICE SWE.5: Unit Verification",
                standard="ASPICE",
                part="SWE.5",
                version="v3.1",
                content="""
                SWE.5 Unit Verification - Capability Level 1 (Performed):

                Base Practices:
                1. Define verification criteria for each unit
                2. Develop unit tests
                3. Execute unit tests and document results
                4. Achieve required coverage (statement, decision, MC/DC)

                Work Products:
                - Unit Test Specification
                - Unit Test Results
                - Coverage Reports

                Coverage Requirements by ASIL:
                - ASIL A: Statement coverage recommended
                - ASIL B: Statement coverage required
                - ASIL C: Decision coverage required
                - ASIL D: MC/DC coverage required
                """,
                metadata={"keywords": ["unit testing", "coverage", "verification"]},
            ),
        ]

    @staticmethod
    def get_hardware_constraint_samples() -> List[StandardDocument]:
        """Get sample hardware constraint documents."""
        return [
            StandardDocument(
                title="Automotive Hardware Constraints for Logging",
                standard="INTERNAL",
                part="HW-001",
                version="1.0",
                content="""
                Mandatory Hardware Constraints for Safety-Critical Logging:

                1. Mailbox Transport:
                   - Interrupt-driven for timely delivery
                   - 64-byte payload maximum for control signaling
                   - Must support priority-based queuing

                2. DMA Data Movement:
                   - Zero-copy transfers to minimize CPU overhead
                   - 64 KB per burst maximum
                   - Must support scatter-gather for non-contiguous buffers

                3. Synchronization:
                   - Global Hardware Timer with 1ns resolution
                   - 64-bit timestamp for log entries
                   - Must be accessible from all cores

                4. Performance Budgets:
                   - Maximum 3% CPU overhead per core
                   - Maximum 10 MB/s aggregate bandwidth
                   - Latency < 100us for critical log paths

                5. Memory:
                   - Per-core local ring buffers
                   - Fixed-size allocation at initialization
                   - Overflow policies: overwrite or stop

                6. Safety:
                   - ECC protection for safety-critical buffers
                   - Lock-free implementation for multi-producer
                   - Atomic operations for head/tail indices
                """,
                metadata={"keywords": ["hardware", "constraints", "DMA", "mailbox"]},
            ),
        ]


def create_default_vector_store(
    persist_dir: str = "./data/vector_store",
    use_local_embeddings: bool = False,
) -> StandardsVectorStore:
    """
    Create a vector store pre-loaded with sample standards.

    Args:
        persist_dir: Directory to persist the store
        use_local_embeddings: Use local embeddings instead of OpenAI

    Returns:
        Initialized StandardsVectorStore with sample documents
    """
    config = VectorStoreConfig(
        persist_directory=persist_dir,
        use_local_embeddings=use_local_embeddings,
    )

    store = StandardsVectorStore(config)

    # Try to load existing store
    if store.load():
        return store

    # Load sample documents
    provider = EmbeddedStandardsProvider()

    for doc in provider.get_iso_26262_samples():
        store.add_documents([doc])

    for doc in provider.get_aspice_samples():
        store.add_documents([doc])

    for doc in provider.get_hardware_constraint_samples():
        store.add_documents([doc])

    # Save for next time
    store.save()

    return store
