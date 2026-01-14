"""
Multi-Store RAG Usage Examples for Solar-Flare.

This script demonstrates how to use the MultiStoreRAG system to manage
separate vector stores for different document types.
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

from solar_flare.memory import (
    MultiStoreRAG,
    MultiStoreConfig,
    StoreConfig,
    create_multi_store_rag,
    WorkingMaterialsIngestor,
    StandardDocument,
)


def example_1_basic_usage():
    """Example 1: Create and use MultiStoreRAG with defaults."""
    print("\n=== Example 1: Basic MultiStoreRAG Usage ===\n")

    # Create multi-store system with default configuration
    rag = create_multi_store_rag()

    # List available stores
    print("Available stores:")
    for store_name in rag.list_stores():
        print(f"  - {store_name}")

    # Add documents to different stores
    rag.add_text(
        store_name="standards",
        text="""
        ISO 26262-4 specifies requirements for system level development.
        Key aspects include technical safety concept, system safety specifications,
        and hardware-software interface specifications.
        """,
        title="ISO 26262-4: System Level Development",
        standard="ISO 26262",
        part="4",
    )

    rag.add_text(
        store_name="internal",
        text="""
        The logging subsystem uses a lock-free ring buffer implementation
        with DMA transfers for overflow handling. CPU overhead is limited
        to 2% per core.
        """,
        title="Logging Subsystem Architecture",
        standard="internal",
        document_type="architecture",
    )

    # Search across specific stores
    print("\nSearching in 'standards' store:")
    results = rag.search_all("system level safety requirements", stores=["standards"], top_k=2)
    for store, items in results.items():
        print(f"  Store: {store}")
        for item in items:
            print(f"    - {item.get('title', 'No title')}")

    # Get unified context for agents
    print("\nRetrieving context for agent:")
    context = rag.retrieve_for_context(
        query="lock-free ring buffer",
        agent_type="embedded_designer",
        asil_level="ASIL_D",
    )
    print(f"  Context length: {len(context)} characters")

    # Save all stores
    rag.save_all()
    print("\nStores saved to disk.")


def example_2_custom_configuration():
    """Example 2: Create MultiStoreRAG with custom configuration."""
    print("\n=== Example 2: Custom Configuration ===\n")

    # Define custom store configurations
    stores = {
        "hardware": StoreConfig(
            name="hardware",
            document_types=["datasheet", "reference_manual", "user_guide"],
            persist_directory="./data/hardware_store",
            chunk_size=700,  # Larger chunks for technical specs
            chunk_overlap=100,
        ),
        "standards": StoreConfig(
            name="standards",
            document_types=["ISO 26262", "ASPICE"],
            persist_directory="./data/standards_store",
            chunk_size=500,
            chunk_overlap=50,
        ),
    }

    # Create config
    config = MultiStoreConfig(
        base_directory="./data/vector_stores",
        stores=stores,
    )

    # Create MultiStoreRAG
    rag = MultiStoreRAG(config)

    print("Custom stores created:")
    for store_name in rag.list_stores():
        store = rag.get_store(store_name)
        print(f"  - {store_name}: chunk_size={store.config.chunk_size}")


def example_3_cross_store_search():
    """Example 3: Search across multiple stores simultaneously."""
    print("\n=== Example 3: Cross-Store Search ===\n")

    rag = create_multi_store_rag()

    # Add related documents to different stores
    rag.add_text(
        store_name="standards",
        text="ASIL D requires diagnostic coverage > 99% for safety mechanisms.",
        title="ASIL D Requirements",
        standard="ISO 26262",
    )

    rag.add_text(
        store_name="internal",
        text="The watchdog timer diagnostic provides 99.9% coverage for CPU faults.",
        title="Watchdog Diagnostic Coverage",
        standard="internal",
    )

    rag.add_text(
        store_name="working",
        text="Meeting notes: Decided to use windowed watchdog for better coverage.",
        title="Safety Meeting 2024-01-15",
        standard="working_material",
    )

    # Search across all stores
    print("Searching across all stores for 'diagnostic coverage':")
    results = rag.search_all("diagnostic coverage", top_k=2)

    for store_name, items in results.items():
        if items:
            print(f"\n  {store_name.upper()}:")
            for item in items:
                title = item.get("title", "No title")
                content = item.get("content", "")[:100]
                print(f"    - {title}")
                print(f"      {content}...")

    # Get unified merged results
    print("\nUnified search (merged results):")
    unified = rag.search_unified("diagnostic coverage", top_k=5)
    for i, result in enumerate(unified, 1):
        print(f"  {i}. [{result.get('source_store')}] {result.get('title', 'No title')}")


def example_4_working_materials_ingestion():
    """Example 4: Ingest working materials from directory."""
    print("\n=== Example 4: Working Materials Ingestion ===\n")

    # Create multi-store system
    rag = create_multi_store_rag()
    ingestor = WorkingMaterialsIngestor(rag, target_store="working")

    # Create temporary directory structure for demo
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create directory structure
        (temp_dir / "meetings").mkdir(parents=True)
        (temp_dir / "discussions").mkdir(parents=True)

        # Create sample files
        (temp_dir / "meetings" / "2024-01-15_safety_review.txt").write_text(
            "Safety Review Meeting\n"
            "Date: 2024-01-15\n"
            "Attendees: John, Jane, Bob\n"
            "\n"
            "Discussed ASIL D requirements for logging subsystem.\n"
            "Agreed to implement lock-free ring buffer."
        )

        (temp_dir / "discussions" / "ring_buffer_overflow.txt").write_text(
            "Email thread about overflow handling\n"
            "\n"
            "Options considered:\n"
            "1. Overwrite oldest (LOG_POLICY_OVERWRITE)\n"
            "2. Stop logging (LOG_POLICY_STOP)\n"
            "\n"
            "Decision: Use overwrite for ASIL QM-B, stop for ASIL C-D"
        )

        # Ingest from directory
        counts = ingestor.ingest_directory(temp_dir)

        print("Ingestion complete:")
        for doc_type, count in counts.items():
            print(f"  - {doc_type}: {count} documents")

        # Search the ingested content
        print("\nSearching ingested content:")
        results = rag.search_all("overflow policy", stores=["working"], top_k=3)
        for store, items in results.items():
            for item in items:
                print(f"  - {item.get('title', 'No title')}: {item.get('content', '')[:80]}...")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def example_5_statistics_and_management():
    """Example 5: Get statistics and manage stores."""
    print("\n=== Example 5: Statistics and Store Management ===\n")

    rag = create_multi_store_rag()

    # Add some documents
    for i in range(5):
        rag.add_text(
            store_name="standards",
            text=f"Standard document {i} content",
            title=f"Standard {i}",
            standard="ISO 26262",
        )

    for i in range(3):
        rag.add_text(
            store_name="internal",
            text=f"Internal document {i} content",
            title=f"Internal {i}",
            standard="internal",
        )

    # Get statistics
    print("Store statistics:")
    stats = rag.get_statistics()
    for store_name, store_stats in stats.items():
        print(f"  {store_name}:")
        for key, value in store_stats.items():
            print(f"    {key}: {value}")

    # Get specific store
    standards_store = rag.get_store("standards")
    print(f"\nStandards store type: {type(standards_store).__name__}")

    # Save and load
    rag.save_all()
    print("\nAll stores saved.")
    rag.load_all()
    print("All stores loaded.")


def example_6_retrieve_for_agent_context():
    """Example 6: Build context for agent prompts."""
    print("\n=== Example 6: Agent Context Building ===\n")

    rag = create_multi_store_rag()

    # Add documents
    rag.add_text(
        store_name="standards",
        text="ISO 26262 ASIL D requires independent fault detection.",
        title="ASIL D Requirements",
        standard="ISO 26262",
    )

    rag.add_text(
        store_name="internal",
        text="Our design uses dual-core lockstep for fault detection.",
        title="Safety Mechanism Design",
        standard="internal",
    )

    # Get context for ISO 26262 analyzer agent
    print("Context for ISO 26262 analyzer:")
    context = rag.retrieve_for_context(
        query="fault detection mechanisms",
        agent_type="iso_26262_analyzer",
        asil_level="ASIL_D",
        include_working_materials=False,
    )
    print(context[:400] + "...")

    # Get context with working materials included
    print("\n\nContext with working materials:")
    context_with_working = rag.retrieve_for_context(
        query="fault detection mechanisms",
        agent_type="design_reviewer",
        asil_level="ASIL_D",
        include_working_materials=True,
    )
    print(context_with_working[:400] + "...")


def example_7_direct_document_ingestion():
    """Example 7: Ingest StandardDocument objects directly."""
    print("\n=== Example 7: Direct Document Ingestion ===\n")

    rag = create_multi_store_rag()

    # Create StandardDocument objects
    docs = [
        StandardDocument(
            title="ISO 26262-6: Software Development",
            standard="ISO 26262",
            part="6",
            version="2018",
            content="Part 6 specifies requirements for software development at ASIL levels QM to D.",
            metadata={"language": "en", "domain": "functional_safety"},
        ),
        StandardDocument(
            title="GB/T 34590-6: Software Development",
            standard="GB/T 34590",
            part="6",
            version="2022",
            content="GB/T 34590-6 规定了软件级产品开发的要求。",
            metadata={"language": "zh", "domain": "functional_safety"},
        ),
    ]

    # Add to standards store
    for doc in docs:
        rag.add_document("standards", doc)
        print(f"Added: {doc.title}")

    # Search and retrieve
    results = rag.search_all("software development requirements", top_k=3)
    print(f"\nFound {len(results.get('standards', []))} results")


def main():
    """Run all examples."""
    load_dotenv()

    print("=" * 60)
    print("Solar-Flare Multi-Store RAG Usage Examples")
    print("=" * 60)

    examples = [
        example_1_basic_usage,
        example_2_custom_configuration,
        example_3_cross_store_search,
        example_4_working_materials_ingestion,
        example_5_statistics_and_management,
        example_6_retrieve_for_agent_context,
        example_7_direct_document_ingestion,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
