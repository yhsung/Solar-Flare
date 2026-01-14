#!/usr/bin/env python3
"""
Basic Usage Example for Solar-Flare

This example demonstrates how to use the Solar-Flare multi-agent system
to design a safety-critical logging service for automotive embedded systems.

Prerequisites:
1. Install dependencies: pip install -e .
2. Set up API keys in .env file:
   - OPENAI_API_KEY or ANTHROPIC_API_KEY
   - TAVILY_API_KEY (for web search)
3. Run: python examples/basic_usage.py
"""

import asyncio
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from solar_flare import (
    create_workflow,
    compile_workflow,
    run_workflow,
    HardwareConstraints,
    ASILLevel,
    CapabilityLevel,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_result(key: str, value: str) -> None:
    """Print a formatted result key-value pair."""
    print(f"{key}: {value}")


async def example_1_simple_design_request() -> None:
    """
    Example 1: Simple logging component design request.

    The system will route the request to the appropriate agents
    and generate a design compliant with hardware constraints.
    """
    print_section("Example 1: Simple Ring Buffer Design Request")

    # Initialize LLM (choose one based on your API key)
    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    elif os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    else:
        raise ValueError("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

    # User request
    user_request = """
    Design a lock-free ring buffer implementation for ASIL-D compliance
    on a Cortex-R5 multi-core system using AUTOSAR OS.
    """

    print(f"User Request: {user_request.strip()}\n")

    # Run the workflow
    result = await run_workflow(
        llm=llm,
        user_message=user_request,
        session_id="example-1",
        max_iterations=10,
    )

    # Display results
    print_section("Results")

    # Show which agents were invoked
    if result.get("worker_results"):
        print("Agents Invoked:")
        for worker_result in result["worker_results"]:
            print(f"  - {worker_result.agent_name} ({worker_result.task_type})")
            print(f"    Status: {worker_result.status}")
            print(f"    Confidence: {worker_result.confidence_score:.0%}")

    # Show final response
    if result.get("final_response"):
        print_section("Final Design Response")
        print(result["final_response"])

    # Show design review if available
    if result.get("design_review"):
        print_section("Design Review Assessment")
        review = result["design_review"]
        print_result("Overall Status", review.overall_status)
        print_result("Completeness", f"{review.completeness_score:.0f}%")

        if review.gaps_identified:
            print(f"\nGaps Identified: {len(review.gaps_identified)}")
            for gap in review.gaps_identified[:3]:
                print(f"  - [{gap.get('severity', 'unknown')}] {gap.get('description', '')}")


async def example_2_custom_hardware_constraints() -> None:
    """
    Example 2: Design with custom hardware constraints.

    Demonstrates how to override default hardware constraints
    for specific platform requirements.
    """
    print_section("Example 2: Custom Hardware Constraints")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Define custom hardware constraints
    custom_constraints = HardwareConstraints(
        mailbox_payload_bytes=128,  # Larger mailbox for this platform
        dma_burst_bytes=32768,  # 32 KB DMA burst
        max_cpu_overhead_percent=2.0,  # Stricter CPU budget
        max_bandwidth_mbps=5.0,  # Lower bandwidth limit
    )

    print("Custom Hardware Constraints:")
    print(f"  Mailbox Payload: {custom_constraints.mailbox_payload_bytes} bytes")
    print(f"  DMA Burst: {custom_constraints.dma_burst_bytes // 1024} KB")
    print(f"  Max CPU Overhead: {custom_constraints.max_cpu_overhead_percent}%")
    print(f"  Max Bandwidth: {custom_constraints.max_bandwidth_mbps} MB/s")

    user_request = """
    Design a DMA-based logging transport for Infineon AURIX TC397
    with ASIL-B requirements. The design must support multi-core
    synchronization using the global hardware timer.
    """

    print(f"\nUser Request: {user_request.strip()}")

    result = await run_workflow(
        llm=llm,
        user_message=user_request,
        hardware_constraints=custom_constraints,
        session_id="example-2",
    )

    if result.get("final_response"):
        print_section("Final Design Response")
        print(result["final_response"])


async def example_3_compliance_analysis() -> None:
    """
    Example 3: ISO 26262 and ASPICE compliance analysis.

    The system analyzes an existing design for compliance gaps
    and provides recommendations.
    """
    print_section("Example 3: Compliance Analysis")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.2)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    user_request = """
    Analyze the following logging design for ISO 26262 ASIL-D compliance:

    Design Overview:
    - Multi-producer, single-consumer ring buffer
    - Lock-free using atomic head/tail indices
    - DMA transfer to external memory on overflow
    - Interrupt-driven notification on buffer full

    Platform: RH850/P1x-H, ASIL-D requirements
    Target: ASPICE Capability Level 3

    Please identify compliance gaps and provide specific recommendations
    for achieving ASIL-D certification.
    """

    print(f"User Request: {user_request.strip()}")

    result = await run_workflow(
        llm=llm,
        user_message=user_request,
        session_id="example-3",
    )

    if result.get("final_response"):
        print_section("Compliance Analysis Results")
        print(result["final_response"])

    # Show compliance-specific findings
    if result.get("worker_results"):
        print_section("Detailed Findings")
        for worker in result["worker_results"]:
            if "ISO" in worker.agent_name.upper() or "ASPICE" in worker.agent_name.upper():
                print(f"\n{worker.agent_name} Findings:")
                for finding in worker.findings[:5]:
                    print(f"  - {finding.get('description', 'N/A')}")


async def example_4_interactive_workflow() -> None:
    """
    Example 4: Interactive workflow with checkpointing.

    Demonstrates how to use the compiled workflow directly
    for multi-turn conversations with state persistence.
    """
    print_section("Example 4: Interactive Workflow")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Compile workflow with checkpointing enabled
    app = compile_workflow(llm, enable_checkpointing=True)

    session_id = "example-4-session"

    # First interaction
    print("Turn 1: Initial Request")
    config = {"configurable": {"thread_id": session_id}}

    from langchain_core.messages import HumanMessage
    from solar_flare.graph.state import create_initial_state, HardwareConstraints

    initial_state = create_initial_state(
        messages=[HumanMessage(content="Design a log entry format for automotive diagnostics")],
        hardware_constraints=HardwareConstraints(),
    )

    result1 = await app.ainvoke(initial_state, config)
    print(f"Phase: {result1.get('current_phase')}")
    print(f"Workers Invoked: {len(result1.get('worker_results', []))}")

    # Follow-up interaction (resumes from checkpoint)
    print("\nTurn 2: Follow-up Request")

    from langchain_core.messages import HumanMessage

    follow_up_state = {
        **result1,
        "messages": result1["messages"] + [HumanMessage(content="Add support for variable-length arguments")],
    }

    result2 = await app.ainvoke(follow_up_state, config)
    print(f"Phase: {result2.get('current_phase')}")

    if result2.get("final_response"):
        print_section("Final Response")
        print(result2["final_response"])


async def example_5_design_review_workflow() -> None:
    """
    Example 5: Full design workflow with review.

    Demonstrates the complete workflow from design request
    through automated design review and gap analysis.
    """
    print_section("Example 5: Full Design with Review")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    user_request = """
    I need a complete logging service design for a brake-by-wire system
    with the following requirements:

    Safety Level: ASIL-D
    Platform: Infineon AURIX TC397 (TriCore)
    RTOS: AUTOSAR OS 4.x
    ASPICE Target: Capability Level 4

    Components needed:
    1. Lock-free ring buffer for log storage
    2. DMA-based transport to external memory
    3. Interrupt-driven notification system
    4. Log entry format with timestamps

    The design must handle:
    - Multi-core concurrent logging
    - Zero-copy data movement
    - Overflow handling with policy selection
    - End-to-end traceability for certification
    """

    print(f"User Request: {user_request.strip()}")

    result = await run_workflow(
        llm=llm,
        user_message=user_request,
        session_id="example-5",
        max_iterations=15,
    )

    # Comprehensive results display
    print_section("Workflow Execution Summary")

    print(f"Total Iterations: {result.get('iteration_count', 0)}")
    print(f"Final Phase: {result.get('current_phase')}")

    # Show all worker results
    if result.get("worker_results"):
        print_section("Agent Execution Summary")
        for worker in result["worker_results"]:
            print(f"\n{worker.agent_name}:")
            print(f"  Task: {worker.task_type}")
            print(f"  Status: {worker.status}")
            print(f"  Confidence: {worker.confidence_score:.0%}")
            if worker.recommendations:
                print(f"  Recommendations: {len(worker.recommendations)}")

    # Show design review results
    if result.get("design_review"):
        review = result["design_review"]
        print_section("Design Review Results")
        print_result("Status", review.overall_status.upper())
        print_result("Completeness", f"{review.completeness_score:.0f}%")

        if review.constraint_violations:
            print(f"\nConstraint Violations: {len(review.constraint_violations)}")
            for violation in review.constraint_violations[:3]:
                print(f"  - {violation.get('constraint', 'unknown')}")

        if review.recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(review.recommendations[:5], 1):
                print(f"  {i}. {rec}")

    # Final response
    if result.get("final_response"):
        print_section("Final Design Deliverable")
        print(result["final_response"])


async def main() -> None:
    """Run all examples."""
    # Load environment variables
    load_dotenv()

    # Verify API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
        return

    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not set. Web search will not work.")

    print("Solar-Flare Multi-Agent System Examples")
    print("=" * 70)

    # Run examples (comment out as needed)
    await example_1_simple_design_request()
    # await example_2_custom_hardware_constraints()
    # await example_3_compliance_analysis()
    # await example_4_interactive_workflow()
    # await example_5_design_review_workflow()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
