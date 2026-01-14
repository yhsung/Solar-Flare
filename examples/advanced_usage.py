#!/usr/bin/env python3
"""
Advanced Usage Examples for Solar-Flare

This example demonstrates advanced features including:
1. Streaming responses in real-time
2. Custom agent configuration
3. Manual workflow graph traversal
4. State inspection and debugging
5. Multi-session management

Prerequisites:
1. Complete basic_usage.py prerequisites
2. Run: python examples/advanced_usage.py
"""

import asyncio
import os
import json
from typing import Any, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from solar_flare import (
    create_workflow,
    compile_workflow,
    HardwareConstraints,
)
from solar_flare.graph.state import create_initial_state


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


async def example_1_streaming_workflow() -> None:
    """
    Example 1: Real-time streaming of workflow execution.

    Demonstrates how to stream agent outputs as they are generated,
    allowing for real-time progress tracking.
    """
    print_header("Example 1: Streaming Workflow Execution")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Compile workflow
    app = compile_workflow(llm, enable_checkpointing=True)

    # Create initial state
    initial_state = create_initial_state(
        messages=[HumanMessage(content="Design a lock-free ring buffer for ASIL-D")],
        hardware_constraints=HardwareConstraints(),
    )

    config = {"configurable": {"thread_id": "streaming-example"}}

    print("Streaming workflow execution...\n")

    # Stream the workflow execution
    async for event in app.astream_events(initial_state, config, version="v1"):
        # Filter for agent completion events
        event_type = event.get("event", "")

        if "LLM" in event_type:
            name = event.get("name", "unknown")
            if "chain" not in name.lower():  # Filter out internal chains
                print(f"[{event_type}] {name}")

        if "chain" in event_type and "end" in event_type:
            name = event.get("name", "")
            if any(agent in name for agent in [
                "orchestrator", "iso_26262", "embedded", "aspice", "review"
            ]):
                print(f"✓ Completed: {name}")

    print("\nStreaming complete!")


async def example_2_step_by_step_execution() -> None:
    """
    Example 2: Step-by-step workflow execution.

    Demonstrates manual control over workflow execution,
    allowing inspection of state at each step.
    """
    print_header("Example 2: Step-by-Step Execution")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Create workflow (not compiled)
    workflow = create_workflow(llm, HardwareConstraints())
    app = workflow.compile()

    # Initial state
    state = create_initial_state(
        messages=[HumanMessage(content="Analyze ASIL-B requirements for CAN bus logging")],
        hardware_constraints=HardwareConstraints(),
    )

    print("Starting step-by-step execution...\n")

    step = 0
    current_state = state

    while True:
        step += 1
        print(f"\n--- Step {step} ---")
        print(f"Current Phase: {current_state.get('current_phase', 'unknown')}")

        # Get the next node to execute
        # This is a simplified example - real implementation would use graph state
        if current_state.get("current_phase") == "complete":
            print("Workflow complete!")
            break

        if step > 10:  # Safety limit
            print("Reached step limit")
            break

        # Execute one step (simplified - actual implementation varies)
        try:
            current_state = await app.ainvoke(current_state)
        except Exception as e:
            print(f"Execution stopped: {e}")
            break

    print(f"\nFinal phase: {current_state.get('current_phase')}")
    print(f"Total iterations: {current_state.get('iteration_count', 0)}")


async def example_3_multi_session_management() -> None:
    """
    Example 3: Managing multiple concurrent sessions.

    Demonstrates how to handle multiple independent design sessions
    simultaneously using different thread IDs.
    """
    print_header("Example 3: Multi-Session Management")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    app = compile_workflow(llm, enable_checkpointing=True)

    # Define multiple sessions with different requests
    sessions = {
        "session-1": "Design ring buffer for powertrain ECU",
        "session-2": "Analyze ASPICE compliance for diagnostic logging",
        "session-3": "Design DMA transport for safety-critical data",
    }

    print(f"Running {len(sessions)} concurrent sessions...\n")

    # Run sessions concurrently
    async def run_session(session_id: str, request: str) -> Dict[str, Any]:
        state = create_initial_state(
            messages=[HumanMessage(content=request)],
            hardware_constraints=HardwareConstraints(),
        )
        config = {"configurable": {"thread_id": session_id}}
        result = await app.ainvoke(state, config)
        return {
            "session_id": session_id,
            "phase": result.get("current_phase"),
            "iterations": result.get("iteration_count", 0),
            "workers": len(result.get("worker_results", [])),
        }

    # Execute all sessions concurrently
    results = await asyncio.gather(*[
        run_session(sid, req)
        for sid, req in sessions.items()
    ])

    # Display results
    print("Session Results:")
    for result in results:
        print(f"\n{result['session_id']}:")
        print(f"  Phase: {result['phase']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Workers Invoked: {result['workers']}")


async def example_4_state_inspection() -> None:
    """
    Example 4: Detailed state inspection and debugging.

    Demonstrates how to inspect the internal state of the workflow
    for debugging and analysis purposes.
    """
    print_header("Example 4: State Inspection & Debugging")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    app = compile_workflow(llm, enable_checkpointing=True)

    state = create_initial_state(
        messages=[HumanMessage(content="Design log entry format for ASIL-D")],
        hardware_constraints=HardwareConstraints(),
    )

    config = {"configurable": {"thread_id": "debug-session"}}

    print("Initial State:")
    print(f"  Phase: {state.get('current_phase')}")
    print(f"  Messages: {len(state.get('messages', []))}")
    print(f"  Iterations: {state.get('iteration_count', 0)}")
    print(f"  Hardware Constraints: {state.get('hardware_constraints')}")

    # Execute
    result = await app.ainvoke(state, config)

    print("\nFinal State Details:")

    # Inspect orchestrator decision
    if result.get("orchestrator_decision"):
        decision = result["orchestrator_decision"]
        print(f"\n  Orchestrator Decision:")
        print(f"    Next Agent: {decision.next_agent}")
        print(f"    Reasoning: {decision.reasoning}")

    # Inspect worker results
    if result.get("worker_results"):
        print(f"\n  Worker Results ({len(result['worker_results'])} agents):")
        for worker in result["worker_results"]:
            print(f"\n    {worker.agent_name}:")
            print(f"      Task Type: {worker.task_type}")
            print(f"      Status: {worker.status}")
            print(f"      Confidence: {worker.confidence_score:.0%}")
            if worker.findings:
                print(f"      Findings: {len(worker.findings)}")
            if worker.recommendations:
                print(f"      Recommendations: {len(worker.recommendations)}")

    # Inspect design review
    if result.get("design_review"):
        review = result["design_review"]
        print(f"\n  Design Review:")
        print(f"    Status: {review.overall_status}")
        print(f"    Completeness: {review.completeness_score:.0f}%")
        print(f"    Gaps: {len(review.gaps_identified)}")
        print(f"    Violations: {len(review.constraint_violations)}")

    # Export state to JSON for analysis
    state_export = {
        "current_phase": result.get("current_phase"),
        "iteration_count": result.get("iteration_count"),
        "worker_count": len(result.get("worker_results", [])),
        "has_design_review": result.get("design_review") is not None,
    }

    print(f"\n  State Export (JSON):")
    print(json.dumps(state_export, indent=2))


async def example_5_error_handling() -> None:
    """
    Example 5: Error handling and recovery.

    Demonstrates how the system handles errors and
    how to implement retry logic.
    """
    print_header("Example 5: Error Handling & Recovery")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    app = compile_workflow(llm, enable_checkpointing=True)

    # Create a state with intentionally high iteration limit
    state = create_initial_state(
        messages=[HumanMessage(content="Design a complete logging subsystem")],
        hardware_constraints=HardwareConstraints(),
        max_iterations=5,  # Low limit to test iteration handling
    )

    config = {"configurable": {"thread_id": "error-test"}}

    print("Testing with iteration limit of 5...\n")

    try:
        result = await app.ainvoke(state, config)

        print(f"Execution completed")
        print(f"  Final Phase: {result.get('current_phase')}")
        print(f"  Iterations Used: {result.get('iteration_count', 0)}")
        print(f"  Max Iterations: {state.get('max_iterations', 10)}")

        # Check if forced synthesis occurred
        if result.get("iteration_count", 0) >= state.get("max_iterations", 10):
            print("\n  Note: Reached iteration limit, forced synthesis")

    except Exception as e:
        print(f"Error occurred: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        print("\n  Recovery: Check state and retry with adjusted parameters")


async def example_6_concurrent_agent_execution() -> None:
    """
    Example 6: Parallel execution patterns.

    Demonstrates how to structure requests for optimal
    parallel agent execution.
    """
    print_header("Example 6: Concurrent Agent Execution")

    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Complex request that will trigger multiple agents
    complex_request = """
    I need a comprehensive logging solution with:

    1. ISO 26262 ASIL-D compliance analysis
    2. Embedded design for Cortex-R52 multi-core
    3. ASPICE level 3 process assessment
    4. Design review for all components

    Please analyze all aspects in parallel where possible.
    """

    print("Request Type: Complex, multi-agent\n")

    state = create_initial_state(
        messages=[HumanMessage(content=complex_request)],
        hardware_constraints=HardwareConstraints(),
        max_iterations=20,
    )

    app = compile_workflow(llm, enable_checkpointing=True)
    config = {"configurable": {"thread_id": "concurrent-test"}}

    import time
    start_time = time.time()

    result = await app.ainvoke(state, config)

    elapsed = time.time() - start_time

    print(f"\nExecution completed in {elapsed:.2f} seconds")
    print(f"Agents invoked: {len(result.get('worker_results', []))}")
    print(f"Total iterations: {result.get('iteration_count', 0)}")

    # Show execution order
    print("\nAgent Execution Order:")
    for i, worker in enumerate(result.get("worker_results", []), 1):
        print(f"  {i}. {worker.agent_name} ({worker.task_type})")


async def example_7_custom_constraints() -> None:
    """
    Example 7: Working with custom hardware constraints.

    Demonstrates how to define and use custom hardware
    constraints for different automotive platforms.
    """
    print_header("Example 7: Custom Hardware Constraints")

    # Define platform-specific constraints
    platform_constraints = {
        "infineon-aurix": HardwareConstraints(
            mailbox_payload_bytes=64,
            dma_burst_bytes=65536,
            max_cpu_overhead_percent=3.0,
            max_bandwidth_mbps=10.0,
        ),
        "renesas-rh850": HardwareConstraints(
            mailbox_payload_bytes=32,
            dma_burst_bytes=32768,
            max_cpu_overhead_percent=2.0,
            max_bandwidth_mbps=8.0,
        ),
        "nxp-s32": HardwareConstraints(
            mailbox_payload_bytes=128,
            dma_burst_bytes=131072,
            max_cpu_overhead_percent=5.0,
            max_bandwidth_mbps=15.0,
        ),
    }

    print("Platform-Specific Constraints:\n")

    for platform, constraints in platform_constraints.items():
        print(f"{platform.upper()}:")
        print(f"  Mailbox: {constraints.mailbox_payload_bytes} bytes")
        print(f"  DMA: {constraints.dma_burst_bytes // 1024} KB")
        print(f"  CPU: {constraints.max_cpu_overhead_percent}%")
        print(f"  Bandwidth: {constraints.max_bandwidth_mbps} MB/s")
        print()

    # Example: Select platform based on request
    if os.getenv("ANTHROPIC_API_KEY"):
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    else:
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # Use AURIX constraints
    selected_constraints = platform_constraints["infineon-aurix"]

    state = create_initial_state(
        messages=[HumanMessage(
            content="Design logging for AURIX TC397 with custom constraints"
        )],
        hardware_constraints=selected_constraints,
    )

    app = compile_workflow(llm)
    result = await app.ainvoke(state)

    print("Design generated with custom constraints applied.")
    print(f"Workers invoked: {len(result.get('worker_results', []))}")


async def main() -> None:
    """Run all advanced examples."""
    load_dotenv()

    # Verify API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
        return

    print("Solar-Flare Advanced Examples")
    print("=" * 70)

    # Run examples (comment out as needed)
    await example_1_streaming_workflow()
    # await example_2_step_by_step_execution()
    # await example_3_multi_session_management()
    # await example_4_state_inspection()
    # await example_5_error_handling()
    # await example_6_concurrent_agent_execution()
    # await example_7_custom_constraints()

    print("\n" + "=" * 70)
    print("Advanced examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
