#!/usr/bin/env python3
"""
Advanced Usage Examples for Solar-Flare

This example demonstrates advanced features including:
1. Streaming responses in real-time
2. Custom agent configuration
3. Manual workflow graph traversal
4. State inspection and debugging
5. Multi-session management
6. Markdown export of agent results
7. Multi-turn requirements clarification with traceability

Prerequisites:
1. Complete basic_usage.py prerequisites
2. Run: python examples/advanced_usage.py
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage

from solar_flare import (
    create_workflow,
    compile_workflow,
    run_workflow,
    HardwareConstraints,
    create_llm,
    LLMProvider,
    export_workflow_results,
    format_workflow_summary,
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

    # Use the create_llm factory for unified provider access
    llm = get_available_llm()

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

    llm = get_available_llm()

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

    llm = get_available_llm()

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

    llm = get_available_llm()

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

    llm = get_available_llm()

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

    llm = get_available_llm()

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
    llm = get_available_llm()

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

async def example_8_markdown_export() -> None:
    """
    Example 8: Export agent results to markdown files.

    Demonstrates how to export each agent's result to well-formatted
    markdown files for documentation and review.
    """
    print_header("Example 8: Markdown Export of Agent Results")

    llm = get_available_llm()

    # Create output directory
    output_dir = Path("./output/agent_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.absolute()}\n")

    # Run workflow with automatic markdown export
    result = await run_workflow(
        llm=llm,
        user_message="Design a lock-free ring buffer for ASIL-D compliance",
        session_id="markdown-export-demo",
        output_dir=str(output_dir),
    )

    # Check exported files
    if "exported_files" in result:
        print("Exported markdown files:")
        for filepath in result["exported_files"]:
            print(f"  ✓ {filepath}")
    else:
        # Manual export if not auto-exported
        print("Manually exporting results...")
        files = export_workflow_results(result, output_dir)
        print("Exported markdown files:")
        for filepath in files:
            print(f"  ✓ {filepath}")

    # Also show summary in console
    print("\n" + "-" * 50)
    print("Workflow Summary Preview:")
    print("-" * 50)
    summary = format_workflow_summary(result)
    # Print first 30 lines of summary
    summary_lines = summary.split("\n")[:30]
    print("\n".join(summary_lines))
    if len(summary.split("\n")) > 30:
        print("\n... (truncated, see full file in output directory)")


def get_available_llm():
    """
    Get an LLM instance based on available providers.
    
    Checks for cloud API keys first, then falls back to local providers.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Using Anthropic Claude")
        return create_llm(provider=LLMProvider.ANTHROPIC, temperature=0.3)
    elif os.getenv("OPENAI_API_KEY"):
        print("Using OpenAI")
        return create_llm(provider=LLMProvider.OPENAI, temperature=0.3)
    elif os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_BASE_URL"):
        print("Using Ollama (local)")
        return create_llm(provider=LLMProvider.OLLAMA, temperature=0.3)
    elif os.getenv("LMSTUDIO_BASE_URL"):
        print("Using LM Studio (local)")
        return create_llm(provider=LLMProvider.LMSTUDIO, temperature=0.3)
    else:
        print("No cloud API keys found. Trying Ollama with default settings...")
        return create_llm(provider=LLMProvider.OLLAMA, temperature=0.3)


async def example_9_multi_turn_requirements() -> None:
    """
    Example 9: Multi-turn requirements clarification with traceability.

    Demonstrates how to:
    1. Conduct iterative requirements clarification across multiple turns
    2. Track requirements traceability across iterations
    3. Export each iteration's output to markdown for audit trail
    4. Build up a complete requirements trace matrix

    This is useful for automotive projects where requirements must be
    fully traced from user needs through to implementation.
    """
    print_header("Example 9: Multi-Turn Requirements Clarification")

    llm = get_available_llm()
    app = compile_workflow(llm, enable_checkpointing=True)
    
    # Create output directory for traceability artifacts
    output_base = Path("./output/requirements_trace")
    output_base.mkdir(parents=True, exist_ok=True)

    # Define a session for multi-turn conversation
    session_id = "requirements-clarification-demo"
    config = {"configurable": {"thread_id": session_id}}

    # Define a set of requirements to trace
    requirements = [
        {
            "id": "REQ-001",
            "title": "Ring Buffer Design",
            "description": "Design a lock-free ring buffer for multi-core logging",
            "priority": "high",
            "asil_level": "ASIL-D",
        },
        {
            "id": "REQ-002",
            "title": "DMA Transport",
            "description": "Implement DMA-based log transport with zero-copy semantics",
            "priority": "high",
            "asil_level": "ASIL-D",
        },
        {
            "id": "REQ-003",
            "title": "Overflow Handling",
            "description": "Handle buffer overflow with configurable policies",
            "priority": "medium",
            "asil_level": "ASIL-B",
        },
    ]

    print("Requirements to trace:")
    for req in requirements:
        print(f"  [{req['id']}] {req['title']} ({req['asil_level']})")
    print()

    # Multi-turn conversation state
    all_results = []
    traceability_matrix = []
    current_state = None

    # =========================================================
    # Iteration 1: Initial requirements analysis
    # =========================================================
    print("-" * 50)
    print("ITERATION 1: Initial Requirements Analysis")
    print("-" * 50)

    initial_message = f"""
    Analyze the following logging system requirements for an automotive ECU:

    Requirements:
    {json.dumps(requirements, indent=2)}

    For each requirement:
    1. Identify ISO 26262 implications based on ASIL level
    2. Suggest high-level design approach
    3. Note any clarifications needed

    Hardware platform: Infineon AURIX TC397
    RTOS: AUTOSAR OS
    """

    current_state = create_initial_state(
        messages=[HumanMessage(content=initial_message)],
        hardware_constraints=HardwareConstraints(),
        max_iterations=10,
    )

    result1 = await app.ainvoke(current_state, config)
    all_results.append(("iteration_1_initial_analysis", result1))

    # Export iteration 1 results
    iter1_dir = output_base / "iteration_1"
    files1 = export_workflow_results(result1, iter1_dir)
    print(f"  Exported {len(files1)} files to {iter1_dir}")

    # Track requirements coverage
    for req in requirements:
        traceability_matrix.append({
            "requirement_id": req["id"],
            "iteration": 1,
            "phase": "initial_analysis",
            "agents_involved": [w.agent_name for w in result1.get("worker_results", [])],
            "status": "analyzed",
        })

    print(f"  Workers invoked: {len(result1.get('worker_results', []))}")

    # =========================================================
    # Iteration 2: Follow-up clarification on REQ-001
    # =========================================================
    print("-" * 50)
    print("ITERATION 2: Clarification for REQ-001 (Ring Buffer)")
    print("-" * 50)

    clarification_message = """
    Follow-up on REQ-001 (Ring Buffer):

    Based on the initial analysis, please clarify:
    1. What is the recommended memory allocation strategy for ASIL-D?
       (static vs dynamic allocation)
    2. How should we handle the atomic operations on Cortex-R52?
    3. What diagnostic coverage is required for ASIL-D?

    Provide specific recommendations with ISO 26262 Part 6 references.
    """

    # Continue the conversation by extending messages
    result2_state = {
        **result1,
        "messages": result1["messages"] + [HumanMessage(content=clarification_message)],
        "iteration_count": 0,  # Reset for new iteration
        "current_phase": "understanding",
    }

    result2 = await app.ainvoke(result2_state, config)
    all_results.append(("iteration_2_req001_clarification", result2))

    # Export iteration 2 results
    iter2_dir = output_base / "iteration_2"
    files2 = export_workflow_results(result2, iter2_dir)
    print(f"  Exported {len(files2)} files to {iter2_dir}")

    traceability_matrix.append({
        "requirement_id": "REQ-001",
        "iteration": 2,
        "phase": "clarification",
        "agents_involved": [w.agent_name for w in result2.get("worker_results", [])],
        "status": "clarified",
    })

    print(f"  Workers invoked: {len(result2.get('worker_results', []))}")

    # =========================================================
    # Iteration 3: Follow-up clarification on REQ-002 & REQ-003
    # =========================================================
    print("-" * 50)
    print("ITERATION 3: Clarification for REQ-002 & REQ-003")
    print("-" * 50)

    combined_clarification = """
    Follow-up on REQ-002 (DMA Transport) and REQ-003 (Overflow Handling):

    For REQ-002:
    - How do we ensure data integrity during DMA transfers?
    - What are the timing constraints for the logging path?

    For REQ-003:
    - What are the ASPICE requirements for overflow handling?
    - How do we trace the overflow policy configuration?

    Please provide a design that addresses both requirements together,
    as they are closely related.
    """

    result3_state = {
        **result2,
        "messages": result2["messages"] + [HumanMessage(content=combined_clarification)],
        "iteration_count": 0,
        "current_phase": "understanding",
    }

    result3 = await app.ainvoke(result3_state, config)
    all_results.append(("iteration_3_req002_003_clarification", result3))

    # Export iteration 3 results
    iter3_dir = output_base / "iteration_3"
    files3 = export_workflow_results(result3, iter3_dir)
    print(f"  Exported {len(files3)} files to {iter3_dir}")

    for req_id in ["REQ-002", "REQ-003"]:
        traceability_matrix.append({
            "requirement_id": req_id,
            "iteration": 3,
            "phase": "clarification",
            "agents_involved": [w.agent_name for w in result3.get("worker_results", [])],
            "status": "clarified",
        })

    print(f"  Workers invoked: {len(result3.get('worker_results', []))}")

    # =========================================================
    # Generate Traceability Report
    # =========================================================
    print("-" * 50)
    print("GENERATING TRACEABILITY REPORT")
    print("-" * 50)

    # Create traceability matrix markdown
    trace_report = generate_traceability_report(
        requirements, traceability_matrix, all_results
    )
    trace_file = output_base / "traceability_matrix.md"
    trace_file.write_text(trace_report, encoding="utf-8")
    print(f"  Traceability matrix: {trace_file}")

    # Summary
    print("\n" + "=" * 50)
    print("MULTI-TURN REQUIREMENTS SUMMARY")
    print("=" * 50)
    print(f"Total iterations: {len(all_results)}")
    print(f"Requirements traced: {len(requirements)}")
    print(f"Traceability entries: {len(traceability_matrix)}")
    print(f"\nOutput directory: {output_base.absolute()}")
    print("\nGenerated artifacts:")
    for subdir in sorted(output_base.iterdir()):
        if subdir.is_dir():
            file_count = len(list(subdir.glob("*.md")))
            print(f"  {subdir.name}/: {file_count} markdown files")
        else:
            print(f"  {subdir.name}")


def generate_traceability_report(
    requirements: list,
    traceability_matrix: list,
    all_results: list,
) -> str:
    """Generate a markdown traceability report."""
    from datetime import datetime

    lines = [
        "# Requirements Traceability Matrix",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Requirements Overview",
        "",
        "| ID | Title | ASIL Level | Priority |",
        "|----|-------|------------|----------|",
    ]

    for req in requirements:
        lines.append(
            f"| {req['id']} | {req['title']} | {req['asil_level']} | {req['priority']} |"
        )

    lines.extend([
        "",
        "## Traceability Matrix",
        "",
        "| Requirement | Iteration | Phase | Agents | Status |",
        "|-------------|-----------|-------|--------|--------|",
    ])

    for trace in traceability_matrix:
        agents = ", ".join(trace["agents_involved"][:2])
        if len(trace["agents_involved"]) > 2:
            agents += f" (+{len(trace['agents_involved']) - 2})"
        lines.append(
            f"| {trace['requirement_id']} | {trace['iteration']} | "
            f"{trace['phase']} | {agents} | {trace['status']} |"
        )

    lines.extend([
        "",
        "## Iteration Summary",
        "",
    ])

    for name, result in all_results:
        worker_count = len(result.get("worker_results", []))
        phase = result.get("current_phase", "unknown")
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"- **Phase:** {phase}")
        lines.append(f"- **Workers:** {worker_count}")
        if result.get("final_response"):
            # Show first 200 chars of response
            response_preview = result["final_response"][:200]
            if len(result["final_response"]) > 200:
                response_preview += "..."
            lines.append(f"- **Response preview:** {response_preview}")
        lines.append("")

    lines.extend([
        "## Artifact Links",
        "",
        "- [Iteration 1: Initial Analysis](./iteration_1/)",
        "- [Iteration 2: REQ-001 Clarification](./iteration_2/)",
        "- [Iteration 3: REQ-002 & REQ-003 Clarification](./iteration_3/)",
    ])

    return "\n".join(lines)




async def main() -> None:
    """Run all advanced examples."""
    load_dotenv()

    # Check for providers
    has_cloud = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    has_local = os.getenv("OLLAMA_MODEL") or os.getenv("LMSTUDIO_BASE_URL")
    
    if not has_cloud and not has_local:
        print("Note: No LLM provider explicitly configured.")
        print("  Cloud: Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
        print("  Local: Set OLLAMA_MODEL or LMSTUDIO_BASE_URL")
        print("  Defaulting to Ollama...")

    print("Solar-Flare Advanced Examples")
    print("=" * 70)

    # Run examples (comment out as needed)
    # await example_1_streaming_workflow()
    # await example_2_step_by_step_execution()
    # await example_3_multi_session_management()
    # await example_4_state_inspection()
    # await example_5_error_handling()
    # await example_6_concurrent_agent_execution()
    # await example_7_custom_constraints()
    # await example_8_markdown_export()
    await example_9_multi_turn_requirements()  # Multi-turn with traceability

    print("\n" + "=" * 70)
    print("Advanced examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
