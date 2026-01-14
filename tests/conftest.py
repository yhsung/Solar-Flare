"""
Pytest configuration and fixtures for Solar-Flare tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import List

from langchain_core.messages import HumanMessage, AIMessage
from solar_flare.graph.state import HardwareConstraints, AgentState, create_initial_state


@pytest.fixture
def hardware_constraints() -> HardwareConstraints:
    """Provide default hardware constraints for tests."""
    return HardwareConstraints()


@pytest.fixture
def custom_hardware_constraints() -> HardwareConstraints:
    """Provide custom hardware constraints for edge case tests."""
    return HardwareConstraints(
        mailbox_payload_bytes=32,
        dma_burst_bytes=32768,  # 32 KB
        max_cpu_overhead_percent=5.0,
        max_bandwidth_mbps=15.0,
    )


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing agents."""
    llm = MagicMock()

    # Create async mock for ainvoke
    async_response = MagicMock()
    async_response.content = "Mock LLM response for testing"

    llm.ainvoke = AsyncMock(return_value=async_response)
    llm.invoke = MagicMock(return_value=async_response)

    return llm


@pytest.fixture
def sample_messages() -> List:
    """Provide sample conversation messages."""
    return [
        HumanMessage(content="Design a ring buffer for ASIL C logging"),
        AIMessage(content="I'll design a lock-free ring buffer..."),
        HumanMessage(content="What about the DMA configuration?"),
    ]


@pytest.fixture
def initial_state(hardware_constraints) -> AgentState:
    """Provide an initial agent state for workflow tests."""
    return create_initial_state(
        messages=[HumanMessage(content="Design a logging service")],
        hardware_constraints=hardware_constraints,
        max_iterations=5,
    )


@pytest.fixture
def state_with_results(hardware_constraints) -> AgentState:
    """Provide a state with worker results for synthesis tests."""
    from solar_flare.graph.state import WorkerResult

    return AgentState(
        messages=[HumanMessage(content="Design a logging service")],
        current_request=None,
        hardware_constraints=hardware_constraints,
        orchestrator_decision=None,
        worker_results=[
            WorkerResult(
                agent_name="embedded_designer",
                task_type="embedded_architecture_design",
                status="success",
                findings=[
                    {
                        "type": "performance",
                        "severity": "low",
                        "description": "Design meets constraints",
                    }
                ],
                recommendations=["Validate on target hardware"],
                artifacts={"performance": {"cpu_overhead_percent": 2.5}},
                confidence_score=0.9,
            )
        ],
        design_review=None,
        final_response=None,
        iteration_count=1,
        max_iterations=5,
        current_phase="executing",
        errors=[],
    )
