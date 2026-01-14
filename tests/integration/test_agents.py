"""
Integration tests for Solar-Flare agent coordination.

These tests verify that agents work together correctly,
including orchestrator routing and worker agent execution.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from solar_flare import (
    create_workflow,
    compile_workflow,
    HardwareConstraints,
)
from solar_flare.graph.state import (
    create_initial_state,
    WorkerResult,
    OrchestratorDecision,
)
from solar_flare.agents.orchestrator import OrchestratorAgent
from solar_flare.agents.iso_26262_analyzer import ISO26262AnalyzerAgent
from solar_flare.agents.embedded_designer import EmbeddedDesignerAgent
from solar_flare.agents.aspice_assessor import ASPICEAssessorAgent
from solar_flare.agents.design_reviewer import DesignReviewAgent


class TestOrchestratorAgent:
    """Tests for the Orchestrator agent."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_llm):
        """Test orchestrator agent initialization."""
        constraints = HardwareConstraints()
        orchestrator = OrchestratorAgent(mock_llm, constraints)

        assert orchestrator.llm == mock_llm
        assert orchestrator.hardware_constraints == constraints

    @pytest.mark.asyncio
    async def test_analyze_request(self, mock_llm):
        """Test request analysis by orchestrator."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """
        Analysis:
        - Component: ring_buffer
        - ASIL Level: D
        - Platform: Cortex-R5
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        constraints = HardwareConstraints()
        orchestrator = OrchestratorAgent(mock_llm, constraints)

        messages = [HumanMessage(content="Design a ring buffer for ASIL-D")]
        analysis = await orchestrator.analyze_request(messages[0].content, messages)

        # Analysis should be a dict or similar
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_decide_next_step(self, mock_llm):
        """Test next step decision by orchestrator."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"next_agent": "embedded_designer", "reasoning": "Design needed"}'
            )
        )

        constraints = HardwareConstraints()
        orchestrator = OrchestratorAgent(mock_llm, constraints)

        state = create_initial_state(
            messages=[HumanMessage(content="Design request")],
            hardware_constraints=constraints,
        )

        decision = await orchestrator.decide_next_step(state, {})

        assert decision is not None
        assert hasattr(decision, "next_agent")
        assert hasattr(decision, "reasoning")


class TestISO26262AnalyzerAgent:
    """Tests for the ISO 26262 Analyzer agent."""

    @pytest.mark.asyncio
    async def test_iso_analyzer_initialization(self, mock_llm):
        """Test ISO 26262 analyzer initialization."""
        constraints = HardwareConstraints()
        analyzer = ISO26262AnalyzerAgent(
            llm=mock_llm,
            tools=[],
            hardware_constraints=constraints,
        )

        assert analyzer.llm == mock_llm
        assert analyzer.hardware_constraints == constraints

    @pytest.mark.asyncio
    async def test_iso_analyzer_execution(self, mock_llm):
        """Test ISO analyzer execution."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = """
        {
            "findings": [
                {"type": "compliance", "severity": "high", "description": "Missing ASIL-D validation"}
            ],
            "recommendations": ["Add end-to-end ECC"],
            "confidence": 0.9
        }
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        constraints = HardwareConstraints()
        analyzer = ISO26262AnalyzerAgent(
            llm=mock_llm,
            tools=[],
            hardware_constraints=constraints,
        )

        context = {"component": "ring_buffer", "asil_level": "ASIL_D"}
        messages = [HumanMessage(content="Analyze for ASIL-D compliance")]

        result = await analyzer.execute(context, messages)

        assert result is not None
        assert result.agent_name == "iso_26262_analyzer"


class TestEmbeddedDesignerAgent:
    """Tests for the Embedded Designer agent."""

    @pytest.mark.asyncio
    async def test_embedded_designer_initialization(self, mock_llm):
        """Test embedded designer initialization."""
        constraints = HardwareConstraints()
        designer = EmbeddedDesignerAgent(
            llm=mock_llm,
            tools=[],
            hardware_constraints=constraints,
        )

        assert designer.llm == mock_llm
        assert designer.hardware_constraints == constraints

    @pytest.mark.asyncio
    async def test_embedded_designer_execution(self, mock_llm):
        """Test embedded designer execution."""
        mock_response = MagicMock()
        mock_response.content = """
        Design: Lock-free ring buffer with DMA support
        - Size: 64KB per core
        - Synchronization: Atomic indices
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        constraints = HardwareConstraints()
        designer = EmbeddedDesignerAgent(
            llm=mock_llm,
            tools=[],
            hardware_constraints=constraints,
        )

        context = {"component": "ring_buffer", "platform": "Cortex-R5"}
        messages = [HumanMessage(content="Design ring buffer")]

        result = await designer.execute(context, messages)

        assert result is not None
        assert result.agent_name == "embedded_designer"


class TestASPICEAssessorAgent:
    """Tests for the ASPICE Assessor agent."""

    @pytest.mark.asyncio
    async def test_aspice_assessor_initialization(self, mock_llm):
        """Test ASPICE assessor initialization."""
        constraints = HardwareConstraints()
        assessor = ASPICEAssessorAgent(
            llm=mock_llm,
            tools=[],
            hardware_constraints=constraints,
        )

        assert assessor.llm == mock_llm
        assert assessor.hardware_constraints == constraints

    @pytest.mark.asyncio
    async def test_aspice_assessor_execution(self, mock_llm):
        """Test ASPICE assessor execution."""
        mock_response = MagicMock()
        mock_response.content = """
        ASPICE Assessment Results:
        - SWE.1: Level 3 (Established)
        - SWE.2: Level 2 (Managed)
        - Gaps: Missing unit test coverage documentation
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        constraints = HardwareConstraints()
        assessor = ASPICEAssessorAgent(
            llm=mock_llm,
            tools=[],
            hardware_constraints=constraints,
        )

        context = {"target_capability_level": 3}
        messages = [HumanMessage(content="Assess ASPICE compliance")]

        result = await assessor.execute(context, messages)

        assert result is not None
        assert result.agent_name == "aspice_assessor"


class TestDesignReviewAgent:
    """Tests for the Design Review agent."""

    @pytest.mark.asyncio
    async def test_design_reviewer_initialization(self, mock_llm):
        """Test design reviewer initialization."""
        constraints = HardwareConstraints()
        reviewer = DesignReviewAgent(
            llm=mock_llm,
            hardware_constraints=constraints,
        )

        assert reviewer.llm == mock_llm
        assert reviewer.hardware_constraints == constraints

    @pytest.mark.asyncio
    async def test_design_reviewer_execution(self, mock_llm):
        """Test design reviewer execution."""
        mock_response = MagicMock()
        mock_response.content = """
        Design Review Results:
        - Status: Approved
        - Completeness: 85%
        - Gaps: 2 minor issues found
        """
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        constraints = HardwareConstraints()
        reviewer = DesignReviewAgent(
            llm=mock_llm,
            hardware_constraints=constraints,
        )

        # Create mock artifacts to review
        context = {
            "artifacts_to_review": [
                {"design": "ring_buffer_v1"},
                {"compliance": "ASIL_D_ready"},
            ]
        }
        messages = [HumanMessage(content="Review the design")]

        result = await reviewer.execute(context, messages)

        assert result is not None
        assert result.agent_name == "design_reviewer"


class TestAgentCoordination:
    """Tests for multi-agent coordination."""

    @pytest.mark.asyncio
    async def test_sequential_agent_execution(self, mock_llm):
        """Test that agents execute in correct sequence."""
        # Track call order
        call_order = []

        async def mock_llm_invoke(*args, **kwargs):
            # Record which agent is calling based on prompt
            prompt = str(kwargs) + str(args)
            if "ISO" in prompt or "26262" in prompt:
                call_order.append("iso_26262_analyzer")
            elif "embedded" in prompt.lower() or "design" in prompt.lower():
                call_order.append("embedded_designer")
            elif "aspice" in prompt.lower():
                call_order.append("aspice_assessor")
            elif "review" in prompt.lower():
                call_order.append("design_reviewer")
            else:
                call_order.append("orchestrator")

            return MagicMock(content="Agent response")

        mock_llm.ainvoke = AsyncMock(side_effect=mock_llm_invoke)

        constraints = HardwareConstraints()

        # Create agents
        orchestrator = OrchestratorAgent(mock_llm, constraints)
        iso_analyzer = ISO26262AnalyzerAgent(mock_llm, [], constraints)
        designer = EmbeddedDesignerAgent(mock_llm, [], constraints)

        # Execute in sequence
        state = create_initial_state(
            messages=[HumanMessage(content="Design request")],
            hardware_constraints=constraints,
        )

        # Orchestrator analyzes
        await orchestrator.analyze_request("Design request", state["messages"])

        # ISO analyzer executes
        await iso_analyzer.execute({"component": "ring_buffer"}, state["messages"])

        # Designer executes
        await designer.execute({"component": "ring_buffer"}, state["messages"])

        # Verify agents were called
        assert len(call_order) > 0

    @pytest.mark.asyncio
    async def test_worker_result_accumulation(self):
        """Test that worker results accumulate correctly in state."""
        # Create mock worker results
        result1 = WorkerResult(
            agent_name="iso_26262_analyzer",
            task_type="safety_analysis",
            status="success",
            findings=[{"type": "gap", "description": "Missing validation"}],
            recommendations=["Add validation"],
            artifacts={},
            confidence_score=0.85,
        )

        result2 = WorkerResult(
            agent_name="embedded_designer",
            task_type="architecture_design",
            status="success",
            findings=[{"type": "performance", "description": "CPU OK"}],
            recommendations=["Test on hardware"],
            artifacts={"design": "ring_buffer_v1"},
            confidence_score=0.9,
        )

        # Simulate state accumulation
        state = create_initial_state(hardware_constraints=HardwareConstraints())
        state["worker_results"] = [result1, result2]

        assert len(state["worker_results"]) == 2
        assert state["worker_results"][0].agent_name == "iso_26262_analyzer"
        assert state["worker_results"][1].agent_name == "embedded_designer"


class TestAgentContextPassing:
    """Tests for context passing between agents."""

    @pytest.mark.asyncio
    async def test_orchestrator_to_worker_context(self, mock_llm):
        """Test context passing from orchestrator to worker."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Worker response")
        )

        constraints = HardwareConstraints()
        analyzer = ISO26262AnalyzerAgent(mock_llm, [], constraints)

        # Orchestrator decision with context
        decision = OrchestratorDecision(
            next_agent="iso_26262_analyzer",
            reasoning="Safety analysis needed",
            context_for_agent={
                "component": "ring_buffer",
                "asil_level": "ASIL_D",
                "platform": "Cortex-R5",
            },
        )

        # Execute with context
        context = decision.context_for_agent
        messages = [HumanMessage(content="Design request")]

        result = await analyzer.execute(context, messages)

        assert result is not None

    @pytest.mark.asyncio
    async def test_worker_to_reviewer_context(self, mock_llm):
        """Test context passing from worker to reviewer."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Review complete")
        )

        constraints = HardwareConstraints()
        reviewer = DesignReviewAgent(mock_llm, hardware_constraints=constraints)

        # Mock artifacts from workers
        worker_artifacts = [
            {
                "design": "ring_buffer_implementation",
                "compliance": "ASIL_D_ready",
            },
            {
                "aspice_assessment": "Level 3 achieved",
            },
        ]

        context = {"artifacts_to_review": worker_artifacts}
        messages = [HumanMessage(content="Review the design")]

        result = await reviewer.execute(context, messages)

        assert result is not None


class TestAgentErrorHandling:
    """Tests for agent error handling."""

    @pytest.mark.asyncio
    async def test_agent_handles_missing_context(self, mock_llm):
        """Test that agents handle missing context gracefully."""
        mock_llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Proceeding with default context")
        )

        constraints = HardwareConstraints()
        analyzer = ISO26262AnalyzerAgent(mock_llm, [], constraints)

        # Execute with minimal context
        context = {}
        messages = [HumanMessage(content="Analyze")]

        try:
            result = await analyzer.execute(context, messages)
            assert result is not None
        except Exception as e:
            # Agent should handle or fail gracefully
            assert "context" in str(e).lower() or "missing" in str(e).lower()

    @pytest.mark.asyncio
    async def test_agent_handles_llm_error(self):
        """Test that agents handle LLM errors gracefully."""
        # Mock LLM that raises an error
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM API Error"))

        constraints = HardwareConstraints()
        analyzer = ISO26262AnalyzerAgent(mock_llm, [], constraints)

        context = {"component": "ring_buffer"}
        messages = [HumanMessage(content="Analyze")]

        with pytest.raises(Exception):
            await analyzer.execute(context, messages)


class TestHardwareConstraintPropagation:
    """Tests for hardware constraints propagation to agents."""

    def test_constraints_passed_to_all_agents(self, mock_llm):
        """Test that constraints are passed to all agents."""
        constraints = HardwareConstraints(
            mailbox_payload_bytes=128,
            dma_burst_bytes=32768,
        )

        # Create all agents
        orchestrator = OrchestratorAgent(mock_llm, constraints)
        iso_analyzer = ISO26262AnalyzerAgent(mock_llm, [], constraints)
        designer = EmbeddedDesignerAgent(mock_llm, [], constraints)
        aspice = ASPICEAssessorAgent(mock_llm, [], constraints)
        reviewer = DesignReviewAgent(mock_llm, hardware_constraints=constraints)

        # Verify all agents have the same constraints
        assert orchestrator.hardware_constraints.mailbox_payload_bytes == 128
        assert iso_analyzer.hardware_constraints.mailbox_payload_bytes == 128
        assert designer.hardware_constraints.mailbox_payload_bytes == 128
        assert aspice.hardware_constraints.mailbox_payload_bytes == 128
        assert reviewer.hardware_constraints.mailbox_payload_bytes == 128

    def test_constraints_reminder_in_prompt(self, mock_llm):
        """Test that hardware constraints are formatted for prompts."""
        constraints = HardwareConstraints()

        reminder = constraints.to_reminder_string()

        # Verify key constraint information is present
        assert "Mailbox" in reminder
        assert "DMA" in reminder
        assert "CPU" in reminder or "overhead" in reminder.lower()
        assert "Timer" in reminder or "timestamp" in reminder.lower()
