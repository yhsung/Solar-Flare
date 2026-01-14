"""
Main orchestrator agent that coordinates worker sub-agents.

This agent analyzes user requests, routes to appropriate workers,
and synthesizes results into cohesive responses.
"""

from typing import List, Dict, Any, Optional, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from solar_flare.graph.state import (
    AgentState,
    OrchestratorDecision,
    DesignRequest,
    HardwareConstraints,
    ASILLevel,
    WorkerResult,
)
from solar_flare.prompts.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    REQUEST_ANALYSIS_PROMPT,
    ROUTING_DECISION_PROMPT,
    SYNTHESIS_PROMPT,
)


class RequestAnalysis(BaseModel):
    """Analysis of the user's request for routing decisions."""

    request_type: Literal[
        "architectural_design",
        "compliance_analysis",
        "implementation_support",
        "process_assessment",
        "design_review",
        "general_query",
    ] = Field(description="Type of request being made")

    components_involved: List[str] = Field(
        default_factory=list,
        description="Logging components involved (ring_buffer, dma, isr, etc.)",
    )

    asil_level: Optional[ASILLevel] = Field(
        default=None,
        description="ISO 26262 ASIL level if specified",
    )

    requires_iso_26262: bool = Field(
        default=False,
        description="Whether ISO 26262 analysis is needed",
    )

    requires_aspice: bool = Field(
        default=False,
        description="Whether ASPICE assessment is needed",
    )

    requires_embedded_design: bool = Field(
        default=False,
        description="Whether embedded architecture design is needed",
    )

    requires_review: bool = Field(
        default=False,
        description="Whether design review is needed",
    )


class OrchestratorAgent:
    """
    Main supervisor agent that coordinates the multi-agent workflow.

    Responsibilities:
    1. Analyze user requests to understand requirements
    2. Route to appropriate worker agents
    3. Coordinate parallel execution when possible
    4. Synthesize final responses from worker results

    Attributes:
        llm: Language model for reasoning
        hardware_constraints: Reference to mandatory constraints
    """

    def __init__(
        self,
        llm: BaseChatModel,
        hardware_constraints: HardwareConstraints,
    ):
        """
        Initialize the orchestrator agent.

        Args:
            llm: Language model for orchestration decisions
            hardware_constraints: Mandatory hardware constraints
        """
        self.llm = llm
        self.hardware_constraints = hardware_constraints
        self.request_parser = PydanticOutputParser(pydantic_object=RequestAnalysis)
        self.decision_parser = PydanticOutputParser(pydantic_object=OrchestratorDecision)

    def _build_analysis_prompt(self) -> ChatPromptTemplate:
        """Build prompt for analyzing user requests."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", ORCHESTRATOR_SYSTEM_PROMPT),
                ("system", self.hardware_constraints.to_reminder_string()),
                ("system", REQUEST_ANALYSIS_PROMPT),
                ("system", "{format_instructions}"),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{input}"),
            ]
        )

    def _build_routing_prompt(self) -> ChatPromptTemplate:
        """Build prompt for routing decisions."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", ORCHESTRATOR_SYSTEM_PROMPT),
                ("system", ROUTING_DECISION_PROMPT),
                ("system", "{format_instructions}"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def _build_synthesis_prompt(self) -> ChatPromptTemplate:
        """Build prompt for synthesizing final response."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", ORCHESTRATOR_SYSTEM_PROMPT),
                ("system", self.hardware_constraints.to_reminder_string()),
                ("human", SYNTHESIS_PROMPT),
            ]
        )

    async def analyze_request(
        self,
        user_input: str,
        messages: List[BaseMessage],
    ) -> RequestAnalysis:
        """
        Analyze the user's request to determine required workers.

        Args:
            user_input: The user's current message
            messages: Conversation history

        Returns:
            RequestAnalysis with categorization and requirements
        """
        prompt = self._build_analysis_prompt()
        chain = prompt | self.llm | self.request_parser

        try:
            return await chain.ainvoke(
                {
                    "messages": messages,
                    "input": user_input,
                    "format_instructions": self.request_parser.get_format_instructions(),
                }
            )
        except Exception as e:
            # Fallback to basic analysis
            return RequestAnalysis(
                request_type="general_query",
                components_involved=[],
                requires_embedded_design=True,  # Default to design support
            )

    async def decide_next_step(
        self,
        state: AgentState,
        analysis: RequestAnalysis,
    ) -> OrchestratorDecision:
        """
        Decide which agent to route to next.

        Implements the delegation strategy from AGENTS.md:
        - Use iso_26262_analyzer for safety compliance
        - Use embedded_designer for component design
        - Use aspice_assessor for process assessment
        - Use design_reviewer for cross-validation

        Args:
            state: Current workflow state
            analysis: Request analysis result

        Returns:
            OrchestratorDecision with next agent and context
        """
        # Get completed agents
        completed_agents = set(r.agent_name for r in state.get("worker_results", []))

        # Check iteration limit
        if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
            return OrchestratorDecision(
                next_agent="synthesize",
                reasoning="Maximum iterations reached, synthesizing available results",
                context_for_agent={},
            )

        # Determine next agent based on requirements and completion status
        # Priority: embedded_design -> iso_26262 -> aspice -> design_review -> synthesize

        if analysis.requires_embedded_design and "embedded_designer" not in completed_agents:
            return OrchestratorDecision(
                next_agent="embedded_designer",
                reasoning="User needs component design or implementation support",
                context_for_agent={
                    "components": analysis.components_involved,
                    "asil_level": analysis.asil_level.value if analysis.asil_level else None,
                },
            )

        if analysis.requires_iso_26262 and "iso_26262_analyzer" not in completed_agents:
            return OrchestratorDecision(
                next_agent="iso_26262_analyzer",
                reasoning="User needs ISO 26262 compliance analysis",
                context_for_agent={
                    "asil_level": analysis.asil_level.value if analysis.asil_level else "ASIL_B",
                    "components": analysis.components_involved,
                },
            )

        if analysis.requires_aspice and "aspice_assessor" not in completed_agents:
            return OrchestratorDecision(
                next_agent="aspice_assessor",
                reasoning="User needs ASPICE process assessment",
                context_for_agent={
                    "process_areas": ["SWE.1", "SWE.2", "SWE.3"],
                    "target_level": 2,
                },
            )

        # Run design review if there are results to review
        worker_results = state.get("worker_results", [])
        if worker_results and "design_reviewer" not in completed_agents:
            return OrchestratorDecision(
                next_agent="design_reviewer",
                reasoning="Review accumulated results for completeness and compliance",
                context_for_agent={
                    "artifacts_to_review": [r.artifacts for r in worker_results],
                    "asil_level": analysis.asil_level.value if analysis.asil_level else None,
                },
            )

        # All workers done or not needed, synthesize
        return OrchestratorDecision(
            next_agent="synthesize",
            reasoning="All required analyses complete, ready to synthesize final response",
            context_for_agent={},
        )

    async def synthesize_response(
        self,
        state: AgentState,
    ) -> str:
        """
        Synthesize final response from all worker results.

        Implements the synthesis strategy from AGENTS.md:
        1. Integrate findings from multiple workers
        2. Resolve conflicts (safety > performance > process)
        3. Prioritize recommendations
        4. Provide clear next steps

        Args:
            state: Complete workflow state with all results

        Returns:
            Synthesized response string
        """
        prompt = self._build_synthesis_prompt()
        chain = prompt | self.llm

        # Format worker results for synthesis
        worker_results = state.get("worker_results", [])
        results_text = self._format_worker_results(worker_results)

        # Format design review if present
        design_review = state.get("design_review")
        review_text = self._format_design_review(design_review) if design_review else "Not performed"

        response = await chain.ainvoke(
            {
                "worker_results": results_text,
                "design_review": review_text,
            }
        )

        return response.content

    def _format_worker_results(self, results: List[WorkerResult]) -> str:
        """Format worker results for synthesis prompt."""
        if not results:
            return "No worker results available."

        sections = []
        for result in results:
            section = f"""### {result.agent_name} ({result.task_type})
**Status**: {result.status}
**Confidence**: {result.confidence_score:.0%}

**Findings**:
{self._format_findings(result.findings)}

**Recommendations**:
{self._format_list(result.recommendations)}
"""
            sections.append(section)

        return "\n---\n".join(sections)

    def _format_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings list."""
        if not findings:
            return "- No findings"

        lines = []
        for f in findings:
            severity = f.get("severity", "medium")
            desc = f.get("description", str(f))
            lines.append(f"- [{severity.upper()}] {desc}")

        return "\n".join(lines)

    def _format_list(self, items: List[str]) -> str:
        """Format a list of strings."""
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items)

    def _format_design_review(self, review) -> str:
        """Format design review result."""
        return f"""**Status**: {review.overall_status}
**Completeness**: {review.completeness_score:.0f}%

**Gaps**: {len(review.gaps_identified)} identified
**Constraint Violations**: {len(review.constraint_violations)}

**Recommendations**:
{self._format_list(review.recommendations)}
"""
