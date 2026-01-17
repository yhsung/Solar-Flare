"""
LangGraph state definitions for the Solar-Flare multi-agent workflow.

This module defines the state schema that flows through all nodes in the
workflow graph, accumulating results from worker agents.
"""

from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from operator import add
from enum import Enum

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ASILLevel(str, Enum):
    """ISO 26262 ASIL (Automotive Safety Integrity Level) classifications."""

    QM = "QM"  # Quality Management (no safety requirements)
    ASIL_A = "ASIL_A"  # Lowest safety integrity
    ASIL_B = "ASIL_B"
    ASIL_C = "ASIL_C"
    ASIL_D = "ASIL_D"  # Highest safety integrity


class CapabilityLevel(int, Enum):
    """ASPICE capability levels for process assessment."""

    INCOMPLETE = 0  # Process not implemented or fails to achieve purpose
    PERFORMED = 1  # Process achieves its purpose
    MANAGED = 2  # Process is managed (planned, monitored, adjusted)
    ESTABLISHED = 3  # Process uses defined process
    PREDICTABLE = 4  # Process operates within defined limits
    INNOVATING = 5  # Process is continuously improved


class HardwareConstraints(BaseModel):
    """
    Mandatory hardware constraints for all logging service designs.

    These constraints are NON-NEGOTIABLE and must be validated in all designs.
    They are derived from the Solar-Flare project's AGENTS.md specification.
    """

    mailbox_payload_bytes: int = Field(
        default=64,
        description="Mailbox payload size for control signaling and descriptors",
    )
    dma_burst_bytes: int = Field(
        default=65536,  # 64 KB
        description="DMA burst size for zero-copy logging",
    )
    timestamp_resolution_ns: int = Field(
        default=1,
        description="Global Hardware System Timer resolution in nanoseconds",
    )
    timestamp_bits: int = Field(
        default=64,
        description="Timestamp bit width for log entries",
    )
    max_cpu_overhead_percent: float = Field(
        default=3.0,
        description="Maximum CPU overhead per core for logging service",
    )
    max_bandwidth_mbps: float = Field(
        default=10.0,
        description="Maximum aggregate bandwidth for logging",
    )

    def to_reminder_string(self) -> str:
        """Format constraints as a reminder string for agent prompts."""
        return f"""## Mandatory Hardware Constraints (MUST be validated):
- Transport: Interrupt-driven Mailbox with {self.mailbox_payload_bytes}-byte payload
- Data Movement: DMA with {self.dma_burst_bytes // 1024} KB per burst (zero-copy)
- Synchronization: Global HW Timer ({self.timestamp_resolution_ns}ns resolution, {self.timestamp_bits}-bit timestamp)
- Performance Budget: ≤{self.max_cpu_overhead_percent}% CPU per core, ≤{self.max_bandwidth_mbps} MB/s aggregate
- Memory: Per-core local fixed-size Ring Buffer
- Overflow Policies: LOG_POLICY_OVERWRITE (continuous) or LOG_POLICY_STOP (post-mortem)"""


class DesignRequest(BaseModel):
    """User's design request with parsed parameters."""

    component: str = Field(
        description="The logging component being designed (e.g., ring_buffer, dma_controller)"
    )
    asil_level: Optional[ASILLevel] = Field(
        default=None,
        description="ISO 26262 ASIL level for the design",
    )
    target_capability_level: Optional[CapabilityLevel] = Field(
        default=None,
        description="Target ASPICE capability level",
    )
    platform: Optional[str] = Field(
        default=None,
        description="Target MCU platform (e.g., Cortex-R5, AURIX TC397, RH850)",
    )
    rtos: Optional[str] = Field(
        default=None,
        description="Target RTOS (e.g., FreeRTOS, AUTOSAR OS)",
    )
    additional_constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional project-specific constraints",
    )


class WorkerResult(BaseModel):
    """Result returned by a worker agent after completing its task."""

    agent_name: str = Field(description="Name of the agent that produced this result")
    task_type: str = Field(
        description="Type of task performed (design, analysis, assessment, review)"
    )
    status: Literal["success", "partial", "failed"] = Field(
        description="Overall status of the task execution"
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of findings (issues, gaps, observations)",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations",
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated artifacts (designs, reports, code)",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent's confidence in the result (0-1)",
    )

    def to_markdown(self) -> str:
        """Format this result as a markdown document."""
        from solar_flare.markdown_export import format_worker_result_markdown
        return format_worker_result_markdown(self)


class DesignReviewResult(BaseModel):
    """Result from the Design Review Agent's cross-validation analysis."""

    overall_status: Literal["approved", "needs_revision", "rejected"] = Field(
        description="Overall assessment of the design"
    )
    completeness_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of required elements present (0-100)",
    )
    gaps_identified: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of gaps with severity and description",
    )
    cross_reference_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Issues found when cross-referencing against standards",
    )
    constraint_violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Hardware constraint violations",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Prioritized recommendations for improvement",
    )

    def to_markdown(self) -> str:
        """Format this review as a markdown document."""
        from solar_flare.markdown_export import format_design_review_markdown
        return format_design_review_markdown(self)


class OrchestratorDecision(BaseModel):
    """Decision made by the orchestrator for routing to the next agent."""

    next_agent: Literal[
        "iso_26262_analyzer",
        "embedded_designer",
        "aspice_assessor",
        "design_reviewer",
        "synthesize",
        "end",
    ] = Field(description="The next agent to route to")
    reasoning: str = Field(description="Explanation for the routing decision")
    context_for_agent: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context to pass to the next agent",
    )


class AgentState(TypedDict):
    """
    Main state schema for the Solar-Flare workflow graph.

    This state flows through all nodes in the graph and accumulates
    results from worker agents. Uses LangGraph's Annotated types for
    proper state accumulation.

    Attributes:
        messages: Conversation history (accumulates with each turn)
        current_request: Parsed user request
        hardware_constraints: Immutable hardware constraints reference
        orchestrator_decision: Current routing decision
        worker_results: Results from completed workers (accumulates)
        design_review: Result from design review agent
        final_response: Synthesized final response
        iteration_count: Current iteration in the workflow
        max_iterations: Maximum iterations before forced synthesis
        current_phase: Current phase of the workflow
        errors: Accumulated errors during execution
    """

    # Conversation history (accumulates with each turn)
    messages: Annotated[List[BaseMessage], add]

    # Current user request (parsed)
    current_request: Optional[DesignRequest]

    # Hardware constraints (immutable reference)
    hardware_constraints: HardwareConstraints

    # Orchestrator routing decisions
    orchestrator_decision: Optional[OrchestratorDecision]

    # Worker results (accumulates as workers complete)
    worker_results: Annotated[List[WorkerResult], add]

    # Design review result
    design_review: Optional[DesignReviewResult]

    # Final synthesized response
    final_response: Optional[str]

    # Workflow metadata
    iteration_count: int
    max_iterations: int
    current_phase: Literal[
        "understanding",
        "delegating",
        "executing",
        "reviewing",
        "synthesizing",
        "complete",
    ]

    # Error tracking
    errors: Annotated[List[Dict[str, Any]], add]


def create_initial_state(
    messages: Optional[List[BaseMessage]] = None,
    hardware_constraints: Optional[HardwareConstraints] = None,
    max_iterations: int = 10,
) -> AgentState:
    """
    Create an initial state for the workflow.

    Args:
        messages: Initial conversation messages
        hardware_constraints: Custom hardware constraints (uses defaults if None)
        max_iterations: Maximum workflow iterations

    Returns:
        Initialized AgentState ready for workflow execution
    """
    return AgentState(
        messages=messages or [],
        current_request=None,
        hardware_constraints=hardware_constraints or HardwareConstraints(),
        orchestrator_decision=None,
        worker_results=[],
        design_review=None,
        final_response=None,
        iteration_count=0,
        max_iterations=max_iterations,
        current_phase="understanding",
        errors=[],
    )
