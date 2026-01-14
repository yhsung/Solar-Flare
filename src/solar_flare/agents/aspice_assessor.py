"""
ASPICE Process Assessor Agent.

Specialized worker for ASPICE (Automotive SPICE) process compliance
assessment and improvement recommendations.
"""

from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from solar_flare.agents.base import BaseWorkerAgent, AgentRegistry
from solar_flare.graph.state import WorkerResult, HardwareConstraints, CapabilityLevel
from solar_flare.prompts.aspice import (
    ASPICE_SYSTEM_PROMPT,
    ASPICE_ASSESSMENT_PROMPT,
    ASPICE_OUTPUT_FORMAT,
)


@AgentRegistry.register("aspice_assessor")
class ASPICEAssessorAgent(BaseWorkerAgent):
    """
    ASPICE process assessment expert.

    Evaluates development processes against ASPICE 3.1 and provides
    improvement recommendations for achieving target capability levels.

    Capabilities:
    - Process assessment against ASPICE 3.1 PRM/PAM
    - Capability level determination (0-5)
    - Base practices and generic practices evaluation
    - Work product analysis
    - Process improvement roadmaps
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        hardware_constraints: HardwareConstraints,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            agent_name="aspice_assessor",
            hardware_constraints=hardware_constraints,
        )

    @property
    def system_prompt(self) -> str:
        return ASPICE_SYSTEM_PROMPT

    def _build_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("system", self.format_constraints_reminder()),
                ("system", ASPICE_OUTPUT_FORMAT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", ASPICE_ASSESSMENT_PROMPT),
            ]
        )

    async def execute(
        self,
        task_context: Dict[str, Any],
        messages: List[BaseMessage],
    ) -> WorkerResult:
        """
        Execute ASPICE process assessment.

        Args:
            task_context: Contains 'process_areas', 'current_level', 'target_level'
            messages: Conversation history for context

        Returns:
            WorkerResult with assessment findings and improvement roadmap
        """
        # Extract context with defaults
        process_areas = task_context.get("process_areas", ["SWE.1", "SWE.2", "SWE.3"])
        current_level = task_context.get("current_level", 1)
        target_level = task_context.get("target_level", 2)
        current_practices = task_context.get("current_practices", "")
        focus_areas = task_context.get("focus_areas", "")

        # Build practices description if not provided
        if not current_practices:
            current_practices = self._build_default_practices_description()

        # Build focus areas if not provided
        if not focus_areas:
            focus_areas = self._build_default_focus_areas(process_areas, target_level)

        # Execute the assessment chain
        prompt = self._prompt_template
        chain = prompt | self.llm

        try:
            response = await chain.ainvoke(
                {
                    "messages": messages,
                    "process_areas": ", ".join(process_areas),
                    "current_level": current_level,
                    "target_level": target_level,
                    "current_practices": current_practices,
                    "focus_areas": focus_areas,
                }
            )

            # Generate structured assessment
            findings = self._generate_assessment_findings(
                process_areas, current_level, target_level
            )
            recommendations = self._generate_recommendations(
                process_areas, current_level, target_level
            )

            return WorkerResult(
                agent_name=self.agent_name,
                task_type="aspice_process_assessment",
                status="success",
                findings=findings,
                recommendations=recommendations,
                artifacts={
                    "process_areas": process_areas,
                    "current_level": current_level,
                    "target_level": target_level,
                    "capability_gap": target_level - current_level,
                    "full_assessment": response.content,
                },
                confidence_score=0.85,
            )

        except Exception as e:
            return self._create_error_result(
                f"ASPICE assessment failed: {str(e)}",
                "aspice_process_assessment",
            )

    def _build_default_practices_description(self) -> str:
        """Build default description of current practices."""
        return """Current development practices include:
- Requirements documented in specification documents
- Architectural design captured in design documents
- Source code developed with coding guidelines
- Unit tests performed by developers
- Integration testing at system level
- Version control using Git
- Issue tracking system in use
- Reviews performed informally"""

    def _build_default_focus_areas(
        self, process_areas: List[str], target_level: int
    ) -> str:
        """Build focus areas based on process areas and target level."""
        areas = []

        if "SWE.1" in process_areas:
            areas.append("Requirements specification completeness and structure")
            areas.append("Requirements traceability establishment")

        if "SWE.2" in process_areas:
            areas.append("Architectural design documentation quality")
            areas.append("Interface specifications between components")

        if "SWE.3" in process_areas:
            areas.append("Detailed design documentation")
            areas.append("Unit construction standards")

        if target_level >= 2:
            areas.append("Process planning and monitoring practices")
            areas.append("Work product management procedures")

        if target_level >= 3:
            areas.append("Standard process definition")
            areas.append("Process tailoring guidelines")

        return "\n".join(f"- {area}" for area in areas)

    def _generate_assessment_findings(
        self, process_areas: List[str], current_level: int, target_level: int
    ) -> List[Dict[str, Any]]:
        """Generate assessment findings based on context."""
        findings = []

        # Capability level finding
        gap = target_level - current_level
        if gap > 0:
            findings.append(
                {
                    "type": "capability_gap",
                    "severity": "high" if gap > 1 else "medium",
                    "description": f"Capability gap of {gap} level(s) to reach target Level {target_level}",
                    "process_areas": process_areas,
                }
            )

        # Work product findings
        work_product_gaps = self._identify_work_product_gaps(process_areas, target_level)
        for gap in work_product_gaps:
            findings.append(
                {
                    "type": "work_product_gap",
                    "severity": "medium",
                    "description": gap,
                }
            )

        # Process practice findings
        if target_level >= 2 and current_level < 2:
            findings.append(
                {
                    "type": "generic_practice_gap",
                    "severity": "high",
                    "description": "Level 2 requires formal process planning and work product management",
                }
            )

        return findings

    def _identify_work_product_gaps(
        self, process_areas: List[str], target_level: int
    ) -> List[str]:
        """Identify common work product gaps."""
        gaps = []

        # Common gaps for logging service development
        gaps.append("Bidirectional traceability matrix may need formalization")
        gaps.append("Architecture decision rationale documentation often missing")

        if target_level >= 2:
            gaps.append("Review records should be formally maintained")
            gaps.append("Process plans may need to be documented")

        return gaps

    def _generate_recommendations(
        self, process_areas: List[str], current_level: int, target_level: int
    ) -> List[str]:
        """Generate improvement recommendations."""
        recs = []

        # Always recommend traceability
        recs.append(
            "Establish bidirectional traceability from requirements to design to test cases"
        )

        # Level-specific recommendations
        if target_level >= 2 and current_level < 2:
            recs.extend(
                [
                    "Define and document process plans for each process area",
                    "Implement work product management procedures",
                    "Establish formal review and approval workflows",
                    "Implement metrics for process monitoring",
                ]
            )

        if target_level >= 3 and current_level < 3:
            recs.extend(
                [
                    "Define organization-wide standard processes",
                    "Create process tailoring guidelines",
                    "Establish process asset library",
                ]
            )

        # Process area specific
        if "SWE.1" in process_areas:
            recs.append("Structure requirements with unique IDs and attributes")

        if "SWE.2" in process_areas:
            recs.append("Document interface specifications between components")

        return recs
