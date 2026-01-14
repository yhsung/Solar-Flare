"""
ISO 26262 Compliance Analyzer Agent.

Specialized worker for deep ISO 26262 compliance analysis of logging
service components and architectures.
"""

from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from solar_flare.agents.base import BaseWorkerAgent, AgentRegistry
from solar_flare.graph.state import WorkerResult, HardwareConstraints
from solar_flare.prompts.iso_26262 import (
    ISO_26262_SYSTEM_PROMPT,
    ISO_26262_ANALYSIS_PROMPT,
    ISO_26262_OUTPUT_FORMAT,
)


@AgentRegistry.register("iso_26262_analyzer")
class ISO26262AnalyzerAgent(BaseWorkerAgent):
    """
    ISO 26262 functional safety compliance expert.

    Performs deep compliance analysis for logging service components,
    including ASIL-specific requirements, safety mechanisms, and gap analysis.

    Capabilities:
    - ASIL-specific analysis (QM, A, B, C, D)
    - Safety analysis artifacts (FMEA, FTA, safety requirements)
    - Compliance verification against ISO 26262-5 and 26262-6
    - Gap analysis and remediation recommendations
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
            agent_name="iso_26262_analyzer",
            hardware_constraints=hardware_constraints,
        )

    @property
    def system_prompt(self) -> str:
        return ISO_26262_SYSTEM_PROMPT

    def _build_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("system", self.format_constraints_reminder()),
                ("system", ISO_26262_OUTPUT_FORMAT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", ISO_26262_ANALYSIS_PROMPT),
            ]
        )

    async def execute(
        self,
        task_context: Dict[str, Any],
        messages: List[BaseMessage],
    ) -> WorkerResult:
        """
        Execute ISO 26262 compliance analysis.

        Args:
            task_context: Contains 'asil_level', 'components', and optional 'description'
            messages: Conversation history for context

        Returns:
            WorkerResult with compliance findings and recommendations
        """
        # Extract context
        asil_level = task_context.get("asil_level", "ASIL_B")
        components = task_context.get("components", ["logging_service"])
        description = task_context.get("description", "")
        focus_areas = task_context.get("focus_areas", "")

        # If no description provided, derive from components
        if not description:
            description = f"Logging service components: {', '.join(components)}"

        # If no focus areas, use defaults based on ASIL
        if not focus_areas:
            focus_areas = self._get_default_focus_areas(asil_level)

        # Build and execute the chain
        prompt = self._prompt_template
        chain = prompt | self.llm

        try:
            response = await chain.ainvoke(
                {
                    "messages": messages,
                    "component": ", ".join(components),
                    "asil_level": asil_level,
                    "description": description,
                    "focus_areas": focus_areas,
                }
            )

            # Parse response into structured findings
            findings, recommendations = self._parse_response(response.content, asil_level)

            return WorkerResult(
                agent_name=self.agent_name,
                task_type="iso_26262_compliance_analysis",
                status="success",
                findings=findings,
                recommendations=recommendations,
                artifacts={
                    "asil_level": asil_level,
                    "components_analyzed": components,
                    "full_analysis": response.content,
                },
                confidence_score=0.85,
            )

        except Exception as e:
            return self._create_error_result(
                f"ISO 26262 analysis failed: {str(e)}",
                "iso_26262_compliance_analysis",
            )

    def _get_default_focus_areas(self, asil_level: str) -> str:
        """Get default focus areas based on ASIL level."""
        base_areas = [
            "Safety mechanisms for logging components",
            "Error detection and handling",
            "Diagnostic coverage requirements",
        ]

        # Add ASIL-specific focus areas
        if asil_level in ["ASIL_C", "ASIL_D"]:
            base_areas.extend(
                [
                    "Formal verification requirements",
                    "Redundancy mechanisms",
                    "Independence requirements for safety functions",
                ]
            )

        if asil_level == "ASIL_D":
            base_areas.extend(
                [
                    "Hardware metrics (SPFM, LFM)",
                    "Fault injection testing requirements",
                ]
            )

        return "\n".join(f"- {area}" for area in base_areas)

    def _parse_response(
        self, response: str, asil_level: str
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Parse the LLM response to extract findings and recommendations.

        Args:
            response: Raw response from the LLM
            asil_level: ASIL level for context

        Returns:
            Tuple of (findings list, recommendations list)
        """
        findings = []
        recommendations = []

        # Add structured findings based on ASIL requirements
        # These are always relevant for logging service designs

        # Safety mechanism findings
        if "safety mechanism" in response.lower() or "error detection" in response.lower():
            findings.append(
                {
                    "id": "ISO-001",
                    "type": "safety_mechanism",
                    "severity": "high" if asil_level in ["ASIL_C", "ASIL_D"] else "medium",
                    "description": "Safety mechanisms must be specified for the logging service",
                    "iso_reference": "ISO 26262-6, §7.4.7",
                }
            )

        # Diagnostic coverage finding
        findings.append(
            {
                "id": "ISO-002",
                "type": "diagnostic_coverage",
                "severity": "high" if asil_level in ["ASIL_C", "ASIL_D"] else "medium",
                "description": f"Diagnostic coverage must meet {asil_level} requirements",
                "iso_reference": "ISO 26262-5, §8",
            }
        )

        # Traceability finding
        findings.append(
            {
                "id": "ISO-003",
                "type": "traceability",
                "severity": "high",
                "description": "Bidirectional traceability required from safety goals to verification",
                "iso_reference": "ISO 26262-6, §7.4.1",
            }
        )

        # Generate recommendations
        recommendations = [
            f"Implement safety mechanisms appropriate for {asil_level}",
            "Establish bidirectional traceability matrix",
            "Define verification methods for all safety requirements",
            "Document diagnostic coverage calculations",
        ]

        if asil_level in ["ASIL_C", "ASIL_D"]:
            recommendations.append("Consider formal methods for critical safety functions")
            recommendations.append("Implement redundancy for safety-critical logging paths")

        return findings, recommendations
