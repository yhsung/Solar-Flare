"""
Design Review Agent.

NEW specialized worker that cross-validates designs against ISO 26262
and ASPICE requirements, identifies gaps, and ensures completeness.
"""

from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

from solar_flare.agents.base import BaseWorkerAgent, AgentRegistry
from solar_flare.graph.state import WorkerResult, DesignReviewResult, HardwareConstraints
from solar_flare.tools.analysis_tools import (
    validate_hardware_constraints,
    cross_reference_requirements,
)
from solar_flare.prompts.design_review import (
    DESIGN_REVIEW_SYSTEM_PROMPT,
    DESIGN_REVIEW_PROMPT,
    DESIGN_REVIEW_OUTPUT_FORMAT,
)


@AgentRegistry.register("design_reviewer")
class DesignReviewAgent(BaseWorkerAgent):
    """
    Design Review Specialist for automotive safety-critical logging systems.

    This is a NEW agent that provides:
    - Cross-validation of designs against ISO 26262 and ASPICE
    - Gap analysis for missing components
    - Hardware constraint validation
    - Consistency checking across artifacts

    Unlike other workers, this agent focuses on reviewing and validating
    the outputs of other agents rather than generating new designs.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        hardware_constraints: HardwareConstraints,
    ):
        # Design reviewer uses analysis tools, not web/github tools
        tools: List[BaseTool] = []  # Tools are called directly, not via LLM
        super().__init__(
            llm=llm,
            tools=tools,
            agent_name="design_reviewer",
            hardware_constraints=hardware_constraints,
        )

    @property
    def system_prompt(self) -> str:
        return DESIGN_REVIEW_SYSTEM_PROMPT

    def _build_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("system", self.format_constraints_reminder()),
                ("system", DESIGN_REVIEW_OUTPUT_FORMAT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", DESIGN_REVIEW_PROMPT),
            ]
        )

    async def execute(
        self,
        task_context: Dict[str, Any],
        messages: List[BaseMessage],
    ) -> WorkerResult:
        """
        Execute design review.

        Args:
            task_context: Contains 'artifacts_to_review', 'asil_level', 'aspice_level'
            messages: Conversation history for context

        Returns:
            WorkerResult with review findings and design_review result
        """
        artifacts = task_context.get("artifacts_to_review", [])
        asil_level = task_context.get("asil_level", "ASIL_B")
        aspice_level = task_context.get("aspice_level", 2)
        components = task_context.get("components", ["logging_service"])
        focus_areas = task_context.get("focus_areas", "")

        try:
            # Step 1: Validate hardware constraints
            constraint_results = self._validate_constraints(artifacts)

            # Step 2: Cross-reference against ISO 26262
            artifact_types = self._extract_artifact_types(artifacts)
            iso_gaps = cross_reference_requirements.invoke(
                {
                    "artifact_types": artifact_types,
                    "requirement_type": "iso_26262",
                }
            )

            # Step 3: Cross-reference against ASPICE
            aspice_gaps = cross_reference_requirements.invoke(
                {
                    "artifact_types": artifact_types,
                    "requirement_type": "aspice",
                }
            )

            # Step 4: LLM-based qualitative review
            prompt = self._prompt_template
            chain = prompt | self.llm

            llm_review = await chain.ainvoke(
                {
                    "messages": messages,
                    "artifacts": str(artifacts)[:2000],  # Limit size
                    "asil_level": asil_level,
                    "aspice_level": aspice_level,
                    "components": ", ".join(components) if isinstance(components, list) else components,
                    "focus_areas": focus_areas or "Overall design completeness and compliance",
                }
            )

            # Step 5: Compile findings
            findings = self._compile_findings(
                constraint_results, iso_gaps, aspice_gaps
            )

            # Step 6: Determine overall status
            review_result = self._determine_review_status(
                findings, constraint_results, iso_gaps, aspice_gaps
            )

            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                findings, asil_level, aspice_level
            )

            return WorkerResult(
                agent_name=self.agent_name,
                task_type="design_review",
                status=review_result.overall_status,
                findings=findings,
                recommendations=recommendations,
                artifacts={
                    "constraint_validation": constraint_results,
                    "iso_26262_coverage": {
                        "coverage_percent": iso_gaps.get("coverage_percent", 0),
                        "missing": iso_gaps.get("missing", []),
                    },
                    "aspice_coverage": {
                        "coverage_percent": aspice_gaps.get("coverage_percent", 0),
                        "missing": aspice_gaps.get("missing", []),
                    },
                    "llm_review": llm_review.content,
                    "review_result": review_result.model_dump(),
                },
                confidence_score=0.85,
            )

        except Exception as e:
            return self._create_error_result(
                f"Design review failed: {str(e)}",
                "design_review",
            )

    def _validate_constraints(
        self, artifacts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate hardware constraints from artifacts."""
        # Extract performance data from artifacts
        cpu_overhead = None
        bandwidth = None

        for artifact in artifacts:
            if isinstance(artifact, dict):
                perf = artifact.get("performance", {})
                if isinstance(perf, dict):
                    cpu_overhead = cpu_overhead or perf.get("cpu_overhead_percent")
                    bandwidth = bandwidth or perf.get("bandwidth_mbps")

        # Run validation
        result = validate_hardware_constraints.invoke(
            {
                "cpu_overhead_percent": cpu_overhead,
                "bandwidth_mbps": bandwidth,
            }
        )

        return result

    def _extract_artifact_types(self, artifacts: List[Dict[str, Any]]) -> List[str]:
        """Extract artifact types from the artifact list."""
        types = set()

        for artifact in artifacts:
            if isinstance(artifact, dict):
                # Check for common artifact type indicators
                if "full_design" in artifact:
                    types.add("architectural_design")
                    types.add("detailed_design")

                if "full_analysis" in artifact:
                    types.add("safety_mechanisms")

                if "performance" in artifact:
                    types.add("performance_analysis")

                if artifact.get("asil_level"):
                    types.add("safety_goals")

        # Add some defaults for logging service designs
        types.add("requirements_specification")

        return list(types)

    def _compile_findings(
        self,
        constraint_results: Dict[str, Any],
        iso_gaps: Dict[str, Any],
        aspice_gaps: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compile all findings from validation and coverage analysis."""
        findings = []

        # Constraint violations
        for violation in constraint_results.get("violations", []):
            findings.append(
                {
                    "type": "constraint_violation",
                    "severity": "critical",
                    "description": violation,
                }
            )

        # Constraint warnings
        for warning in constraint_results.get("warnings", []):
            findings.append(
                {
                    "type": "constraint_warning",
                    "severity": "medium",
                    "description": warning,
                }
            )

        # ISO 26262 gaps
        for gap in iso_gaps.get("gaps", []):
            findings.append(
                {
                    "type": "iso_26262_gap",
                    "severity": gap.get("severity", "medium"),
                    "description": gap.get("description", f"Missing: {gap.get('requirement')}"),
                }
            )

        # ASPICE gaps
        for gap in aspice_gaps.get("gaps", []):
            findings.append(
                {
                    "type": "aspice_gap",
                    "severity": gap.get("severity", "medium"),
                    "description": gap.get("description", f"Missing: {gap.get('requirement')}"),
                }
            )

        return findings

    def _determine_review_status(
        self,
        findings: List[Dict[str, Any]],
        constraint_results: Dict[str, Any],
        iso_gaps: Dict[str, Any],
        aspice_gaps: Dict[str, Any],
    ) -> DesignReviewResult:
        """Determine overall review status."""
        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")

        # Calculate completeness score
        iso_coverage = iso_gaps.get("coverage_percent", 0)
        aspice_coverage = aspice_gaps.get("coverage_percent", 0)
        constraint_valid = constraint_results.get("valid", True)

        completeness = (iso_coverage + aspice_coverage) / 2
        if not constraint_valid:
            completeness *= 0.5  # Penalize constraint violations

        # Determine status
        if critical_count > 0 or not constraint_valid:
            status = "rejected"
        elif high_count > 2 or completeness < 50:
            status = "needs_revision"
        else:
            status = "approved"

        # Extract gaps and issues
        gaps = [f for f in findings if "gap" in f.get("type", "")]
        constraint_violations = [
            f for f in findings if f.get("type") == "constraint_violation"
        ]
        cross_ref_issues = [
            f for f in findings if f.get("type") in ["iso_26262_gap", "aspice_gap"]
        ]

        return DesignReviewResult(
            overall_status=status,
            completeness_score=completeness,
            gaps_identified=gaps,
            cross_reference_issues=cross_ref_issues,
            constraint_violations=constraint_violations,
            recommendations=[],  # Filled separately
        )

    def _generate_recommendations(
        self,
        findings: List[Dict[str, Any]],
        asil_level: str,
        aspice_level: int,
    ) -> List[str]:
        """Generate prioritized recommendations based on findings."""
        recs = []

        # Address critical items first
        critical_findings = [f for f in findings if f.get("severity") == "critical"]
        if critical_findings:
            recs.append(
                f"CRITICAL: Address {len(critical_findings)} critical issue(s) before proceeding"
            )

        # ISO 26262 gaps
        iso_gaps = [f for f in findings if f.get("type") == "iso_26262_gap"]
        if iso_gaps:
            recs.append(
                f"Complete ISO 26262 artifacts: {len(iso_gaps)} item(s) missing for {asil_level}"
            )

        # ASPICE gaps
        aspice_gaps = [f for f in findings if f.get("type") == "aspice_gap"]
        if aspice_gaps:
            recs.append(
                f"Complete ASPICE work products: {len(aspice_gaps)} item(s) missing for Level {aspice_level}"
            )

        # Standard recommendations
        recs.extend(
            [
                "Establish bidirectional traceability for all requirements",
                "Document performance calculations with evidence",
                "Ensure all safety mechanisms have diagnostic coverage specified",
            ]
        )

        return recs
