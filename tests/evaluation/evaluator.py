"""
Evaluation framework for Solar-Flare multi-agent system.

This module provides automated evaluation of agent outputs against
test cases and quality metrics for ISO 26262/ASPICE compliance.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from solar_flare import run_workflow, HardwareConstraints
from solar_flare.graph.state import WorkerResult, DesignReviewResult


class EvaluationMetric(BaseModel):
    """A single evaluation metric."""

    name: str = Field(description="Metric name")
    value: float = Field(description="Metric value (0-1 for scores)")
    description: str = Field(description="Human-readable description")
    passed: bool = Field(description="Whether the metric meets threshold")
    threshold: float = Field(default=0.7, description="Pass threshold")


class EvaluationResult(BaseModel):
    """Result of evaluating a single test case."""

    test_case_id: str = Field(description="Test case identifier")
    test_case_name: str = Field(description="Test case name")
    passed: bool = Field(description="Overall pass/fail")
    metrics: List[EvaluationMetric] = Field(description="Individual metrics")
    agent_invocations: List[str] = Field(description="Agents that were invoked")
    execution_time_seconds: float = Field(description="Execution time")
    timestamp: datetime = Field(default_factory=datetime.now)
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class TestCase(BaseModel):
    """A test case for evaluation."""

    id: str = Field(description="Unique test case identifier")
    name: str = Field(description="Test case name")
    description: str = Field(description="Test case description")
    user_request: str = Field(description="User request to send to workflow")
    expected_agents: List[str] = Field(
        description="Agents that should be invoked"
    )
    expected_asil_level: Optional[str] = Field(
        default=None,
        description="Expected ASIL level in response"
    )
    expected_components: List[str] = Field(
        default_factory=list,
        description="Components that should be designed",
    )
    required_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that must appear in response",
    )
    forbidden_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that must NOT appear in response",
    )
    evaluation_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation criteria",
    )


class AgentEvaluator:
    """
    Automated evaluator for Solar-Flare agent outputs.

    Evaluates:
    - Correct agent invocation
    - Response completeness
    - Hardware constraint compliance
    - ISO 26262 terminology accuracy
    - ASPICE coverage
    """

    def __init__(
        self,
        llm: BaseChatModel,
        test_cases_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            llm: Language model for running workflows
            test_cases_path: Path to test cases JSON file
        """
        self.llm = llm
        self.test_cases: List[TestCase] = []
        self.results: List[EvaluationResult] = []

        if test_cases_path:
            self.load_test_cases(test_cases_path)

    def load_test_cases(self, path: str) -> None:
        """
        Load test cases from a JSON file.

        Args:
            path: Path to test cases JSON file
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.test_cases = [
            TestCase(**tc) for tc in data.get("test_cases", [])
        ]

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case."""
        self.test_cases.append(test_case)

    async def evaluate_single(
        self,
        test_case: TestCase,
        session_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single test case.

        Args:
            test_case: Test case to evaluate
            session_id: Optional session ID for workflow

        Returns:
            EvaluationResult with metrics
        """
        start_time = datetime.now()
        metrics = []
        errors = []
        agent_invocations = []

        try:
            # Run the workflow
            result = await run_workflow(
                llm=self.llm,
                user_message=test_case.user_request,
                session_id=session_id or f"eval-{test_case.id}",
                max_iterations=20,
            )

            # Track which agents were invoked
            if result.get("worker_results"):
                agent_invocations = [
                    w.agent_name for w in result["worker_results"]
                ]

            # Metric 1: Expected agents invoked
            expected_invoked = set(test_case.expected_agents).issubset(
                set(agent_invocations)
            )
            metrics.append(
                EvaluationMetric(
                    name="expected_agents_invoked",
                    value=1.0 if expected_invoked else 0.0,
                    description=f"Expected agents {test_case.expected_agents} were invoked",
                    passed=expected_invoked,
                )
            )

            # Metric 2: Response completeness
            final_response = result.get("final_response", "")
            completeness = self._calculate_completeness(
                final_response,
                test_case.required_keywords,
                test_case.forbidden_keywords,
            )
            metrics.append(
                EvaluationMetric(
                    name="response_completeness",
                    value=completeness,
                    description="Response contains required keywords and avoids forbidden ones",
                    passed=completeness >= 0.8,
                )
            )

            # Metric 3: Hardware constraint validation
            constraint_validation = self._validate_constraints_in_response(
                final_response,
                result.get("design_review"),
            )
            metrics.append(
                EvaluationMetric(
                    name="hardware_constraint_validation",
                    value=constraint_validation,
                    description="Hardware constraints are properly validated",
                    passed=constraint_validation >= 0.9,
                )
            )

            # Metric 4: ISO 26262 terminology
            iso_terminology = self._check_iso_terminology(
                final_response,
                test_case.expected_asil_level,
            )
            metrics.append(
                EvaluationMetric(
                    name="iso_terminology_accuracy",
                    value=iso_terminology,
                    description="ISO 26262 terminology used correctly",
                    passed=iso_terminology >= 0.7,
                )
            )

            # Metric 5: Design review quality
            design_review = result.get("design_review")
            review_quality = self._assess_design_review(design_review)
            metrics.append(
                EvaluationMetric(
                    name="design_review_quality",
                    value=review_quality,
                    description="Design review provides useful feedback",
                    passed=review_quality >= 0.6,
                )
            )

        except Exception as e:
            errors.append(str(e))

        execution_time = (datetime.now() - start_time).total_seconds()

        # Overall pass if most metrics pass
        passed_count = sum(1 for m in metrics if m.passed)
        overall_passed = passed_count >= len(metrics) * 0.6

        return EvaluationResult(
            test_case_id=test_case.id,
            test_case_name=test_case.name,
            passed=overall_passed,
            metrics=metrics,
            agent_invocations=agent_invocations,
            execution_time_seconds=execution_time,
            errors=errors,
        )

    async def evaluate_all(
        self,
        session_prefix: str = "eval",
    ) -> List[EvaluationResult]:
        """
        Evaluate all loaded test cases.

        Args:
            session_prefix: Prefix for session IDs

        Returns:
            List of evaluation results
        """
        results = []

        for test_case in self.test_cases:
            result = await self.evaluate_single(
                test_case,
                session_id=f"{session_prefix}-{test_case.id}",
            )
            results.append(result)

        self.results = results
        return results

    def _calculate_completeness(
        self,
        response: str,
        required_keywords: List[str],
        forbidden_keywords: List[str],
    ) -> float:
        """
        Calculate response completeness score.

        Args:
            response: The response text
            required_keywords: Keywords that must appear
            forbidden_keywords: Keywords that must NOT appear

        Returns:
            Score between 0 and 1
        """
        if not response:
            return 0.0

        response_lower = response.lower()

        # Check required keywords
        required_score = 0.0
        if required_keywords:
            found = sum(
                1 for kw in required_keywords
                if kw.lower() in response_lower
            )
            required_score = found / len(required_keywords)
        else:
            required_score = 1.0

        # Check forbidden keywords
        forbidden_penalty = 0.0
        if forbidden_keywords:
            found = sum(
                1 for kw in forbidden_keywords
                if kw.lower() in response_lower
            )
            forbidden_penalty = found / len(forbidden_keywords)

        return max(0.0, required_score - forbidden_penalty)

    def _validate_constraints_in_response(
        self,
        response: str,
        design_review: Optional[DesignReviewResult],
    ) -> float:
        """
        Validate that hardware constraints are mentioned.

        Args:
            response: The response text
            design_review: Design review result

        Returns:
            Score between 0 and 1
        """
        response_lower = response.lower()

        # Key constraint terms that should appear
        constraint_terms = [
            "mailbox",
            "dma",
            "timer",
            "cpu",
            "bandwidth",
            "ring buffer",
        ]

        mentioned = sum(
            1 for term in constraint_terms
            if term in response_lower
        )

        base_score = mentioned / len(constraint_terms)

        # Bonus if design review validates constraints
        if design_review and not design_review.constraint_violations:
            base_score = min(1.0, base_score + 0.1)

        return base_score

    def _check_iso_terminology(
        self,
        response: str,
        expected_asil: Optional[str],
    ) -> float:
        """
        Check ISO 26262 terminology usage.

        Args:
            response: The response text
            expected_asil: Expected ASIL level

        Returns:
            Score between 0 and 1
        """
        response_lower = response.lower()

        # Key ISO 26262 terms
        iso_terms = [
            "asil",
            "safety goal",
            "functional safety",
            "fmea",
            "fta",
            "diagnostic coverage",
            "fault tolerant",
            "safe state",
        ]

        # Check for general terminology
        terminology_score = sum(
            1 for term in iso_terms
            if term in response_lower
        ) / len(iso_terms)

        # Bonus if correct ASIL mentioned
        if expected_asil and expected_asil.lower() in response_lower:
            terminology_score = min(1.0, terminology_score + 0.1)

        return terminology_score

    def _assess_design_review(
        self,
        design_review: Optional[DesignReviewResult],
    ) -> float:
        """
        Assess the quality of design review output.

        Args:
            design_review: Design review result

        Returns:
            Score between 0 and 1
        """
        if not design_review:
            return 0.5  # Neutral score if no review

        score = 0.0

        # Check status
        if design_review.overall_status in ["approved", "needs_revision"]:
            score += 0.3

        # Check completeness
        if design_review.completeness_score >= 70:
            score += 0.3

        # Check for recommendations
        if design_review.recommendations:
            score += 0.2

        # Check for gap identification
        if design_review.gaps_identified:
            score += 0.2

        return min(1.0, score)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate an evaluation report.

        Returns:
            Dictionary with evaluation summary
        """
        if not self.results:
            return {"error": "No evaluation results available"}

        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        # Aggregate metrics
        metric_scores: Dict[str, List[float]] = {}
        for result in self.results:
            for metric in result.metrics:
                if metric.name not in metric_scores:
                    metric_scores[metric.name] = []
                metric_scores[metric.name].append(metric.value)

        avg_metrics = {
            name: sum(scores) / len(scores)
            for name, scores in metric_scores.items()
        }

        # Agent invocation statistics
        agent_counts: Dict[str, int] = {}
        for result in self.results:
            for agent in result.agent_invocations:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

        return {
            "summary": {
                "total_test_cases": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": passed / total if total > 0 else 0,
            },
            "average_metrics": avg_metrics,
            "agent_invocation_counts": agent_counts,
            "test_results": [
                {
                    "id": r.test_case_id,
                    "name": r.test_case_name,
                    "passed": r.passed,
                    "metrics": {m.name: m.value for m in r.metrics},
                }
                for r in self.results
            ],
            "timestamp": datetime.now().isoformat(),
        }

    def save_report(self, path: str) -> None:
        """
        Save evaluation report to a file.

        Args:
            path: Path to save report
        """
        report = self.generate_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=2)


class CoverageAnalyzer:
    """
    Analyzes test coverage of agent capabilities.

    Identifies gaps in test coverage based on agent features.
    """

    # Required coverage areas for each agent
    AGENT_COVERAGE_AREAS = {
        "iso_26262_analyzer": [
            "ASIL QM analysis",
            "ASIL A analysis",
            "ASIL B analysis",
            "ASIL C analysis",
            "ASIL D analysis",
            "Safety goal validation",
            "Functional safety concept",
            "Hardware safety requirements",
            "Software safety requirements",
        ],
        "embedded_designer": [
            "Ring buffer design",
            "DMA controller design",
            "Mailbox transport design",
            "Multi-core synchronization",
            "Memory layout design",
            "Interrupt handling",
            "Lock-free implementation",
        ],
        "aspice_assessor": [
            "Capability level 0",
            "Capability level 1",
            "Capability level 2",
            "Capability level 3",
            "Capability level 4",
            "Capability level 5",
            "SWE.1 Requirements",
            "SWE.2 Architecture",
            "SWE.3 Detailed Design",
            "SWE.4 Construction",
            "SWE.5 Verification",
        ],
        "design_reviewer": [
            "Hardware constraint validation",
            "ISO 26262 coverage check",
            "ASPICE work product check",
            "Gap identification",
            "Cross-reference validation",
        ],
    }

    def __init__(self, test_cases: List[TestCase]):
        """
        Initialize the coverage analyzer.

        Args:
            test_cases: List of test cases to analyze
        """
        self.test_cases = test_cases

    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Analyze test coverage.

        Returns:
            Dictionary with coverage information
        """
        coverage: Dict[str, Dict[str, bool]] = {}

        # Initialize coverage
        for agent, areas in self.AGENT_COVERAGE_AREAS.items():
            coverage[agent] = {area: False for area in areas}

        # Mark covered areas based on test cases
        for test_case in self.test_cases:
            for agent in test_case.expected_agents:
                if agent in coverage:
                    # Mark areas as covered based on test case content
                    request_lower = test_case.user_request.lower()
                    description_lower = test_case.description.lower()

                    for area in coverage[agent]:
                        area_lower = area.lower()
                        if area_lower in request_lower or area_lower in description_lower:
                            coverage[agent][area] = True

        # Calculate percentages
        coverage_percentages = {}
        for agent, areas in coverage.items():
            covered = sum(1 for v in areas.values() if v)
            total = len(areas)
            coverage_percentages[agent] = covered / total if total > 0 else 0

        return {
            "coverage": coverage,
            "percentages": coverage_percentages,
            "overall_coverage": sum(coverage_percentages.values()) / len(coverage_percentages)
            if coverage_percentages else 0,
        }

    def find_gaps(self) -> List[Dict[str, str]]:
        """
        Find gaps in test coverage.

        Returns:
            List of uncovered areas
        """
        coverage_info = self.analyze_coverage()
        gaps = []

        for agent, areas in coverage_info["coverage"].items():
            for area, covered in areas.items():
                if not covered:
                    gaps.append({
                        "agent": agent,
                        "area": area,
                    })

        return gaps


def create_default_test_cases() -> List[TestCase]:
    """
    Create a default set of test cases.

    Returns:
        List of default test cases
    """
    return [
        TestCase(
            id="tc-001",
            name="Basic Ring Buffer Design",
            description="Design a basic ring buffer for logging",
            user_request="Design a lock-free ring buffer for logging on a Cortex-R5",
            expected_agents=["embedded_designer"],
            expected_components=["ring buffer"],
            required_keywords=["ring buffer", "lock-free", "head", "tail"],
            forbidden_keywords=["malloc", "dynamic allocation"],
        ),
        TestCase(
            id="tc-002",
            name="ASIL-D Compliance Analysis",
            description="Analyze a design for ASIL-D compliance",
            user_request="Analyze this ring buffer design for ISO 26262 ASIL-D compliance",
            expected_agents=["iso_26262_analyzer"],
            expected_asil_level="ASIL_D",
            required_keywords=["ASIL D", "safety", "diagnostic"],
            evaluation_criteria={
                "min_diagnostic_coverage_mentioned": True,
            },
        ),
        TestCase(
            id="tc-003",
            name="DMA Transport Design",
            description="Design DMA-based logging transport",
            user_request="Design a DMA-based logging transport with zero-copy",
            expected_agents=["embedded_designer"],
            expected_components=["DMA", "transport"],
            required_keywords=["DMA", "zero-copy", "interrupt"],
        ),
        TestCase(
            id="tc-004",
            name="ASPICE Level 3 Assessment",
            description="Assess ASPICE capability level 3",
            user_request="Assess the ASPICE capability level for this design process",
            expected_agents=["aspice_assessor"],
            required_keywords=["ASPICE", "capability level", "process"],
            evaluation_criteria={
                "target_capability_level": 3,
            },
        ),
        TestCase(
            id="tc-005",
            name="Complete Design with Review",
            description="Full design workflow with design review",
            user_request="""
            Design a complete logging subsystem for a brake-by-wire system:
            - Safety Level: ASIL-D
            - Platform: Infineon AURIX TC397
            - RTOS: AUTOSAR OS
            - ASPICE Target: Level 3
            """,
            expected_agents=[
                "iso_26262_analyzer",
                "embedded_designer",
                "aspice_assessor",
                "design_reviewer",
            ],
            expected_asil_level="ASIL_D",
            required_keywords=[
                "ring buffer",
                "DMA",
                "ASIL D",
                "safety",
            ],
        ),
        TestCase(
            id="tc-006",
            name="Multi-Core Synchronization",
            description="Design for multi-core synchronization",
            user_request="Design a logging service for multi-core with synchronization",
            expected_agents=["embedded_designer"],
            required_keywords=["multi-core", "synchronization", "atomic", "timer"],
        ),
        TestCase(
            id="tc-007",
            name="Hardware Constraints Validation",
            description="Validate hardware constraints are respected",
            user_request="Design a logging service with strict hardware constraints",
            expected_agents=["embedded_designer", "design_reviewer"],
            required_keywords=["mailbox", "DMA", "CPU", "bandwidth"],
            evaluation_criteria={
                "constraints_validated": True,
            },
        ),
        TestCase(
            id="tc-008",
            name="ASIL-B Design",
            description="Design for ASIL-B requirements",
            user_request="Design a logging buffer for ASIL-B requirements",
            expected_agents=["iso_26262_analyzer", "embedded_designer"],
            expected_asil_level="ASIL_B",
            required_keywords=["ASIL B", "safety"],
        ),
    ]


def save_test_cases(test_cases: List[TestCase], path: str) -> None:
    """
    Save test cases to a JSON file.

    Args:
        test_cases: List of test cases
        path: Path to save
    """
    data = {
        "version": "1.0",
        "test_cases": [tc.model_dump() for tc in test_cases],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def run_evaluation(
    llm: BaseChatModel,
    test_cases_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a complete evaluation.

    Args:
        llm: Language model for evaluation
        test_cases_path: Optional path to test cases file
        output_path: Optional path to save report

    Returns:
        Evaluation report dictionary
    """
    evaluator = AgentEvaluator(llm, test_cases_path)

    # Load default test cases if none provided
    if not test_cases_path:
        default_cases = create_default_test_cases()
        evaluator.test_cases = default_cases

    # Run evaluation
    await evaluator.evaluate_all()

    # Generate and save report
    report = evaluator.generate_report()
    if output_path:
        evaluator.save_report(output_path)

    return report
