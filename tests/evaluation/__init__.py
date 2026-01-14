"""Evaluation framework for Solar-Flare multi-agent system."""

from solar_flare.tests.evaluation.evaluator import (
    EvaluationMetric,
    EvaluationResult,
    TestCase,
    AgentEvaluator,
    CoverageAnalyzer,
    create_default_test_cases,
    save_test_cases,
    run_evaluation,
)

__all__ = [
    "EvaluationMetric",
    "EvaluationResult",
    "TestCase",
    "AgentEvaluator",
    "CoverageAnalyzer",
    "create_default_test_cases",
    "save_test_cases",
    "run_evaluation",
]
