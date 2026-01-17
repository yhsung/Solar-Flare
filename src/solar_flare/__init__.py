"""
Solar-Flare: LangChain-based Multi-Agent System for ISO 26262/ASPICE Logging Service Design

This package provides an agentic workflow for designing safety-critical logging
services for automotive embedded systems, with built-in compliance checking for
ISO 26262 functional safety and ASPICE process standards.
"""

__version__ = "0.1.0"

from solar_flare.graph.state import (
    AgentState,
    HardwareConstraints,
    ASILLevel,
    CapabilityLevel,
    WorkerResult,
    DesignReviewResult,
)
from solar_flare.graph.workflow import create_workflow, compile_workflow, run_workflow
from solar_flare.llm_providers import (
    create_llm,
    LLMProvider,
    list_providers,
    get_default_model,
)
from solar_flare.markdown_export import (
    export_workflow_results,
    format_workflow_summary,
    format_worker_result_markdown,
    format_design_review_markdown,
)

__all__ = [
    "AgentState",
    "HardwareConstraints",
    "ASILLevel",
    "CapabilityLevel",
    "WorkerResult",
    "DesignReviewResult",
    "create_workflow",
    "compile_workflow",
    "run_workflow",
    "create_llm",
    "LLMProvider",
    "list_providers",
    "get_default_model",
    "export_workflow_results",
    "format_workflow_summary",
    "format_worker_result_markdown",
    "format_design_review_markdown",
]

