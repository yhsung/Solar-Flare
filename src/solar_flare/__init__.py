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
)
from solar_flare.graph.workflow import create_workflow, compile_workflow
from solar_flare.llm_providers import (
    create_llm,
    LLMProvider,
    list_providers,
    get_default_model,
)

__all__ = [
    "AgentState",
    "HardwareConstraints",
    "ASILLevel",
    "CapabilityLevel",
    "create_workflow",
    "compile_workflow",
    "create_llm",
    "LLMProvider",
    "list_providers",
    "get_default_model",
]
