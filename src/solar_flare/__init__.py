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

__all__ = [
    "AgentState",
    "HardwareConstraints",
    "ASILLevel",
    "CapabilityLevel",
    "create_workflow",
    "compile_workflow",
]
