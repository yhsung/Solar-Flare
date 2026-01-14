"""
Base agent class for Solar-Flare workers.

This module provides the abstract base class and common functionality
for all worker agents in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

from solar_flare.graph.state import WorkerResult, HardwareConstraints


class BaseWorkerAgent(ABC):
    """
    Abstract base class for all worker agents.

    Provides common functionality for:
    - Tool management
    - Prompt construction with hardware constraints
    - Result formatting
    - Error handling

    Attributes:
        llm: Language model for agent reasoning
        tools: List of tools available to the agent
        agent_name: Unique identifier for the agent
        hardware_constraints: Reference to hardware constraints
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        agent_name: str,
        hardware_constraints: HardwareConstraints,
    ):
        """
        Initialize a worker agent.

        Args:
            llm: Language model for agent reasoning
            tools: List of tools available to this agent
            agent_name: Unique identifier (e.g., "iso_26262_analyzer")
            hardware_constraints: Reference to mandatory hardware constraints
        """
        self.llm = llm
        self.tools = tools
        self.agent_name = agent_name
        self.hardware_constraints = hardware_constraints
        self._prompt_template = self._build_prompt_template()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt for this agent.

        Must be implemented by subclasses to define the agent's
        role, responsibilities, and expertise.
        """
        pass

    @abstractmethod
    def _build_prompt_template(self) -> ChatPromptTemplate:
        """
        Build the prompt template for this agent.

        Must be implemented by subclasses to construct the
        full prompt including system message, constraints, and placeholders.
        """
        pass

    @abstractmethod
    async def execute(
        self,
        task_context: Dict[str, Any],
        messages: List[BaseMessage],
    ) -> WorkerResult:
        """
        Execute the agent's task.

        Args:
            task_context: Context provided by orchestrator including
                         specific task parameters and requirements
            messages: Conversation history for context

        Returns:
            WorkerResult with findings, recommendations, and artifacts
        """
        pass

    def get_tools(self) -> List[BaseTool]:
        """Return the list of tools available to this agent."""
        return self.tools

    def get_tool_names(self) -> List[str]:
        """Return the names of tools available to this agent."""
        return [tool.name for tool in self.tools]

    def format_constraints_reminder(self) -> str:
        """
        Format hardware constraints as a reminder string.

        Returns a formatted string suitable for inclusion in prompts
        to remind the agent of mandatory constraints.
        """
        return self.hardware_constraints.to_reminder_string()

    def _create_error_result(self, error: str, task_type: str) -> WorkerResult:
        """
        Create a WorkerResult for error cases.

        Args:
            error: Error message describing what went wrong
            task_type: Type of task that was being attempted

        Returns:
            WorkerResult with failed status and error information
        """
        return WorkerResult(
            agent_name=self.agent_name,
            task_type=task_type,
            status="failed",
            findings=[{"type": "error", "description": error}],
            recommendations=["Review error and retry with corrected input"],
            artifacts={},
            confidence_score=0.0,
        )

    def _format_findings_as_markdown(self, findings: List[Dict[str, Any]]) -> str:
        """
        Format findings as a markdown string.

        Args:
            findings: List of finding dictionaries

        Returns:
            Markdown-formatted string of findings
        """
        if not findings:
            return "No findings."

        lines = ["## Findings\n"]
        for i, finding in enumerate(findings, 1):
            severity = finding.get("severity", "medium").upper()
            finding_type = finding.get("type", "observation")
            description = finding.get("description", "")

            lines.append(f"### Finding {i}: [{severity}] {finding_type}")
            lines.append(f"{description}\n")

            if "recommendation" in finding:
                lines.append(f"**Recommendation**: {finding['recommendation']}\n")

        return "\n".join(lines)


class AgentRegistry:
    """
    Registry for managing and accessing worker agents.

    Provides a centralized way to register, retrieve, and
    enumerate available agents in the system.
    """

    _agents: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an agent class.

        Args:
            name: Unique name for the agent

        Returns:
            Decorator function
        """

        def decorator(agent_class: type):
            cls._agents[name] = agent_class
            return agent_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """
        Get an agent class by name.

        Args:
            name: Registered agent name

        Returns:
            Agent class

        Raises:
            KeyError: If agent not found
        """
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' not found. Available: {list(cls._agents.keys())}")
        return cls._agents[name]

    @classmethod
    def list_agents(cls) -> List[str]:
        """Return list of registered agent names."""
        return list(cls._agents.keys())

    @classmethod
    def create(
        cls,
        name: str,
        llm: BaseChatModel,
        tools: List[BaseTool],
        hardware_constraints: HardwareConstraints,
    ) -> BaseWorkerAgent:
        """
        Create an agent instance by name.

        Args:
            name: Registered agent name
            llm: Language model
            tools: List of tools
            hardware_constraints: Hardware constraints

        Returns:
            Initialized agent instance
        """
        agent_class = cls.get(name)
        return agent_class(
            llm=llm,
            tools=tools,
            agent_name=name,
            hardware_constraints=hardware_constraints,
        )
