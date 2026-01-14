"""
LangGraph workflow definition for Solar-Flare multi-agent system.

This module defines the main workflow graph that orchestrates
the multi-agent logging service design system.
"""

from typing import Dict, Any, Literal, Optional, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage

from solar_flare.graph.state import (
    AgentState,
    HardwareConstraints,
    OrchestratorDecision,
    create_initial_state,
)
from solar_flare.agents.orchestrator import OrchestratorAgent, RequestAnalysis
from solar_flare.agents.iso_26262_analyzer import ISO26262AnalyzerAgent
from solar_flare.agents.embedded_designer import EmbeddedDesignerAgent
from solar_flare.agents.aspice_assessor import ASPICEAssessorAgent
from solar_flare.agents.design_reviewer import DesignReviewAgent
from solar_flare.tools.web_search import tavily_web_search
from solar_flare.tools.url_reader import read_url_content
from solar_flare.tools.github_tools import github_get_file, github_list_directory


def create_workflow(
    llm: BaseChatModel,
    hardware_constraints: Optional[HardwareConstraints] = None,
) -> StateGraph:
    """
    Create the Solar-Flare multi-agent workflow graph.

    Graph Structure:
    ```
    START
      │
      ▼
    [understand_request] ──► Orchestrator analyzes request
      │
      ▼
    [route_to_agent] ──► Conditional routing
      │
      ├──► [iso_26262_analyzer]
      ├──► [embedded_designer]
      ├──► [aspice_assessor]
      └──► [design_reviewer]
                │
                ▼
          [re_route] ◄── Loop until all needed workers complete
                │
                ▼
        [synthesize_response]
                │
                ▼
              END
    ```

    Args:
        llm: Language model for all agents
        hardware_constraints: Hardware constraints (uses defaults if None)

    Returns:
        Configured StateGraph workflow (not compiled)
    """
    if hardware_constraints is None:
        hardware_constraints = HardwareConstraints()

    # Initialize agents
    orchestrator = OrchestratorAgent(llm, hardware_constraints)

    # Workers with their tools
    iso_analyzer = ISO26262AnalyzerAgent(
        llm=llm,
        tools=[tavily_web_search, read_url_content],
        hardware_constraints=hardware_constraints,
    )

    embedded_designer = EmbeddedDesignerAgent(
        llm=llm,
        tools=[tavily_web_search, read_url_content, github_get_file, github_list_directory],
        hardware_constraints=hardware_constraints,
    )

    aspice_assessor = ASPICEAssessorAgent(
        llm=llm,
        tools=[tavily_web_search, read_url_content],
        hardware_constraints=hardware_constraints,
    )

    design_reviewer = DesignReviewAgent(
        llm=llm,
        hardware_constraints=hardware_constraints,
    )

    # Store for closure access
    agents = {
        "orchestrator": orchestrator,
        "iso_26262_analyzer": iso_analyzer,
        "embedded_designer": embedded_designer,
        "aspice_assessor": aspice_assessor,
        "design_reviewer": design_reviewer,
    }

    # Create graph with AgentState
    workflow = StateGraph(AgentState)

    # ============================================================
    # Node Definitions
    # ============================================================

    async def understand_request(state: AgentState) -> Dict[str, Any]:
        """
        Initial node: orchestrator analyzes the user's request.

        Parses the request to determine which workers are needed
        and in what order.
        """
        messages = state.get("messages", [])
        if not messages:
            return {"current_phase": "understanding"}

        # Find the last human message
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break

        if last_human_msg is None:
            return {"current_phase": "understanding"}

        # Analyze the request
        analysis = await orchestrator.analyze_request(last_human_msg.content, messages)

        # Decide first routing
        decision = await orchestrator.decide_next_step(state, analysis)

        return {
            "orchestrator_decision": decision,
            "current_phase": "delegating",
        }

    async def execute_iso_26262(state: AgentState) -> Dict[str, Any]:
        """Execute ISO 26262 compliance analyzer."""
        decision = state.get("orchestrator_decision")
        context = decision.context_for_agent if decision else {}
        messages = state.get("messages", [])

        result = await iso_analyzer.execute(context, messages)

        return {
            "worker_results": [result],
            "current_phase": "executing",
        }

    async def execute_embedded_designer(state: AgentState) -> Dict[str, Any]:
        """Execute embedded architecture designer."""
        decision = state.get("orchestrator_decision")
        context = decision.context_for_agent if decision else {}
        messages = state.get("messages", [])

        result = await embedded_designer.execute(context, messages)

        return {
            "worker_results": [result],
            "current_phase": "executing",
        }

    async def execute_aspice(state: AgentState) -> Dict[str, Any]:
        """Execute ASPICE process assessor."""
        decision = state.get("orchestrator_decision")
        context = decision.context_for_agent if decision else {}
        messages = state.get("messages", [])

        result = await aspice_assessor.execute(context, messages)

        return {
            "worker_results": [result],
            "current_phase": "executing",
        }

    async def execute_design_review(state: AgentState) -> Dict[str, Any]:
        """Execute design review agent."""
        decision = state.get("orchestrator_decision")
        context = decision.context_for_agent if decision else {}
        messages = state.get("messages", [])

        # Provide artifacts from previous workers
        worker_results = state.get("worker_results", [])
        context["artifacts_to_review"] = [r.artifacts for r in worker_results]

        result = await design_reviewer.execute(context, messages)

        # Extract the design review result from artifacts
        review_result = result.artifacts.get("review_result")

        return {
            "worker_results": [result],
            "design_review": review_result,
            "current_phase": "reviewing",
        }

    async def synthesize_response(state: AgentState) -> Dict[str, Any]:
        """Synthesize final response from all worker results."""
        final_response = await orchestrator.synthesize_response(state)

        return {
            "final_response": final_response,
            "current_phase": "complete",
        }

    async def re_route(state: AgentState) -> Dict[str, Any]:
        """
        Re-evaluate routing after a worker completes.

        Determines if more workers need to run or if it's time to synthesize.
        """
        messages = state.get("messages", [])

        # Find the last human message for re-analysis
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break

        if last_human_msg is None:
            return {
                "orchestrator_decision": OrchestratorDecision(
                    next_agent="synthesize",
                    reasoning="No user message found",
                    context_for_agent={},
                ),
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        # Re-analyze and decide next step
        analysis = await orchestrator.analyze_request(last_human_msg.content, messages)
        decision = await orchestrator.decide_next_step(state, analysis)

        return {
            "orchestrator_decision": decision,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # ============================================================
    # Add Nodes to Graph
    # ============================================================

    workflow.add_node("understand_request", understand_request)
    workflow.add_node("iso_26262_analyzer", execute_iso_26262)
    workflow.add_node("embedded_designer", execute_embedded_designer)
    workflow.add_node("aspice_assessor", execute_aspice)
    workflow.add_node("design_reviewer", execute_design_review)
    workflow.add_node("synthesize", synthesize_response)
    workflow.add_node("re_route", re_route)

    # ============================================================
    # Routing Function
    # ============================================================

    def route_to_agent(
        state: AgentState,
    ) -> Literal[
        "iso_26262_analyzer",
        "embedded_designer",
        "aspice_assessor",
        "design_reviewer",
        "synthesize",
        "end",
    ]:
        """Route to the next agent based on orchestrator decision."""
        decision = state.get("orchestrator_decision")

        if decision is None:
            return "end"

        # Check iteration limit
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)

        if iteration_count >= max_iterations:
            return "synthesize"

        # Route based on decision
        next_agent = decision.next_agent

        if next_agent == "end":
            return "end"

        return next_agent

    # ============================================================
    # Add Edges
    # ============================================================

    # Start -> understand_request
    workflow.add_edge(START, "understand_request")

    # understand_request -> routing
    workflow.add_conditional_edges(
        "understand_request",
        route_to_agent,
        {
            "iso_26262_analyzer": "iso_26262_analyzer",
            "embedded_designer": "embedded_designer",
            "aspice_assessor": "aspice_assessor",
            "design_reviewer": "design_reviewer",
            "synthesize": "synthesize",
            "end": END,
        },
    )

    # Worker nodes -> re_route
    workflow.add_edge("iso_26262_analyzer", "re_route")
    workflow.add_edge("embedded_designer", "re_route")
    workflow.add_edge("aspice_assessor", "re_route")
    workflow.add_edge("design_reviewer", "re_route")

    # re_route -> routing
    workflow.add_conditional_edges(
        "re_route",
        route_to_agent,
        {
            "iso_26262_analyzer": "iso_26262_analyzer",
            "embedded_designer": "embedded_designer",
            "aspice_assessor": "aspice_assessor",
            "design_reviewer": "design_reviewer",
            "synthesize": "synthesize",
            "end": END,
        },
    )

    # synthesize -> END
    workflow.add_edge("synthesize", END)

    return workflow


def compile_workflow(
    llm: BaseChatModel,
    hardware_constraints: Optional[HardwareConstraints] = None,
    enable_checkpointing: bool = True,
):
    """
    Compile the workflow with optional checkpointing.

    Args:
        llm: Language model for all agents
        hardware_constraints: Hardware constraints (uses defaults if None)
        enable_checkpointing: Whether to enable state persistence

    Returns:
        Compiled graph ready for execution
    """
    workflow = create_workflow(llm, hardware_constraints)

    if enable_checkpointing:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


async def run_workflow(
    llm: BaseChatModel,
    user_message: str,
    hardware_constraints: Optional[HardwareConstraints] = None,
    session_id: str = "default",
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Convenience function to run a single workflow execution.

    Args:
        llm: Language model for agents
        user_message: User's request
        hardware_constraints: Hardware constraints (uses defaults if None)
        session_id: Session identifier for checkpointing
        max_iterations: Maximum workflow iterations

    Returns:
        Final state with response
    """
    if hardware_constraints is None:
        hardware_constraints = HardwareConstraints()

    # Compile workflow
    app = compile_workflow(llm, hardware_constraints, enable_checkpointing=True)

    # Create initial state
    initial_state = create_initial_state(
        messages=[HumanMessage(content=user_message)],
        hardware_constraints=hardware_constraints,
        max_iterations=max_iterations,
    )

    # Run workflow
    config = {"configurable": {"thread_id": session_id}}
    result = await app.ainvoke(initial_state, config)

    return result
