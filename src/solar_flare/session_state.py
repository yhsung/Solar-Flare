"""
Session state persistence for multi-turn requirements workflows.

This module provides functionality to persist and manage session state
across multiple runs, enabling:
- Append new iterations to existing history
- Track revisions of previous iterations
- Maintain cumulative traceability matrices
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


class IterationStatus(str, Enum):
    """Status of a requirements iteration."""
    PENDING = "pending"
    COMPLETE = "complete"
    REVISED = "revised"


@dataclass
class TraceEntry:
    """Single entry in the traceability matrix."""
    requirement_id: str
    iteration_id: int
    phase: str
    agents_involved: List[str]
    status: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    revision: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEntry":
        return cls(**data)


@dataclass
class IterationRecord:
    """Record of a single iteration in a session."""
    iteration_id: int
    timestamp: str
    user_message: str
    phase: str
    worker_count: int
    agents_used: List[str]
    response_preview: str
    status: IterationStatus = IterationStatus.COMPLETE
    revision_of: Optional[int] = None
    revision_number: int = 0
    output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        data["status"] = IterationStatus(data.get("status", "complete"))
        return cls(**data)


@dataclass
class RequirementDef:
    """Definition of a requirement being traced."""
    id: str
    title: str
    description: str
    priority: str
    asil_level: str
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementDef":
        return cls(**data)


@dataclass
class SessionState:
    """
    Persistent state for multi-turn requirements sessions.
    
    This class manages the complete state of a requirements clarification
    session, including all iterations, traceability entries, and metadata.
    """
    session_id: str
    created_at: str
    updated_at: str
    requirements: List[RequirementDef] = field(default_factory=list)
    iterations: List[IterationRecord] = field(default_factory=list)
    traceability: List[TraceEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "requirements": [r.to_dict() for r in self.requirements],
            "iterations": [i.to_dict() for i in self.iterations],
            "traceability": [t.to_dict() for t in self.traceability],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            requirements=[RequirementDef.from_dict(r) for r in data.get("requirements", [])],
            iterations=[IterationRecord.from_dict(i) for i in data.get("iterations", [])],
            traceability=[TraceEntry.from_dict(t) for t in data.get("traceability", [])],
            metadata=data.get("metadata", {}),
        )

    def get_next_iteration_id(self) -> int:
        """Get the next iteration ID."""
        if not self.iterations:
            return 1
        return max(i.iteration_id for i in self.iterations) + 1

    def get_iteration(self, iteration_id: int) -> Optional[IterationRecord]:
        """Get an iteration by ID."""
        for iteration in self.iterations:
            if iteration.iteration_id == iteration_id:
                return iteration
        return None

    def get_revision_count(self, iteration_id: int) -> int:
        """Get the number of revisions for an iteration."""
        count = 0
        for iteration in self.iterations:
            if iteration.revision_of == iteration_id:
                count += 1
        return count


def create_session(
    session_id: str,
    requirements: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionState:
    """
    Create a new session state.

    Args:
        session_id: Unique identifier for the session
        requirements: List of requirement definitions
        metadata: Optional metadata for the session

    Returns:
        New SessionState instance
    """
    now = datetime.now().isoformat()
    reqs = []
    if requirements:
        for r in requirements:
            reqs.append(RequirementDef(
                id=r.get("id", ""),
                title=r.get("title", ""),
                description=r.get("description", ""),
                priority=r.get("priority", "medium"),
                asil_level=r.get("asil_level", "QM"),
            ))
    
    return SessionState(
        session_id=session_id,
        created_at=now,
        updated_at=now,
        requirements=reqs,
        iterations=[],
        traceability=[],
        metadata=metadata or {},
    )


def load_session(session_dir: Union[str, Path]) -> Optional[SessionState]:
    """
    Load session state from a directory.

    Args:
        session_dir: Path to the session directory

    Returns:
        SessionState if found, None otherwise
    """
    session_path = Path(session_dir) / "session.json"
    if not session_path.exists():
        return None

    try:
        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionState.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to load session: {e}")
        return None


def save_session(state: SessionState, session_dir: Union[str, Path]) -> Path:
    """
    Save session state to a directory.

    Args:
        state: Session state to save
        session_dir: Path to the session directory

    Returns:
        Path to the saved session file
    """
    session_path = Path(session_dir)
    session_path.mkdir(parents=True, exist_ok=True)
    
    # Update timestamp
    state.updated_at = datetime.now().isoformat()
    
    file_path = session_path / "session.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
    
    return file_path


def append_iteration(
    state: SessionState,
    user_message: str,
    worker_results: List[Any],
    phase: str = "complete",
    output_dir: str = "",
) -> IterationRecord:
    """
    Append a new iteration to the session.

    Args:
        state: Current session state
        user_message: The user's message for this iteration
        worker_results: List of WorkerResult objects
        phase: Current phase of the iteration
        output_dir: Output directory for this iteration

    Returns:
        The new IterationRecord
    """
    iteration_id = state.get_next_iteration_id()
    
    # Extract agent names from worker results
    agents = []
    for result in worker_results:
        if hasattr(result, "agent_name"):
            agents.append(result.agent_name)
        elif isinstance(result, dict):
            agents.append(result.get("agent_name", "unknown"))
    
    # Create response preview (first 200 chars)
    response_preview = ""
    if worker_results:
        first_result = worker_results[0]
        if hasattr(first_result, "artifacts"):
            artifacts = first_result.artifacts
            if artifacts:
                preview = str(list(artifacts.values())[0])[:200]
                response_preview = preview
    
    iteration = IterationRecord(
        iteration_id=iteration_id,
        timestamp=datetime.now().isoformat(),
        user_message=user_message[:500],  # Truncate long messages
        phase=phase,
        worker_count=len(worker_results),
        agents_used=agents,
        response_preview=response_preview,
        status=IterationStatus.COMPLETE,
        output_dir=output_dir,
    )
    
    state.iterations.append(iteration)
    state.updated_at = datetime.now().isoformat()
    
    return iteration


def revise_iteration(
    state: SessionState,
    iteration_id: int,
    user_message: str,
    worker_results: List[Any],
    phase: str = "complete",
    output_dir: str = "",
) -> IterationRecord:
    """
    Create a revision of an existing iteration.

    Args:
        state: Current session state
        iteration_id: ID of the iteration to revise
        user_message: The user's message for this revision
        worker_results: List of WorkerResult objects
        phase: Current phase of the iteration
        output_dir: Output directory for this revision

    Returns:
        The new revision IterationRecord
    """
    # Find the original iteration
    original = state.get_iteration(iteration_id)
    if original:
        original.status = IterationStatus.REVISED
    
    # Get revision count
    revision_number = state.get_revision_count(iteration_id) + 1
    new_iteration_id = state.get_next_iteration_id()
    
    # Extract agent names
    agents = []
    for result in worker_results:
        if hasattr(result, "agent_name"):
            agents.append(result.agent_name)
        elif isinstance(result, dict):
            agents.append(result.get("agent_name", "unknown"))
    
    iteration = IterationRecord(
        iteration_id=new_iteration_id,
        timestamp=datetime.now().isoformat(),
        user_message=user_message[:500],
        phase=phase,
        worker_count=len(worker_results),
        agents_used=agents,
        response_preview="",
        status=IterationStatus.COMPLETE,
        revision_of=iteration_id,
        revision_number=revision_number,
        output_dir=output_dir,
    )
    
    state.iterations.append(iteration)
    state.updated_at = datetime.now().isoformat()
    
    return iteration


def add_trace_entries(
    state: SessionState,
    iteration_id: int,
    requirement_ids: List[str],
    phase: str,
    agents: List[str],
    status: str,
) -> List[TraceEntry]:
    """
    Add traceability entries for requirements.

    Args:
        state: Current session state
        iteration_id: ID of the iteration
        requirement_ids: List of requirement IDs being traced
        phase: Phase of analysis (e.g., "initial_analysis", "clarification")
        agents: List of agent names involved
        status: Status of the requirements (e.g., "analyzed", "clarified")

    Returns:
        List of created TraceEntry objects
    """
    entries = []
    for req_id in requirement_ids:
        entry = TraceEntry(
            requirement_id=req_id,
            iteration_id=iteration_id,
            phase=phase,
            agents_involved=agents,
            status=status,
        )
        state.traceability.append(entry)
        entries.append(entry)
    
    state.updated_at = datetime.now().isoformat()
    return entries


def generate_session_summary(state: SessionState) -> str:
    """
    Generate a markdown summary of the session.

    Args:
        state: Session state to summarize

    Returns:
        Markdown formatted summary
    """
    lines = [
        "# Session Summary",
        "",
        f"**Session ID:** {state.session_id}",
        f"**Created:** {state.created_at}",
        f"**Last Updated:** {state.updated_at}",
        f"**Total Iterations:** {len(state.iterations)}",
        f"**Requirements Tracked:** {len(state.requirements)}",
        "",
    ]

    if state.requirements:
        lines.extend([
            "## Requirements",
            "",
            "| ID | Title | ASIL | Priority |",
            "|----|-------|------|----------|",
        ])
        for req in state.requirements:
            lines.append(f"| {req.id} | {req.title} | {req.asil_level} | {req.priority} |")
        lines.append("")

    if state.iterations:
        lines.extend([
            "## Iterations",
            "",
        ])
        for iteration in state.iterations:
            status_suffix = ""
            if iteration.revision_of:
                status_suffix = f" (revision of #{iteration.revision_of})"
            elif iteration.status == IterationStatus.REVISED:
                status_suffix = " [REVISED]"
            
            lines.append(f"### Iteration {iteration.iteration_id}{status_suffix}")
            lines.append("")
            lines.append(f"- **Timestamp:** {iteration.timestamp}")
            lines.append(f"- **Agents:** {len(iteration.agents_used)}")
            lines.append(f"- **Status:** {iteration.status.value}")
            if iteration.output_dir:
                lines.append(f"- **Output:** [{iteration.output_dir}](./{iteration.output_dir}/)")
            lines.append("")

    return "\n".join(lines)
